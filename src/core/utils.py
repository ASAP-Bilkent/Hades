import numpy as np
import functools
import inspect
import math
import time
import pandas as pd
from collections import defaultdict

function_call_counts = {}
function_call_stack = {}

IGNORE_MSE_LOSS_CALLS = True

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func

VERBOSITY = 0
_call_stats = {}

class GlobalTimer:
    def __init__(self):
        self.timers = defaultdict(float)
        self.counts = defaultdict(int)
        self.stack = []
        self.active_timers = {}
        self.n_clients = 1
        self.num_samples = 0
        self.batch_size = 1
        self.total_iterations = 1
        
    def set_n_clients(self, n_clients):
        self.n_clients = n_clients
        
    def set_batch_info(self, num_samples, batch_size):
        self.num_samples = num_samples
        self.batch_size = batch_size
        
    def set_total_iterations(self, total_iterations):
        self.total_iterations = total_iterations
        
    def start_timer(self, category):
        start_time = time.time()
        if category in self.active_timers:
            self.stack.append((category, self.active_timers[category]))
        self.active_timers[category] = start_time
        
    def end_timer(self, category):
        if category not in self.active_timers:
            return
            
        end_time = time.time()
        elapsed = end_time - self.active_timers[category]
        self.timers[category] += elapsed
        self.counts[category] += 1
        
        if self.stack:
            prev_category, prev_start = self.stack.pop()
            if prev_category == category:
                self.active_timers[category] = prev_start
            else:
                self.stack.append((prev_category, prev_start))
                del self.active_timers[category]
        else:
            del self.active_timers[category]
            
    def get_timing_table(self):
        categories = ['Precompute', 'LocalIter', 'FeedForward', 'BackProp', 'UpdateWeights', 'Combine', 'Bootstrap', 'Total']
        
        data = []
        for cat in categories:
            total_time = self.timers[cat]
            count = self.counts[cat]
            per_client_time = total_time / self.n_clients if self.n_clients > 0 else total_time
            per_client_per_iter_time = total_time / (self.n_clients * self.total_iterations) if (self.n_clients > 0 and self.total_iterations > 0) else total_time
            data.append({
                'Category': cat,
                'Total Time (s)': f"{total_time:.6f}",
                'Per-Client Total Time (s)': f"{per_client_time:.6f}",
                'Per-Client Per-Iteration Time (s)': f"{per_client_per_iter_time:.6f}",
                'Count': count
            })
            
        return data
        
    def print_timing_table(self):
        data = self.get_timing_table()
        
        print("\n" + "="*120)
        print("GLOBAL RUNTIME TRACKING RESULTS")
        print("="*120)
        
        print(f"{'Category':<15} {'Total Time (s)':<15} {'Per-Client Total Time (s)':<25} {'Per-Client Per-Iteration Time (s)':<30} {'Count':<10}")
        print("-"*120)
        
        for row in data:
            print(f"{row['Category']:<15} {row['Total Time (s)']:<15} {row['Per-Client Total Time (s)']:<25} {row['Per-Client Per-Iteration Time (s)']:<30} {row['Count']:<10}")
        
        print("="*120)

global_timer = GlobalTimer()

def timer_context(category):
    class TimerContext:
        def __init__(self, cat):
            self.category = cat
            
        def __enter__(self):
            global_timer.start_timer(self.category)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            global_timer.end_timer(self.category)
            
    return TimerContext(category)

def timed_function(category):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timer_context(category):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def count_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        
        stack = inspect.stack()
        
        if IGNORE_MSE_LOSS_CALLS:
            for frame in stack:
                if frame.function == "mse_loss":
                    return func(*args, **kwargs)
        
        if func_name == "mult":
            mult_types = None
            if len(args) >= 3:
                mult_types = args[2]
            elif 'mult_types' in kwargs:
                mult_types = kwargs['mult_types']
            else:
                mult_types = "CC"
                
            if mult_types == "CC":
                func_name = "multc"
            else:
                func_name = "multpt"
        
        elif func_name == "_rotate_ciphertext" and len(args) >= 4:
            N = args[3]
            logN = int(np.ceil(np.log2(N)))
            if logN < 1:
                return func(*args, **kwargs)
        
        call_path = []
        for i in range(1, min(5, len(stack))):
            frame = stack[i]
            call_path.append(frame.function)
        
        call_path.reverse()
        path_str = "->".join(call_path)
        
        if len(stack) > 1:
            caller_frame = stack[1]
            caller_file = caller_frame.filename
            caller_line = caller_frame.lineno
        else:
            caller_file = "unknown"
            caller_line = 0
            
        caller_info = f"{path_str} ({caller_file}:{caller_line})"
        
        if func_name not in function_call_counts:
            function_call_counts[func_name] = 0
        function_call_counts[func_name] += 1
        
        if func_name not in function_call_stack:
            function_call_stack[func_name] = {}
        if caller_info not in function_call_stack[func_name]:
            function_call_stack[func_name][caller_info] = 0
        function_call_stack[func_name][caller_info] += 1
        
        return func(*args, **kwargs)
    return wrapper

def print_call_stats():
    print("\nFunction call counts:")
    for func_name, count in sorted(function_call_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{func_name}: {count} calls")
        if func_name in function_call_stack:
            print("  Called from:")
            for caller, caller_count in sorted(function_call_stack[func_name].items(), key=lambda x: x[1], reverse=True):
                parts = caller.split(" (")
                if len(parts) > 1:
                    path = parts[0]
                    location = parts[1].rstrip(")")
                    print(f"    - {caller_count} times from {path} ({location})")
                else:
                    print(f"    - {caller_count} times from {caller}")

def set_ignore_mse_loss(value=True):
    """Set whether to ignore function calls from mse_loss
    
    Args:
        value (bool): True to ignore calls from mse_loss, False to count them
    """
    global IGNORE_MSE_LOSS_CALLS
    IGNORE_MSE_LOSS_CALLS = value
    print(f"Function calls from mse_loss will {'be ignored' if value else 'be counted'}")

def get_detailed_shape(arr):
    if isinstance(arr, (list, tuple, np.ndarray)):
        lengths = [get_detailed_shape(sub) for sub in arr]

        if all(isinstance(length, tuple) and length == lengths[0] for length in lengths):
            return (len(arr),) + lengths[0]
        else:
            return (len(arr), lengths)
    else:
        return ()

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def next_power_of_two(n):
    if is_power_of_two(n):
        return n
    return 2**int(np.ceil(np.log2(n)))

def distance_to_next_power_of_two(n):
    if is_power_of_two(n):
        return 0
    return next_power_of_two(n) - n

VERBOSITY = 0
def vprint(*args):
    if VERBOSITY > 0:
        print(*args)

class FlushingStdout:
    def __init__(self, stdout):
        self.stdout = stdout
        
    def write(self, text):
        self.stdout.write(text)
        self.stdout.flush()
        
    def flush(self):
        self.stdout.flush()
        
    def __getattr__(self, attr):
        return getattr(self.stdout, attr)
