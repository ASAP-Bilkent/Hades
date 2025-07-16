#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import datetime
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

def extract_timestamp(filename):
    """Extract timestamp from filename"""
    match = re.search(r'exp\d+-(\d{10,12})\.json$', os.path.basename(filename))
    if match:
        timestamp_str = match.group(1)
        try:
            timestamp = datetime.strptime(timestamp_str, "%d%m%Y%H%M")
            return timestamp
        except ValueError:
            return datetime.min
    return datetime.fromtimestamp(os.path.getmtime(filename))

def format_filename(file_path):
    """Format the filename for display"""
    filename = os.path.basename(file_path)
    match = re.search(r'(exp\d+)-(\d{2})(\d{2})(\d{4})(\d{2})(\d{2})\.json$', filename)
    if match:
        base, day, month, year, hour, minute = match.groups()
        return f"{base} [{day}-{month}-{year} {hour}:{minute}]"
    return filename

def extract_params_from_command(cmd):
    """Extract parameters from command string"""
    params = {}
    
    dataset_match = re.search(r'--dataset=(\w+)', cmd)
    if dataset_match:
        params['dataset'] = dataset_match.group(1)
    
    if params.get('dataset') == 'synth':
        n_synth_features_match = re.search(r'--n_synth_features=(\d+)', cmd)
        if n_synth_features_match:
            params['n_synth_features'] = int(n_synth_features_match.group(1))
    fusion_match = re.search(r'--fusion=(\d+)', cmd)
    if fusion_match:
        params['fusion'] = int(fusion_match.group(1))
    else:
        dataset_val = params.get('dataset')
        if dataset_val == 'bco':
            params['fusion'] = 9
        elif dataset_val == 'bcd':
            params['fusion'] = 30
        elif dataset_val == 'synth':
            if 'n_synth_features' in params:
                params['fusion'] = params['n_synth_features']
            else:
                params['fusion'] = 64
        elif dataset_val == 'mnist':
            params['fusion'] = 756
    
    dims_match = re.search(r'--dims=(\d+)', cmd)
    if dims_match:
        params['dims'] = int(dims_match.group(1))
    else:
        params['dims'] = None
    
    data_amount_match = re.search(r'--data_amount=(\d+)', cmd)
    if data_amount_match:
        params['data_amount'] = int(data_amount_match.group(1))
    else:
        params['data_amount'] = 1
    
    n_clients_match = re.search(r'--n_clients=(\d+)', cmd)
    if n_clients_match:
        params['n_clients'] = int(n_clients_match.group(1))
    else:
        params['n_clients'] = 1
    
    return params

def generate_latex_table(data):
    """Generate a compact LaTeX table from the data"""
    
    grouped_experiments = defaultdict(lambda: {'times': [], 'params_cache': None})

    for entry in data:
        params = extract_params_from_command(entry['cli_command'])
        
        n_synth_val = None
        if params['dataset'] == 'synth':
            n_synth_val = params.get('n_synth_features')

        experiment_key = (
            params['dataset'],
            params['fusion'],
            params['dims'],
            params['data_amount'],
            params['n_clients'],
            n_synth_val
        )
        
        grouped_experiments[experiment_key]['times'].append(entry['total_train_time_s'])
        if grouped_experiments[experiment_key]['params_cache'] is None:
            grouped_experiments[experiment_key]['params_cache'] = params

    processed_data = []
    for key, group_info in grouped_experiments.items():
        cached_params = group_info['params_cache']
        avg_total_train_time_s = np.mean(group_info['times'])
        std_total_train_time_s = np.std(group_info['times'], ddof=1) if len(group_info['times']) > 1 else 0
        
        print(f"Experiment Group Key: {key}")
        print(f"  Individual Times (s): {group_info['times']}")
        print(f"  Average Time (s): {avg_total_train_time_s:.2f}")
        print(f"  Std Dev Time (s): {std_total_train_time_s:.2f}")
        
        dataset = cached_params['dataset']
        fusion = cached_params['fusion']
        dims = cached_params['dims']
        data_amount = cached_params['data_amount']
        n_clients = cached_params['n_clients']
        
        normalized_time = avg_total_train_time_s
        normalized_std = std_total_train_time_s
        
        processed_data.append({
            'dataset': dataset,
            'fusion': fusion,
            'dims': dims,
            'time': normalized_time,
            'std': normalized_std,
            'n_clients': n_clients
        })
    
    datasets = sorted(set(entry['dataset'] for entry in processed_data))
    
    n_clients_values = sorted(set(entry['n_clients'] for entry in processed_data))
    multiple_clients = len(n_clients_values) > 1
    
    if len(datasets) == 1:
        dataset = datasets[0]
        
        processed_data.sort(key=lambda x: (
            x['n_clients'],
            x['dims'] is not None, 
            x['dims'] if x['dims'] is not None else 0,
            x['fusion']
        ))
        
        client_groups = {}
        for entry in processed_data:
            client = entry['n_clients']
            if client not in client_groups:
                client_groups[client] = []
            client_groups[client].append(entry)
        
        latex = "\\begin{table}[htbp]\n\\centering\n"
        latex += f"\\caption{{Total Training Time (seconds) for {dataset.upper()} Dataset}}\n"
        
        latex += "\\begin{tabular}{cccc}\n"
        latex += "\\hline\n"
        latex += "Clients & Fusion & Dims & Time(s) \\\\\n"
        latex += "\\hline\n"
        
        for n_clients, group in sorted(client_groups.items()):
            no_dims = [e for e in group if e['dims'] is None]
            with_dims = [e for e in group if e['dims'] is not None]
            
            first_row = True
            fusion_groups = {}
            for entry in no_dims:
                if entry['fusion'] not in fusion_groups:
                    fusion_groups[entry['fusion']] = []
                fusion_groups[entry['fusion']].append(entry)
            
            for fusion, entries in sorted(fusion_groups.items()):
                time_str = f"{entries[0]['time']:.2f}" if entries[0]['std'] == 0 else f"{entries[0]['time']:.2f} \\pm {entries[0]['std']:.2f}"
                if first_row:
                    latex += f"{n_clients} & {fusion} & - & {time_str} \\\\\n"
                    first_row = False
                else:
                    latex += f" & {fusion} & - & {time_str} \\\\\n"
            
            if no_dims and with_dims:
                latex += "\\hline\n"
            
            first_dim_row = True
            for entry in with_dims:
                time_str = f"{entry['time']:.2f}" if entry['std'] == 0 else f"{entry['time']:.2f} \\pm {entry['std']:.2f}"
                if first_dim_row and first_row:
                    latex += f"{n_clients} & {entry['fusion']} & {entry['dims']} & {time_str} \\\\\n"
                    first_dim_row = False
                elif first_dim_row:
                    latex += f" & {entry['fusion']} & {entry['dims']} & {time_str} \\\\\n"
                    first_dim_row = False
                else:
                    latex += f" & {entry['fusion']} & {entry['dims']} & {time_str} \\\\\n"
            
            if n_clients != sorted(client_groups.keys())[-1]:
                latex += "\\hline\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += f"\\label{{tab:exp3_{dataset}}}\n"
        latex += "\\end{table}"
        
        return latex
    
    dataset_data = defaultdict(list)
    for entry in processed_data:
        dataset_data[entry['dataset']].append({
            'fusion': entry['fusion'],
            'dims': entry['dims'],
            'time': entry['time'],
            'n_clients': entry['n_clients']
        })
    
    for dataset in dataset_data:
        dataset_data[dataset].sort(key=lambda x: x['n_clients'])
        dataset_data[dataset].sort(key=lambda x: (x['dims'] is not None, x['dims'] if x['dims'] is not None else 0))
        dataset_data[dataset].sort(key=lambda x: x['fusion'])
    
    latex = "\\begin{table}[htbp]\n\\centering\n\\caption{Total Training Time (seconds)}\n"
    
    if multiple_clients:
        latex += "\\begin{tabular}{cccc|ccc}\n"
        latex += "\\hline\n"
        latex += "\\multicolumn{4}{c|}{BCO} & \\multicolumn{3}{c}{BCD} \\\\\n"
        latex += "\\hline\n"
        latex += "Clients & Fusion & Dims & Time(s) & Clients & Dims & Time(s) \\\\\n"
        latex += "\\hline\n"
        
        def process_dataset_entries(dataset, entries, with_dims=False):
            rows = []
            for entry in entries:
                row = []
                time_str = f"{entry['time']:.2f}" if entry['std'] == 0 else f"{entry['time']:.2f} \\pm {entry['std']:.2f}"
                if dataset == 'bco':
                    if with_dims:
                        row.extend([str(entry['n_clients']), str(entry['fusion']), str(entry['dims']), time_str, "", "", ""])
                    else:
                        row.extend([str(entry['n_clients']), str(entry['fusion']), "-", time_str, "", "", ""])
                else:
                    if with_dims:
                        row.extend(["", "", "", "", str(entry['n_clients']), str(entry['dims']), time_str])
                    else:
                        row.extend(["", "", "", "", str(entry['n_clients']), "-", time_str])
                rows.append(" & ".join(row) + " \\\\\n")
            return rows
        
        for dataset in ['bco', 'bcd']:
            entries = [e for e in dataset_data[dataset] if e['dims'] is None]
            rows = process_dataset_entries(dataset, entries, with_dims=False)
            latex += "".join(rows)
        
        latex += "\\hline\n"
        
        for dataset in ['bco', 'bcd']:
            entries = [e for e in dataset_data[dataset] if e['dims'] is not None]
            rows = process_dataset_entries(dataset, entries, with_dims=True)
            latex += "".join(rows)
    else:
        latex += "\\begin{tabular}{ccc|cc}\n"
        latex += "\\hline\n"
        latex += "\\multicolumn{3}{c|}{BCO} & \\multicolumn{2}{c}{BCD} \\\\\n"
        latex += "\\hline\n"
        latex += "Fusion & Dims & Time(s) & Dims & Time(s) \\\\\n"
        latex += "\\hline\n"
        
        for dataset in ['bco', 'bcd']:
            entries = [e for e in dataset_data[dataset] if e['dims'] is None]
            for entry in entries:
                row = []
                time_str = f"{entry['time']:.2f}" if entry['std'] == 0 else f"{entry['time']:.2f} \\pm {entry['std']:.2f}"
                if dataset == 'bco':
                    row.extend([str(entry['fusion']), "-", time_str, "", ""])
                else:
                    row.extend(["", "", "", "-", time_str])
                latex += " & ".join(row) + " \\\\\n"
        
        latex += "\\hline\n"
        
        for dataset in ['bco', 'bcd']:
            entries = [e for e in dataset_data[dataset] if e['dims'] is not None]
            for entry in entries:
                row = []
                time_str = f"{entry['time']:.2f}" if entry['std'] == 0 else f"{entry['time']:.2f} \\pm {entry['std']:.2f}"
                if dataset == 'bco':
                    row.extend([str(entry['fusion']), str(entry['dims']), time_str, "", ""])
                else:
                    row.extend(["", "", "", str(entry['dims']), time_str])
                latex += " & ".join(row) + " \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\label{tab:exp3}\n"
    latex += "\\end{table}"
    
    return latex

def main():
    results_dir = os.path.join(project_root, 'data', 'results', 'exp3')
    exp3_files = glob.glob(os.path.join(results_dir, 'exp3*.json'))
    
    print(f"Looking for files in: {results_dir}")
    print(f"Found {len(exp3_files)} files")
    
    if not exp3_files:
        print("No exp3 JSON files found in the results directory.")
        return

    exp3_files.sort(key=extract_timestamp, reverse=True)

    print("Available exp3 files (newest first):")
    for i, file in enumerate(exp3_files):
        formatted_name = format_filename(file)
        print(f"[{i}] {formatted_name}")

    try:
        selection = int(input(f"\nEnter file number (0-{len(exp3_files) - 1}): "))
        if selection < 0 or selection >= len(exp3_files):
            raise ValueError("Invalid selection")
        selected_file = exp3_files[selection]
    except ValueError:
        print("Invalid selection, using the most recent file.")
        selected_file = exp3_files[0]

    print(f"Using file: {format_filename(selected_file)}")

    with open(selected_file, 'r') as f:
        data = json.load(f)

    latex_table = generate_latex_table(data)
    
    print(latex_table)
    
    output_dir = os.path.join(project_root, 'output', 'tables')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{os.path.basename(selected_file).replace(".json", "")}_table.tex')
    
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_file}")

if __name__ == "__main__":
    main() 