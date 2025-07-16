#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

def extract_timestamp(filename):
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
    filename = os.path.basename(file_path)
    match = re.search(r'(exp\d+)-(\d{2})(\d{2})(\d{4})(\d{2})(\d{2})\.json$', filename)
    if match:
        base, day, month, year, hour, minute = match.groups()
        return f"{base} [{day}-{month}-{year} {hour}:{minute}]"
    return filename

def extract_params_from_command(cmd):
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
    
    dims_match = re.search(r'--dims=([0-9,]+)', cmd)
    if dims_match:
        dims_str = dims_match.group(1)
        if ',' in dims_str:
            layers = dims_str.split(',')
            params['dims'] = dims_str
            params['n_layers'] = len(layers)
            params['layer_size'] = int(layers[0])
        else:
            params['dims'] = int(dims_str)
            params['n_layers'] = 1
            params['layer_size'] = int(dims_str)
    else:
        params['dims'] = None
        params['n_layers'] = 0
        params['layer_size'] = None
    
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

def organize_data(data):
    grouped_experiments = defaultdict(lambda: {'times': [], 'params_cache': None})

    for entry in data:
        params = extract_params_from_command(entry['cli_command'])
        
        n_synth_val = None
        if params['dataset'] == 'synth':
            n_synth_val = params.get('n_synth_features')

        experiment_key = (
            params['dataset'],
            params['fusion'],
            params['n_layers'],
            params['layer_size'],
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
        avg_time = np.mean(group_info['times'])
        std_time = np.std(group_info['times'], ddof=1) if len(group_info['times']) > 1 else 0
        
        processed_data.append({
            'dataset': cached_params['dataset'],
            'fusion': cached_params['fusion'],
            'n_layers': cached_params['n_layers'],
            'layer_size': cached_params['layer_size'],
            'time': avg_time,
            'std': std_time,
            'n_clients': cached_params['n_clients'],
            'data_amount': cached_params['data_amount']
        })
    
    return processed_data

def create_fhe_vs_time_plot(processed_data, output_dir):
    plt.figure(figsize=(10, 6))
    
    layer_groups = defaultdict(list)
    for entry in processed_data:
        layer_groups[entry['n_layers']].append(entry)
    
    if not layer_groups:
        print("No data found")
        return None
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    for i, (n_layers, entries) in enumerate(sorted(layer_groups.items())):
        entries.sort(key=lambda x: x['fusion'])
        
        fusions = np.array([e['fusion'] for e in entries])
        times = np.array([e['time'] for e in entries])
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        if n_layers == 0:
            label = 'No Hidden Layer'
        elif n_layers == 1:
            label = f'1 Hidden Layer'
        else:
            label = f'{n_layers} Hidden Layers'
        
        plt.plot(fusions, times, label=label, 
                marker=marker, color=color, linewidth=2.5, markersize=8)
    
    plt.xscale('log', base=2)
    all_fusions = sorted(set(e['fusion'] for e in processed_data))
    plt.xticks(all_fusions, all_fusions)
    
    plt.xlabel('Number of Encrypted Features ($F_{HE}$)', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Training Time vs Number of Encrypted Features for Different Network Architectures', fontsize=14)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "exp32_fhe_vs_time.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_latex_figure(fhe_plot, output_dir):
    latex_file = os.path.join(output_dir, "exp32_figures.tex")
    
    latex_content = []
    latex_content.append("\\begin{figure}[t]")
    latex_content.append("\\centering")
    
    if fhe_plot:
        latex_content.append(f"\\includegraphics[width=0.8\\linewidth]{{{fhe_plot}}}")
    
    caption = "Training time analysis showing the impact of number of encrypted features ($F_{HE}$) "
    caption += "for different network architectures. Each line represents a different number of hidden layers, "
    caption += "with each hidden layer having 16 neurons."
    
    latex_content.append(f"\\caption{{{caption}}}")
    latex_content.append("\\label{fig:exp32_performance}")
    latex_content.append("\\end{figure}")
    
    with open(latex_file, 'w') as f:
        f.write("\n".join(latex_content))
    
    print(f"LaTeX figure code written to {latex_file}")

def main():
    results_dir = os.path.join(project_root, 'data', 'results', 'exp32')
    exp32_files = glob.glob(os.path.join(results_dir, 'exp32*.json'))
    
    print(f"Looking for files in: {results_dir}")
    print(f"Found {len(exp32_files)} files")
    
    if not exp32_files:
        print("No exp32 JSON files found in the results directory.")
        return

    exp32_files.sort(key=extract_timestamp, reverse=True)

    print("Available exp32 files (newest first):")
    for i, file in enumerate(exp32_files):
        formatted_name = format_filename(file)
        print(f"[{i}] {formatted_name}")

    try:
        selection = int(input(f"\nEnter file number (0-{len(exp32_files) - 1}): "))
        if selection < 0 or selection >= len(exp32_files):
            raise ValueError("Invalid selection")
        selected_file = exp32_files[selection]
    except ValueError:
        print("Invalid selection, using the most recent file.")
        selected_file = exp32_files[0]

    print(f"Using file: {format_filename(selected_file)}")

    with open(selected_file, 'r') as f:
        data = json.load(f)

    processed_data = organize_data(data)
    
    output_dir = os.path.join(project_root, 'output', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    fhe_plot = create_fhe_vs_time_plot(processed_data, output_dir)
    
    if fhe_plot:
        print(f"F_HE vs Time plot saved to: {fhe_plot}")
    
    generate_latex_figure(fhe_plot, output_dir)

if __name__ == "__main__":
    main() 