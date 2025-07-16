#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import datetime
from collections import defaultdict

def parse_cli_args(cmd):
    """Extract parameters from command string"""
    params = {}
    
    dataset_match = re.search(r'--dataset=([a-zA-Z0-9]+)', cmd)
    if dataset_match:
        params['dataset'] = dataset_match.group(1)
    
    batch_size_match = re.search(r'--batch_size=([0-9]+)', cmd)
    if batch_size_match:
        params['batch_size'] = int(batch_size_match.group(1))
    
    fusion_match = re.search(r'--fusion=([0-9]+)', cmd)
    if fusion_match:
        params['fusion'] = int(fusion_match.group(1))
    else:
        if params.get('dataset') == 'bco':
            params['fusion'] = 9
        elif params.get('dataset') == 'bcd':
            params['fusion'] = 30
        elif params.get('dataset') == 'mnist':
            params['fusion'] = 784
        else:
            params['fusion'] = None
    
    dims_match = re.search(r'--dims=([0-9]+)', cmd)
    if dims_match:
        params['dims'] = int(dims_match.group(1))
    else:
        params['dims'] = None
    
    return params

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



def process_exp4_data(data):
    processed = []
    
    for run in data:
        params = parse_cli_args(run['cli_command'])
        
        if 'dataset' not in params or 'batch_size' not in params:
            continue
            
        total_time_per_client = run['total_training_time_s']
        one_gi_time_per_client = run['one_gi_time_s']
        
        processed.append({
            'dataset': params['dataset'],
            'dims': params.get('dims', None),
            'batch_size': params['batch_size'],
            'fusion': params['fusion'],
            'one_gi_time': one_gi_time_per_client,
            'total_time': total_time_per_client
        })
    
    return processed

def calculate_gi_load(batch_size):
    """Calculate GI-Load based on batch size"""
    baseline_batch_size = 16
    if batch_size == baseline_batch_size:
        return "X"
    else:
        return f"{baseline_batch_size // batch_size}X"

def generate_latex_table(organized_data):
    """Generate a LaTeX table from the organized data"""
    dataset_info = {
        'bco': {'name': 'BCO', 'n_feat': 9},
        'bcd': {'name': 'BCD', 'n_feat': 30},
        'mnist': {'name': 'MNIST', 'n_feat': 784}
    }
    
    latex_content = []
    latex_content.append("\\begin{table}[ht]")
    latex_content.append("\\centering")
    latex_content.append("\\begin{tabular}{l c c c c c c c}")
    latex_content.append("\\toprule")
    latex_content.append("Dataset & $n_{\\text{feat}}$ & Hidden Layer Size & Batch Size & $|\\mathcal{F}_{HE}|$ & GI-Load & One-GI (s) & Total Training (s)\\\\")
    latex_content.append("\\midrule")
    
    datasets = ['bco', 'bcd', 'mnist']
    
    for dataset in datasets:
        if dataset not in organized_data:
            continue
        
        dataset_name = dataset_info[dataset]['name']
        n_feat = dataset_info[dataset]['n_feat']
        
        data_entries = sorted(organized_data[dataset], key=lambda x: x['batch_size'])
        
        for entry in data_entries:
            batch_size = entry['batch_size']
            fusion = entry['fusion']
            dims = entry['dims']
            one_gi_time = entry['one_gi_time']
            total_training_time = entry['total_training_time']
            gi_load = calculate_gi_load(batch_size)
            
            hidden_layer_size = str(dims) if dims is not None else "-"
            
            latex_content.append(f"{dataset_name} & {n_feat} & {hidden_layer_size} & {batch_size} & {fusion} & {gi_load} & {one_gi_time:.6f} & {total_training_time:.6f}\\\\")
        
        if dataset != 'mnist':
            latex_content.append("\\hline")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    
    caption = "One-GI and Total Training timing analysis for \\textsc{BCO}, \\textsc{BCD}, and \\textsc{MNIST} datasets under different model dimensions, batch sizes, and encrypted feature set ($\\mathcal{F}_{HE}$). "
    caption += "\\emph{GI-Load} is the number of training iterations required to process a fixed amount of data (baseline~$X$). "
    caption += "One-GI is the time it takes for a single global iteration of federated training, and Total Training is the per-client total training time, both in seconds. "
    caption += "To account for GI-Load, we run each configuration for a number of iterations equal to its GI-Load value, using $X = \\frac{100}{16}$."
    
    latex_content.append(f"\\caption{{{caption}}}")
    latex_content.append("\\label{tab:gi_dataset}")
    latex_content.append("\\end{table}")
    
    return "\n".join(latex_content)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    results_dir = os.path.join(project_root, "data", "results", "exp4")
    print(f"Looking for files in: {results_dir}")
    
    exp4_files = glob.glob(os.path.join(results_dir, "exp4*.json"))
    
    print(f"Found {len(exp4_files)} files")
    
    if not exp4_files:
        print("No exp4 JSON files found in the results directory.")
        return
    
    exp4_files.sort(key=extract_timestamp, reverse=True)
    
    print("Available exp4 files (newest first):")
    for i, file in enumerate(exp4_files):
        formatted_name = format_filename(file)
        print(f"[{i}] {formatted_name}")
    
    try:
        selection = int(input(f"\nEnter file number (0-{len(exp4_files) - 1}): "))
        if selection < 0 or selection >= len(exp4_files):
            raise ValueError("Invalid selection")
        selected_file = exp4_files[selection]
    except ValueError:
        print("Invalid selection, using the most recent file.")
        selected_file = exp4_files[0]
    
    print(f"Using file: {format_filename(selected_file)}")
    
    with open(selected_file, 'r') as f:
        raw_data = json.load(f)
    
    processed_data = process_exp4_data(raw_data)
    
    if not processed_data:
        print("No exp4 data found in the selected file.")
        return
    
    organized_data = defaultdict(list)
    for run in processed_data:
        dataset = run['dataset']
        organized_data[dataset].append({
            'batch_size': run['batch_size'],
            'fusion': run['fusion'],
            'dims': run['dims'],
            'one_gi_time': run['one_gi_time'],
            'total_training_time': run['total_time']
        })
    
    latex_table = generate_latex_table(organized_data)
    
    output_dir = os.path.join(project_root, "output", "tables")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%d%m%Y%H%M")
    output_file = os.path.join(output_dir, f"exp4_{timestamp}_table.tex")
    
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to: {output_file}")
    print(f"\nGenerated table preview:\n{latex_table}")

if __name__ == "__main__":
    main() 