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
    
    params = {}
    dataset_feat = {
        'bcd': 30,
        'mnist': 784,
        'svhn': 3072,
    }
    
    dataset_match = re.search(r'--dataset=([a-zA-Z0-9]+)', cmd)
    if dataset_match:
        params['dataset'] = dataset_match.group(1)
    
    batch_size_match = re.search(r'--batch_size=([0-9]+)', cmd)
    if batch_size_match:
        params['batch_size'] = int(batch_size_match.group(1))
    
    hades_match = re.search(r'--hades=([0-9]+)', cmd)
    if hades_match:
        hades_val = int(hades_match.group(1))
        if hades_val == 0:
            params['fusion'] = dataset_feat.get(params.get('dataset'))
        else:
            params['fusion'] = hades_val
    else:
        fusion_match = re.search(r'--fusion=([0-9]+)', cmd)
        if fusion_match:
            params['fusion'] = int(fusion_match.group(1))
        else:
            params['fusion'] = dataset_feat.get(params.get('dataset'))
    
    dims_match = re.search(r'--dims=([0-9,]+)', cmd)
    if dims_match:
        dims_str = dims_match.group(1)
        if "," in dims_str:
            params['dims'] = dims_str
        else:
            params['dims'] = int(dims_str)
    else:
        params['dims'] = None
    
    return params

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
    
    baseline_batch_size = 16
    if batch_size == baseline_batch_size:
        return "X"
    else:
        return f"{baseline_batch_size // batch_size}X"

def calculate_total_training_time(dataset, batch_size, one_gi_time):
    dataset_sizes = {
        'bcd': 390,
        'mnist': 49000,
        'svhn': 73250,
    }
    base = dataset_sizes.get(dataset)
    if base is None:
        return None
    return base * one_gi_time / batch_size

def generate_latex_table(organized_data):
    
    dataset_info = {
        'bcd': {'name': 'BCD', 'n_feat': 30},
        'mnist': {'name': 'MNIST', 'n_feat': 784},
        'svhn': {'name': 'SVHN', 'n_feat': 3072}
    }
    
    latex_content = []
    latex_content.append("\\begin{table}[ht]")
    latex_content.append("\\centering")
    latex_content.append("\\begin{tabular}{l c c c c c c c}")
    latex_content.append("\\toprule")
    latex_content.append("Dataset & $|\\mathcal{F}|$ & Hidden Layer Size & Batch Size & $|\\mathcal{F}_{HE}|$ & GI-Load & One-GI (s) & Total Training (s)\\\\")
    latex_content.append("\\midrule")
    
    datasets = ['bcd', 'mnist', 'svhn']
    
    for dataset in datasets:
        if dataset not in organized_data:
            continue
        
        dataset_name = dataset_info[dataset]['name']
        n_feat = dataset_info[dataset]['n_feat']

        data_entries = sorted(
            organized_data[dataset],
            key=lambda x: (x['batch_size'], -(x['fusion'] if x['fusion'] is not None else 0))
        )
        n_rows = len(data_entries)
        first_dims = data_entries[0]['dims'] if data_entries else None
        hidden_layer_value = str(first_dims) if first_dims is not None else "-"
        
        for row_idx, entry in enumerate(data_entries):
            batch_size = entry['batch_size']
            fusion = entry['fusion']
            dims = entry['dims']
            one_gi_time = entry['one_gi_time']
            total_training_time = calculate_total_training_time(dataset, batch_size, one_gi_time)
            if total_training_time is None:
                total_training_time = entry['total_training_time']
            gi_load = calculate_gi_load(batch_size)

            if row_idx == 0:
                dataset_cell = f"\\multirow{{{n_rows}}}{{*}}{{{dataset_name}}}" if n_rows > 1 else dataset_name
                n_feat_cell = f"\\multirow{{{n_rows}}}{{*}}{{{n_feat}}}" if n_rows > 1 else f"{n_feat}"
                hidden_cell = f"\\multirow{{{n_rows}}}{{*}}{{{hidden_layer_value}}}" if n_rows > 1 else hidden_layer_value
            else:
                dataset_cell = ""
                n_feat_cell = ""
                hidden_cell = ""

            latex_content.append(f"{dataset_cell} & {n_feat_cell} & {hidden_cell} & {batch_size} & {fusion} & {gi_load} & {one_gi_time:.3f} & {total_training_time:.3f}\\\\")
        
        if dataset != 'svhn':
            latex_content.append("\\hline")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    
    caption = ("One-GI and Total Training timing analysis for \\textsc{BCD}, \\textsc{MNIST}, and "
               "\\textsc{SVHN} under different batch sizes and encrypted feature set sizes "
               "($|\\mathcal{F}_{HE}|$). \\emph{GI-Load} denotes the number of global iterations (GI) "
               "required to process a fixed amount of data (baseline~$X$). \\emph{One-GI} is the "
               "wall-clock time for a single global iteration, and \\emph{Total Training} is the "
               "per-client end-to-end training time, both in seconds. For comparability, we run "
               "configurations with GI-Load $=16X$ for 100 iterations and scale the iteration count "
               "linearly for smaller loads as $I_{kX}=100\\cdot k/16$ (e.g., $X=100/16$).")
    
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
