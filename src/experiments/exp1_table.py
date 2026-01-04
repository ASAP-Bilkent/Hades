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
    
    dataset_match = re.search(r'--dataset=([a-zA-Z0-9]+)', cmd)
    if dataset_match:
        params['dataset'] = dataset_match.group(1)
    
    fusion_match = re.search(r'--fusion=([0-9]+)', cmd)
    if fusion_match:
        params['fusion'] = int(fusion_match.group(1))
    else:
        params['fusion'] = 0
    
    idlg_type_match = re.search(r'--idlg_type=([a-zA-Z0-9]+)', cmd)
    if idlg_type_match:
        params['idlg_type'] = idlg_type_match.group(1)
    
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

def load_and_organize_data(json_path):
    
    with open(json_path, 'r') as f:
        runs = json.load(f)
    
    organized_data = defaultdict(lambda: defaultdict(dict))
    
    for run in runs:
        if 'fidelity_easy_train' not in run or 'fidelity_hard_train' not in run:
            continue
        
        params = parse_cli_args(run['cli_command'])
        if 'dataset' not in params or 'idlg_type' not in params or 'fusion' not in params:
            continue
        
        dataset = params['dataset']
        idlg_type = params['idlg_type']
        fusion = params['fusion']
        
        organized_data[dataset][idlg_type][fusion] = {
            'fidelity_easy': run['fidelity_easy_train'],
            'fidelity_hard': run['fidelity_hard_train']
        }
    
    return organized_data

def generate_latex_table(organized_data):
    
    dataset_names = {
        'bcd': 'Breast Cancer Diagnostic (BCD)',
        'mnist': 'MNIST',
        'svhn': 'SVHN'
    }
    
    fusion_values = [0, 8, 16, 64, 128, 256]
    idlg_types = ['oracle', 'zero', 'rand']
    idlg_display = {
        'oracle': '\\textbf{Oracle}',
        'zero': '\\textbf{Zero}',
        'rand': '\\textbf{Random}'
    }
    thresholds = [
        ('fidelity_easy', 0.01),
        ('fidelity_hard', 0.0001)
    ]
    
    datasets = ['bcd', 'mnist', 'svhn']
    
    latex_content = []
    latex_content.append("\\begin{tabular}{l|c|c|cccccc}")
    latex_content.append("\\toprule")
    latex_content.append("\\textbf{Dataset} & \\textbf{Attack Type} & \\textbf{Fidelity Threshold} & ")
    latex_content.append("\\multicolumn{6}{c}{$\\boldsymbol{|\\mathcal{F}_{HE}|}$} \\\\")
    latex_content.append("\\cmidrule(lr){4-9}")
    header_cols = [f"\\textbf{{{f}}}" for f in fusion_values[:-1]] + [f"\\textbf{{{fusion_values[-1]}+}}"]
    latex_content.append("& & & " + " & ".join(header_cols) + " \\\\")
    latex_content.append("\\midrule")
    
    for dataset in datasets:
        if dataset not in organized_data:
            continue
            
        dataset_name = dataset_names.get(dataset, dataset.upper())
        num_rows = len(idlg_types) * len(thresholds)
        row_count = 0
        
        for idlg_type in idlg_types:
            for threshold_key, threshold_val in thresholds:
                row_parts = []
                
                if row_count == 0:
                    row_parts.append(f"\\multirow{{{num_rows}}}{{*}}{{{dataset_name}}}")
                else:
                    row_parts.append("")
                
                if row_count % 2 == 0:
                    row_parts.append(f"\\multirow{{2}}{{*}}{{{idlg_display[idlg_type]}}}")
                else:
                    row_parts.append("")
                
                row_parts.append(str(threshold_val))
                
                for fusion in fusion_values:
                    if (idlg_type in organized_data[dataset] and 
                        fusion in organized_data[dataset][idlg_type]):
                        data = organized_data[dataset][idlg_type][fusion]
                        value = data[threshold_key]
                        if value is not None:
                            row_parts.append(f"{value:.1f}")
                        else:
                            row_parts.append("NA")
                    else:
                        row_parts.append("NA")
                
                latex_content.append(" & ".join(row_parts) + " \\\\")
                row_count += 1
        
        if dataset != datasets[-1]:
            latex_content.append("\\midrule")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    
    caption = "iDLG Attack Fidelity Percentages by Dataset, Attack Method, Fidelity Threshold, and $|\\mathcal{F}_{HE}|$. "
    caption += "Fidelity measures the percentage of test samples that can be reconstructed with MSE below a given threshold. "
    caption += "Higher percentages indicate more successful attacks. "
    caption += "\\textbf{Oracle}: initialization with ground truth data (establishes upper bound); "
    caption += "\\textbf{Zero}: initialization with all zeros (common attack scenario); "
    caption += "\\textbf{Random}: initialization with random values (another common attack scenario). "
    caption += "$|\\mathcal{F}_{HE}|$ controls the number of features selected via PCA to be encrypted."
    
    latex_content.append(f"\\caption{{{caption}}}")
    latex_content.append("\\label{tab:fidelity_combined}")
    
    return "\n".join(latex_content)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    results_dir = os.path.join(project_root, "data", "results", "exp1")
    print(f"Looking for files in: {results_dir}")
    
    exp1_files = glob.glob(os.path.join(results_dir, "exp1*.json"))
    
    print(f"Found {len(exp1_files)} files")
    
    if not exp1_files:
        print("No exp1 JSON files found. Checking runs.json for exp1 data...")
        general_runs_file = os.path.join(project_root, "data", "results", "runs.json")
        if os.path.exists(general_runs_file):
            print(f"Found general runs file: {general_runs_file}")
            selected_file = general_runs_file
        else:
            print("No suitable files found for exp1 data.")
            return
    else:
        exp1_files.sort(key=extract_timestamp, reverse=True)
    
        print("Available exp1 files (newest first):")
        for i, file in enumerate(exp1_files):
            formatted_name = format_filename(file)
            print(f"[{i}] {formatted_name}")
    
        try:
            selection = int(input(f"\nEnter file number (0-{len(exp1_files) - 1}): "))
            if selection < 0 or selection >= len(exp1_files):
                raise ValueError("Invalid selection")
            selected_file = exp1_files[selection]
        except ValueError:
            print("Invalid selection, using the most recent file.")
            selected_file = exp1_files[0]
    
        print(f"Using file: {format_filename(selected_file)}")
    
    organized_data = load_and_organize_data(selected_file)
    
    if not organized_data:
        print("No exp1 data found in the selected file.")
        return
    
    latex_table = generate_latex_table(organized_data)
    
    full_table = "\\begin{table}[htbp]\n\\centering\n" + latex_table + "\n\\end{table}"
    
    output_dir = os.path.join(project_root, "output", "tables")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{os.path.basename(selected_file).replace('.json', '')}_table.tex")
    if selected_file.endswith("runs.json"):
        output_file = os.path.join(output_dir, "exp1_table.tex")
    
    with open(output_file, 'w') as f:
        f.write(full_table)
    
    print(f"LaTeX table saved to {output_file}")
    
    print("\nLaTeX Table:")
    print("===========")
    print(latex_table)

if __name__ == "__main__":
    main() 