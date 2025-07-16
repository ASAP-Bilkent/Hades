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

def load_and_organize_data(json_path):
    """Load the results JSON file and organize data by dataset, idlg_type, and fusion"""
    with open(json_path, 'r') as f:
        runs = json.load(f)
    
    organized_data = defaultdict(lambda: defaultdict(dict))
    
    for run in runs:
        if 'fidelity_easy_test' not in run or 'fidelity_hard_test' not in run:
            continue
        
        params = parse_cli_args(run['cli_command'])
        if 'dataset' not in params or 'idlg_type' not in params or 'fusion' not in params:
            continue
        
        dataset = params['dataset']
        idlg_type = params['idlg_type']
        fusion = params['fusion']
        
        organized_data[dataset][idlg_type][fusion] = {
            'fidelity_easy': run['fidelity_easy_test'],
            'fidelity_hard': run['fidelity_hard_test']
        }
    
    return organized_data

def generate_latex_table(organized_data):
    """Generate a LaTeX table from the organized data"""
    dataset_names = {
        'bcd': 'Breast Cancer Diagnostic',
        'bco': 'Breast Cancer Original'
    }
    
    datasets = sorted(organized_data.keys())
    
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\small")
    latex_content.append("\\resizebox{\\columnwidth}{!}{%")
    latex_content.append("\\begin{tabular}{|c|c||c|c||c|c||c|c|}")
    latex_content.append("\\hline")
    latex_content.append("\\multicolumn{2}{|c||}{\\textbf{Configuration}} & \\multicolumn{6}{c|}{\\textbf{Fidelity Thresholds (\\%)}} \\\\ \\hline")
    latex_content.append("\\textbf{Dataset} & \\textbf{$|\mathcal{F}_{HE}|$} & \\multicolumn{2}{c||}{\\textbf{Cheat}} & \\multicolumn{2}{c||}{\\textbf{Zero}} & \\multicolumn{2}{c|}{\\textbf{Random}} \\\\ \\hline")
    latex_content.append(" &  & \\textbf{0.01} & \\textbf{0.0001} & \\textbf{0.01} & \\textbf{0.0001} & \\textbf{0.01} & \\textbf{0.0001} \\\\ \\hline")
    
    for dataset in datasets:
        dataset_name = dataset_names.get(dataset, dataset.upper())
        
        fusion_values = set()
        for idlg_type in organized_data[dataset]:
            fusion_values.update(organized_data[dataset][idlg_type].keys())
        fusion_values = sorted(fusion_values)
        
        for i, fusion in enumerate(fusion_values):
            row_parts = []
            
            if i == 0:
                row_parts.append(f"\\multirow{{{len(fusion_values)}}}{{*}}{{{dataset_name}}}")
            else:
                row_parts.append("")
            
            row_parts.append(str(fusion))
            
            for idlg_type in ['cheat', 'zero', 'rand']:
                if idlg_type in organized_data[dataset] and fusion in organized_data[dataset][idlg_type]:
                    data = organized_data[dataset][idlg_type][fusion]
                    row_parts.append(f"{data['fidelity_easy']:.1f}")
                    row_parts.append(f"{data['fidelity_hard']:.1f}")
                else:
                    row_parts.append("-")
                    row_parts.append("-")
            
            if i > 0:
                latex_content.append("\\cline{2-8}")
            latex_content.append(" & ".join(row_parts) + " \\\\")
        
        if dataset != datasets[-1]:
            latex_content.append("\\hline")
    
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("}%")
    
    caption = "iDLG Attack Fidelity Percentages by Dataset, Attack Method, and $|\\mathcal{F}_{HE}|$. "
    caption += "Fidelity measures the percentage of test samples that can be reconstructed with MSE below a given threshold (0.01 or 0.0001). "
    caption += "Higher percentages indicate more successful attacks. "
    caption += "\\textbf{Cheat}: initialization with ground truth data (theoretical maximum recovery); "
    caption += "\\textbf{Zero}: initialization with all zeros (common attack scenario); "
    caption += "\\textbf{Random}: initialization with random values (another common attack scenario). "
    caption += "$|\\mathcal{F}_{HE}|$ controls the number of features selected via PCA to be encypted."
    
    latex_content.append(f"\\caption{{{caption}}}")
    latex_content.append("\\label{tab:fidelity_combined}")
    latex_content.append("\\end{table}")
    
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
    
    output_dir = os.path.join(project_root, "output", "tables")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{os.path.basename(selected_file).replace('.json', '')}_table.tex")
    if selected_file.endswith("runs.json"):
        output_file = os.path.join(output_dir, "exp1_table.tex")
    
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_file}")
    
    print("\nLaTeX Table:")
    print("===========")
    print(latex_table)

if __name__ == "__main__":
    main() 