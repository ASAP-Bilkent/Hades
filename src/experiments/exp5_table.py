#!/usr/bin/env python3

import json
import os
import glob
import re
from datetime import datetime
from collections import defaultdict

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

def parse_cli_args(cmd):
    params = {}
    
    dataset_match = re.search(r'--dataset=([a-zA-Z0-9]+)', cmd)
    if dataset_match:
        params['dataset'] = dataset_match.group(1).lower()
    else:
        params['dataset'] = None
    
    single_feat_split_match = re.search(r'--single_feat_split=([0-9]+)', cmd)
    if single_feat_split_match:
        params['single_feat_split'] = int(single_feat_split_match.group(1))
    else:
        pca_ablation_match = re.search(r'--pca_ablation=([0-9]+)', cmd)
        params['single_feat_split'] = int(pca_ablation_match.group(1)) if pca_ablation_match else None
    
    fusion_match = re.search(r'--fusion=([0-9]+)', cmd)
    if fusion_match:
        params['fusion'] = int(fusion_match.group(1))
    else:
        hades_match = re.search(r'--hades=([0-9]+)', cmd)
        params['fusion'] = int(hades_match.group(1)) if hades_match else None
    
    clients_match = re.search(r'--n_clients=([0-9]+)', cmd)
    params['n_clients'] = int(clients_match.group(1)) if clients_match else None
    
    act_fun_match = re.search(r'--act_fun=([a-zA-Z_]+)', cmd)
    params['act_fun'] = act_fun_match.group(1) if act_fun_match else None
    
    return params

def select_results_file(results_dir, prefix):
    print(f"Looking for files in: {results_dir}")
    
    exp_files = glob.glob(os.path.join(results_dir, f"{prefix}*.json"))
    
    print(f"Found {len(exp_files)} files")
    
    if not exp_files:
        print(f"No {prefix} JSON files found in the results directory.")
        return None
    
    exp_files.sort(key=extract_timestamp, reverse=True)
    
    print(f"\nAvailable {prefix} files (newest first):")
    for i, file in enumerate(exp_files):
        formatted_name = format_filename(file)
        print(f"[{i}] {formatted_name}")
    
    try:
        selection = int(input(f"\nEnter file number (0-{len(exp_files) - 1}): "))
        if selection < 0 or selection >= len(exp_files):
            raise ValueError("Invalid selection")
        selected_file = exp_files[selection]
    except (ValueError, EOFError, KeyboardInterrupt):
        print("Invalid selection, using the most recent file.")
        selected_file = exp_files[0]
    
    print(f"\nUsing file: {format_filename(selected_file)}")
    return selected_file

def extract_sys_accuracies(results, fhe_map):
    sys_scores = {}
    sys_priority = {}
    
    for result in results:
        params = parse_cli_args(result['cli_command'])
        dataset = params.get('dataset')
        if dataset not in fhe_map:
            continue
        
        fusion = params.get('fusion')
        if fusion != fhe_map[dataset]:
            continue
        
        act_priority = 0 if params.get('act_fun') == 'approx_sigmoid' else 1
        client_priority = 0 if params.get('n_clients') == 10 else 1
        priority = (act_priority, client_priority)
        
        acc = result.get('acc_test')
        if acc is None:
            continue
        acc_percent = acc * 100
        
        current_priority = sys_priority.get(dataset)
        if current_priority is None or priority < current_priority or (priority == current_priority and acc_percent > sys_scores.get(dataset, 0)):
            sys_priority[dataset] = priority
            sys_scores[dataset] = acc_percent
    
    return sys_scores

def build_pca_vs_sys_table(pca_scores, sys_scores, dataset_order, fhe_values):
    latex_lines = []
    latex_lines.append('\\begin{table}[ht!]')
    latex_lines.append('\\centering')
    latex_lines.append('\\small')
    latex_lines.append('\\begin{tabular}{lcc}')
    latex_lines.append('\\toprule')
    latex_lines.append('Dataset & PCA-Only & \\sys \\\\')
    latex_lines.append('\\midrule')
    
    for dataset in dataset_order:
        dataset_upper = dataset.upper()
        pca_val = pca_scores.get(dataset)
        sys_val = sys_scores.get(dataset)
        pca_str = f'{pca_val:.2f}' if pca_val is not None else '--'
        sys_str = f'{sys_val:.2f}' if sys_val is not None else '--'
        latex_lines.append(f'{dataset_upper} & {pca_str} & {sys_str} \\\\')
    
    latex_lines.append('\\bottomrule')
    latex_lines.append('\\end{tabular}')
    latex_lines.append(f'\\caption{{PCA-Only is a single network trained with $|\\mathcal{{F}}_{{HE}}|$ features while \\sys is a fusion network that encrypts $|\\mathcal{{F}}_{{HE}}|$ features and keeps the rest plaintext. We set $|\\mathcal{{F}}_{{HE}}|=[{", ".join(str(fhe_values[d]) for d in dataset_order)}]$ for BCD, MNIST, respectively.}}')
    latex_lines.append('\\label{tab:pca_vs_sys}')
    latex_lines.append('\\end{table}')
    
    return '\n'.join(latex_lines)

def generate_text_table(data_by_dataset):
    
    
    datasets = sorted(data_by_dataset.keys())
    
    max_dataset_len = max(len(ds.upper()) for ds in datasets) if datasets else 10
    max_sf_len = 15
    
    header = f"{'Dataset':<{max_dataset_len}} | {'SF':<{max_sf_len}} | {'Test Accuracy (%)':<18}"
    separator = "-" * len(header)
    
    lines = []
    lines.append(header)
    lines.append(separator)
    
    for dataset in datasets:
        dataset_upper = dataset.upper()
        sf_results = sorted(data_by_dataset[dataset].items())
        
        for i, (sf, acc) in enumerate(sf_results):
            if i == 0:
                dataset_col = dataset_upper
            else:
                dataset_col = ""
            
            sf_col = str(sf)
            acc_col = f"{acc * 100:.2f}"
            
            line = f"{dataset_col:<{max_dataset_len}} | {sf_col:<{max_sf_len}} | {acc_col:<18}"
            lines.append(line)
    
    return "\n".join(lines)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    output_dir = os.path.join(project_root, "output", "tables")
    os.makedirs(output_dir, exist_ok=True)
    
    fhe_map = {'bcd': 8, 'mnist': 256}
    dataset_order = ['bcd', 'mnist']
    
    results_dir = os.path.join(project_root, "data", "results", "exp5")
    selected_exp5 = select_results_file(results_dir, "exp5")
    if not selected_exp5:
        return
    
    with open(selected_exp5, 'r') as f:
        results = json.load(f)
    
    data_by_dataset = defaultdict(dict)
    
    for result in results:
        params = parse_cli_args(result['cli_command'])
        
        if 'dataset' not in params or params['single_feat_split'] is None:
            continue
        
        dataset = params['dataset']
        sf = params['single_feat_split']
        acc_test = result['acc_test']
        
        data_by_dataset[dataset][sf] = acc_test
    
    if not data_by_dataset:
        print("No valid data found in the selected file.")
        return
    
    text_table = generate_text_table(data_by_dataset)
    
    print("\n" + "=" * 60)
    print("Experiment 5 Results - Test Accuracy by Dataset and Selected Features")
    print("=" * 60)
    print()
    print(text_table)
    print()
    
    pca_scores = {}
    for dataset, fhe in fhe_map.items():
        acc = data_by_dataset.get(dataset, {}).get(fhe)
        if acc is not None:
            pca_scores[dataset] = acc * 100
    
    exp2_dir = os.path.join(project_root, "data", "results", "exp2")
    selected_exp2 = select_results_file(exp2_dir, "exp2")
    if not selected_exp2:
        return
    
    with open(selected_exp2, 'r') as f:
        exp2_results = json.load(f)
    
    sys_scores = extract_sys_accuracies(exp2_results, fhe_map)
    
    latex_table = build_pca_vs_sys_table(pca_scores, sys_scores, dataset_order, fhe_map)
    
    output_file = os.path.join(output_dir, "latex5_pca_vs_sys.tex")
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print("PCA-Only vs \\sys LaTeX Table:")
    print("=============================")
    print(latex_table)
    print(f"\nLaTeX table saved to {output_file}")

if __name__ == "__main__":
    main()
