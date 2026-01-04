#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import datetime
from collections import defaultdict

def main():
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    results_dir = os.path.join(project_root, "data", "results", "exp2")
    print(f"Looking for files in: {results_dir}")
    
    exp2_files = glob.glob(os.path.join(results_dir, "exp2*.json"))
    
    print(f"Found {len(exp2_files)} files")
    
    if not exp2_files:
        print("No exp2 JSON files found in the results directory.")
        exit(1)

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

    exp2_files.sort(key=extract_timestamp, reverse=True)

    def format_filename(file_path):
        filename = os.path.basename(file_path)
        match = re.search(r'(exp\d+)-(\d{2})(\d{2})(\d{4})(\d{2})(\d{2})\.json$', filename)
        if match:
            base, day, month, year, hour, minute = match.groups()
            return f"{base} [{day}-{month}-{year} {hour}:{minute}]"
        return filename

    print("Available exp2 files (newest first):")
    for i, file in enumerate(exp2_files):
        formatted_name = format_filename(file)
        print(f"[{i}] {formatted_name}")

    try:
        selection = int(input(f"\nEnter file number (0-{len(exp2_files) - 1}): "))
        if selection < 0 or selection >= len(exp2_files):
            raise ValueError("Invalid selection")
        selected_file = exp2_files[selection]
    except ValueError:
        print("Invalid selection, using the most recent file.")
        selected_file = exp2_files[0]

    print(f"Using file: {format_filename(selected_file)}")

    with open(selected_file, 'r') as f:
        results = json.load(f)
    
    param_by_dataset = extract_parameters(results)
    fusion_params = {dataset: params['fusion'] for dataset, params in param_by_dataset.items()}

    data = []

    columns = [
        'Dataset',
        'CE F_{HE}=0',
        'CE F_{HE}=PARAM',
        'FL F_{HE}=PARAM',
        'FL F_{HE}=PARAM approx.'
    ]

    for dataset in param_by_dataset.keys():
        f_he = fusion_params[dataset]
        row = {
            'Dataset': dataset.upper(),
            'CE F_{HE}=0': None,
            'CE F_{HE}=PARAM': None,
            'FL F_{HE}=PARAM': None,
            'FL F_{HE}=PARAM approx.': None
        }
        
        row['F_{HE}_PARAM'] = f_he
        
        for result in results:
            cmd = result['cli_command']
            if f'--dataset={dataset}' not in cmd:
                continue
            
            acc_test = result['acc_test']
            has_fusion0 = '--fusion=0' in cmd
            has_fusion_param = f'--fusion={f_he}' in cmd
            has_hades_param = f'--hades={f_he}' in cmd
            has_n_clients_10 = '--n_clients=10' in cmd or has_hades_param
            
            if has_fusion0 and '--act_fun=sigmoid' in cmd and '--n_clients=' not in cmd:
                row['CE F_{HE}=0'] = acc_test * 100
            elif has_fusion_param and '--act_fun=sigmoid' in cmd and '--n_clients=' not in cmd:
                row['CE F_{HE}=PARAM'] = acc_test * 100
            elif has_fusion_param and '--act_fun=sigmoid' in cmd and has_n_clients_10:
                row['FL F_{HE}=PARAM'] = acc_test * 100
            elif (has_fusion_param or has_hades_param) and '--act_fun=approx_sigmoid' in cmd and has_n_clients_10:
                row['FL F_{HE}=PARAM approx.'] = acc_test * 100
        
        data.append(row)

    df = pd.DataFrame(data)

    df.set_index('Dataset', inplace=True)

    df = df.round(2)

    df = df.drop(columns=['F_{HE}_PARAM'])

    f_he_values = []
    dataset_order = []
    for dataset in df.index:
        dataset_lower = dataset.lower()
        if dataset_lower in fusion_params:
            f_he_values.append(fusion_params[dataset_lower])
            dataset_order.append(dataset)
    
    f_he_str = ', '.join(f_he_values)
    
    latex_lines = []
    latex_lines.append('\\begin{table}[ht!]')
    latex_lines.append(f'\\caption{{Test accuracies (\\%) for centralized (CE) and federated (FL) training, comparing $\\mathcal{{F}}_{{HE}}$ and sigmoid approximation.  $\\mathcal{{F}}_{{HE}} = 0$ means all data is processed with a fully plaintext model. Bold $\\boldsymbol{{\\mathcal{{F}}_{{HE}}}}$ denotes that a model portion is trained under encryption, and we set $|\\mathcal{{F}}_{{HE}}|=[{f_he_str}]$ for {", ".join(dataset_order)}, respectively.}}')
    latex_lines.append('\\label{tab:exp2_results}')
    latex_lines.append('\\small')
    latex_lines.append('\\begin{tabular}{lcccc}')
    latex_lines.append('\\toprule')

    header = ['Dataset', 'CE- $\\mathcal{F}_{HE}=0$', 'CE- $\\boldsymbol{\\mathcal{F}_{HE}}$',
              'FL- $\\boldsymbol{\\mathcal{F}_{HE}}$', 'FL- $\\boldsymbol{\\mathcal{F}_{HE}}$ approx. (\\sys)']
    latex_lines.append(' & '.join(header) + ' \\\\')
    latex_lines.append('\\midrule')

    for dataset in df.index:
        f_he = fusion_params[dataset.lower()]
        row_values = [dataset]
        
        for col in df.columns:
            val = df.loc[dataset, col]
            if pd.isna(val):
                row_values.append('--')
            else:
                row_values.append(f'{val:.2f}')
        
        latex_lines.append(' & '.join(row_values) + ' \\\\')

    latex_lines.append('\\bottomrule')
    latex_lines.append('\\end{tabular}')
    latex_lines.append('\\end{table}')

    latex_table = '\n'.join(latex_lines)

    output_dir = os.path.join(project_root, 'output', 'tables')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "exp2_table.tex")
    
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_file}")

    print("\nLaTeX Table:")
    print("===========")
    print(latex_table)
    
    generate_parameters_table(project_root, selected_file, param_by_dataset)

def extract_parameters(results):
    
    param_by_dataset = defaultdict(lambda: defaultdict(dict))
    
    for result in results:
        cmd = result['cli_command']
        
        dataset_match = re.search(r'--dataset=(\w+)', cmd)
        if not dataset_match:
            continue
        
        dataset = dataset_match.group(1)
        
        dims_match = re.search(r'--dims=(\w+)', cmd)
        if dims_match:
            param_by_dataset[dataset]['dims'] = dims_match.group(1)
        
        fusion_match = re.search(r'--fusion=(\d+)', cmd)
        hades_match = re.search(r'--hades=(\d+)', cmd)
        fusion_val = None
        if fusion_match and fusion_match.group(1) != '0':
            fusion_val = fusion_match.group(1)
        elif hades_match:
            fusion_val = hades_match.group(1)
        if fusion_val is not None:
            param_by_dataset[dataset]['fusion'] = fusion_val
        
        lr_match = re.search(r'--lrate=(\d+\.\d+)', cmd)
        if lr_match:
            param_by_dataset[dataset]['lrate'] = lr_match.group(1)
        
        clients_match = re.search(r'--n_clients=(\d+)', cmd)
        if clients_match:
            param_by_dataset[dataset]['n_clients'] = clients_match.group(1)
        elif hades_match:

            param_by_dataset[dataset]['n_clients'] = '10'
            
        epochs_match = re.search(r'--epoch(?:s)?=(\d+)', cmd)
        if epochs_match:
            param_by_dataset[dataset]['epochs'] = epochs_match.group(1)
    
    result_dict = {}
    for dataset, params in param_by_dataset.items():
        result_dict[dataset] = {
            'dims': params.get('dims', 'N/A'),
            'fusion': params.get('fusion', '0'),
            'lrate': params.get('lrate', '0.01'),
            'n_clients': params.get('n_clients', '10'),
            'epochs': params.get('epochs', '10')
        }
    
    return result_dict

def generate_parameters_table(project_root, selected_file, param_by_dataset):
    
    
    param_data = []
    
    for dataset, params in param_by_dataset.items():
        row = {
            'Dataset': dataset.upper(),
            'Hidden Layer Size': params['dims'],
            'F_{HE}': params['fusion'],
            'Learning Rate': params['lrate'],
            'Number of Clients (FL)': params['n_clients'],
            'Epochs': params['epochs']
        }
        param_data.append(row)
    
    param_df = pd.DataFrame(param_data)
    param_df.set_index('Dataset', inplace=True)
    
    common_epochs = len(set(row['Epochs'] for row in param_data)) == 1
    common_clients = len(set(row['Number of Clients (FL)'] for row in param_data)) == 1
    
    epoch_value = param_data[0]['Epochs'] if param_data else "10"
    clients_value = param_data[0]['Number of Clients (FL)'] if param_data else "10"
    n_datasets = len(param_data)
    
    latex_lines = []
    latex_lines.append('\\begin{table}[h!]')
    latex_lines.append('\\caption{Experiment 2 Parameters for Each Dataset}')
    latex_lines.append('\\label{tab:exp2_parameters}')
    latex_lines.append('\\resizebox{\\columnwidth}{!}{')
    latex_lines.append('\\begin{tabular}{lccccc}')
    
    latex_lines.append('\\toprule')

    header = ['Dataset', 'Hidden Layer Size', r'$|\mathcal{F}_{HE}|$', 'Learning Rate', 'Number of Clients (FL)', 'Epochs']
    latex_lines.append(' & '.join(header) + ' \\\\')
    latex_lines.append('\\midrule')

    for i, dataset in enumerate(param_df.index):
        row_values = [dataset]
        
        row_values.append(str(param_df.loc[dataset, 'Hidden Layer Size']))
        

        row_values.append(str(param_df.loc[dataset, 'F_{HE}']))
        

        row_values.append(str(param_df.loc[dataset, 'Learning Rate']))
        
        if i == 0 and common_clients:
            row_values.append(f'\\multirow{{{n_datasets}}}{{*}}{{{clients_value}}}')
        elif common_clients:
            row_values.append('')
        else:
            row_values.append(str(param_df.loc[dataset, 'Number of Clients (FL)']))
        
        if i == 0 and common_epochs:
            row_values.append(f'\\multirow{{{n_datasets}}}{{*}}{{{epoch_value}}}')
        elif common_epochs:
            row_values.append('')
        else:
            row_values.append(str(param_df.loc[dataset, 'Epochs']))
        
        latex_lines.append(' & '.join(row_values) + ' \\\\')

    latex_lines.append('\\bottomrule')
    latex_lines.append('\\end{tabular}')
    latex_lines.append('}')
    latex_lines.append('\\end{table}')

    param_latex_table = '\n'.join(latex_lines)

    output_dir = os.path.join(project_root, 'output', 'tables')
    param_output_file = os.path.join(output_dir, "exp2_parameters_table.tex")
    
    with open(param_output_file, 'w') as f:
        f.write(param_latex_table)
    
    print(f"\nParameters LaTeX table saved to {param_output_file}")

    print("\nParameters LaTeX Table:")
    print("=======================")
    print(param_latex_table)

if __name__ == "__main__":
    main() 