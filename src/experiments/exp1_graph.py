import json
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
from collections import defaultdict
import datetime

def parse_cli_args(cmd_str):
    params = {}
    dataset_match = re.search(r'--dataset=(\w+)', cmd_str)
    if dataset_match:
        params['dataset'] = dataset_match.group(1)
    
    fusion_match = re.search(r'--fusion=(\d+)', cmd_str)
    if fusion_match:
        params['fusion'] = int(fusion_match.group(1))
    
    idlg_match = re.search(r'--idlg_type=(\w+)', cmd_str)
    if idlg_match:
        params['idlg_type'] = idlg_match.group(1)
    
    return params

def is_power_of_two(n):
    return n == 0 or n == 1 or (n > 1 and (n & (n-1)) == 0)

def load_and_organize_data(json_path):
    with open(json_path, 'r') as f:
        runs = json.load(f)
    
    run_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for run in runs:
        if 'fidelity_easy_test' not in run or 'fidelity_hard_test' not in run:
            continue
        
        params = parse_cli_args(run['cli_command'])
        if 'dataset' not in params or 'idlg_type' not in params or 'fusion' not in params:
            continue
        
        dataset = params['dataset']
        idlg_type = params['idlg_type']
        fusion = params['fusion']
        
        timestamp = run.get('timestamp', datetime.datetime.now().timestamp())
        
        run_data[dataset][idlg_type][fusion].append({
            'fidelity_easy': run['fidelity_easy_test'],
            'fidelity_hard': run['fidelity_hard_test'],
            'timestamp': timestamp
        })
    
    organized_data = defaultdict(lambda: defaultdict(dict))
    for dataset in run_data:
        for idlg_type in run_data[dataset]:
            for fusion in run_data[dataset][idlg_type]:
                if is_power_of_two(fusion):
                    most_recent = sorted(run_data[dataset][idlg_type][fusion], 
                                        key=lambda x: x['timestamp'])[-1]
                    organized_data[dataset][idlg_type][fusion] = most_recent
    
    return organized_data

def create_fidelity_graphs(organized_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_names = {'bcd': 'Breast Cancer Diagnostic', 'mnist': 'MNIST'}
    idlg_names = {'oracle': 'Oracle Mode', 'zero': 'Zero Mode', 'rand': 'Random Mode'}
    
    graph_info = []
    
    for dataset in organized_data:
        for idlg_type in organized_data[dataset]:
            plt.figure(figsize=(3.5, 2.5))
            
            thresholds = [0.01, 0.0001][::-1]
            threshold_labels = ['0.0001', '0.01']
            
            sorted_fusions = sorted(organized_data[dataset][idlg_type].keys())
            
            fusion_data = {}
            
            for fusion in sorted_fusions:
                run_data = organized_data[dataset][idlg_type][fusion]
                
                fidelity_values = [run_data['fidelity_easy'], run_data['fidelity_hard']][::-1]
                
                fusion_data[fusion] = {
                    'fidelity_easy': run_data['fidelity_easy'],
                    'fidelity_hard': run_data['fidelity_hard']
                }
                
                plt.plot(thresholds, fidelity_values, marker='o', linewidth=1.5, 
                         label=f'$F_{{HE}}={fusion}$')
            
            plt.xlabel('Fidelity Threshold (MSE)', fontsize=8)
            plt.ylabel('\\% of Good Fidelity', fontsize=8)
            plt.title(f'{dataset.upper()}-{idlg_type}', fontsize=9)
            plt.xscale('log')
            plt.xticks(thresholds, threshold_labels, fontsize=7)
            plt.gca().invert_xaxis()
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=7, loc='best')
            plt.tight_layout()
            
            output_file = os.path.join(output_dir, f"{dataset}_{idlg_type}_fidelity.pdf")
            plt.savefig(output_file, bbox_inches='tight')
            plt.close()
            
            graph_info.append({
                'dataset': dataset,
                'dataset_name': dataset_names[dataset],
                'idlg_type': idlg_type,
                'idlg_name': idlg_names[idlg_type],
                'filename': output_file,
                'fusion_data': fusion_data,
                'thresholds': threshold_labels
            })
            
            print(f"Created graph for {dataset} with {idlg_type} iDLG type")
    
    return graph_info

def generate_latex_figure(graph_info, output_dir):
    latex_file = os.path.join(output_dir, "fidelity_figure.tex")
    
    dataset_full_names = {'bcd': 'Breast Cancer Diagnostic'}
    idlg_explanations = {
        'oracle': 'initial data is set to the ground truth (establishes upper bound)',
        'zero': 'initial data is set to zeros (common attack scenario)',
        'rand': 'initial data is set to random values (represents robustness against noise)'
    }
    
    latex_content = []
    latex_content.append("\\begin{figure*}[t]")
    latex_content.append("\\centering")
    
    graph_info.sort(key=lambda x: (x['dataset'], x['idlg_type']))
    
    datasets = sorted(set(g['dataset'] for g in graph_info))
    
    first_dataset = datasets[0]
    first_dataset_graphs = [g for g in graph_info if g['dataset'] == first_dataset]
    first_dataset_graphs.sort(key=lambda x: list(idlg_explanations.keys()).index(x['idlg_type']))
    
    for graph in first_dataset_graphs:
        latex_content.append(f"\\begin{{subfigure}}{{0.3\\textwidth}}")
        latex_content.append("\\centering")
        latex_content.append(f"\\includegraphics[width=\\linewidth]{{{graph['filename']}}}")
        latex_content.append(f"%\\caption{{{graph['dataset_name'][0:3]}. {graph['idlg_name'][0:5]}}}")
        latex_content.append(f"\\label{{fig:fidelity_{graph['dataset']}_{graph['idlg_type']}}}")
        latex_content.append("\\end{subfigure}%")
    
    latex_content.append("")
    latex_content.append("\\vspace{0.1cm}")
    latex_content.append("")
    
    if len(datasets) > 1:
        second_dataset = datasets[1]
        second_dataset_graphs = [g for g in graph_info if g['dataset'] == second_dataset]
        second_dataset_graphs.sort(key=lambda x: list(idlg_explanations.keys()).index(x['idlg_type']))
        
        for graph in second_dataset_graphs:
            latex_content.append(f"\\begin{{subfigure}}{{0.3\\textwidth}}")
            latex_content.append("\\centering")
            latex_content.append(f"\\includegraphics[width=\\linewidth]{{{graph['filename']}}}")
            latex_content.append(f"%\\caption{{{graph['dataset_name'][0:3]}. {graph['idlg_name'][0:5]}}}")
            latex_content.append(f"\\label{{fig:fidelity_{graph['dataset']}_{graph['idlg_type']}}}")
            latex_content.append("\\end{subfigure}%")
    
    caption = "iDLG Attack Fidelity across thresholds for different datasets and attack initialization methods. "
    caption += "Top row: BCD (Breast Cancer Diagnostic) dataset. "
    caption += "Columns from left to right show different attack initialization strategies for the homomorphically processed data: "
    caption += "Oracle (initialized with ground truth), Random (initialized with random values), and Zero (initialized with zeros). "
    caption += "Each line represents a different fusion parameter (F). "
    caption += "Higher fidelity percentages indicate more successful data recovery attacks. "
    caption += "The x-axis shows fidelity thresholds (MSE between original and recovered data), "
    caption += "with lower thresholds representing more precise recovery."
    
    latex_content.append("")
    latex_content.append(f"\\caption{{{caption}}}")
    latex_content.append("\\label{fig:fidelity_overall}")
    latex_content.append("\\end{figure*}")
    

    with open(latex_file, 'w') as f:
        f.write("\n".join(latex_content))
    
    print(f"LaTeX figure code written to {latex_file}")

def generate_latex_tables(graph_info, output_dir):
    latex_file = os.path.join(output_dir, "fidelity_tables.tex")
    
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\small")
    
    latex_content.append("\\resizebox{\\columnwidth}{!}{%")
    
    latex_content.append("\\begin{tabular}{|c|c||c|c||c|c||c|c|}")
    latex_content.append("\\hline")
    latex_content.append("\\multicolumn{2}{|c||}{{\\textbf{Configuration}}} & \\multicolumn{6}{c|}{{\\textbf{Fidelity Thresholds (\\%)}}} \\\\ \\hline")
    latex_content.append("\\textbf{Dataset} & \\textbf{$F_{HE}$} & \\multicolumn{2}{c||}{{\\textbf{Oracle}}} & \\multicolumn{2}{c||}{{\\textbf{Zero}}} & \\multicolumn{2}{c|}{{\\textbf{Random}}} \\\\ \\hline")
    latex_content.append(" &  & \\textbf{0.01} & \\textbf{0.0001} & \\textbf{0.01} & \\textbf{0.0001} & \\textbf{0.01} & \\textbf{0.0001} \\\\ \\hline")
    
    datasets = sorted(set(graph['dataset'] for graph in graph_info))
    dataset_names = {g['dataset']: g['dataset_name'] for g in graph_info}
    
    all_fusions = sorted(set(fusion 
                           for graph in graph_info 
                           for fusion in graph['fusion_data'].keys()))
    
    for dataset in datasets:
        dataset_graphs = [g for g in graph_info if g['dataset'] == dataset]
        dataset_name = dataset_names.get(dataset, "Unknown")
        
        for j, fusion in enumerate(all_fusions):
            row_parts = []
            
            if j == 0:
                row_parts.append(f"\\multirow{{{len(all_fusions)}}}{{*}}{{{dataset_name}}}")
            else:
                row_parts.append("")
            
            row_parts.append(str(fusion))
            
            for idlg_type in ['oracle', 'zero', 'rand']:
                matching_graph = next((g for g in dataset_graphs if g['idlg_type'] == idlg_type), None)
                
                if matching_graph and fusion in matching_graph['fusion_data']:
                    data = matching_graph['fusion_data'][fusion]
                    row_parts.append(f"{data['fidelity_easy']:.1f}")
                    row_parts.append(f"{data['fidelity_hard']:.1f}")
                else:
                    row_parts.append("-")
                    row_parts.append("-")
            
            if j > 0:
                latex_content.append("\\cline{2-8}")
            latex_content.append(" & ".join(row_parts) + " \\\\")
        
        if dataset != datasets[-1]:
            latex_content.append("\\hline")
    
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("}%")
    
    caption = "iDLG Attack Fidelity Percentages by Dataset, Attack Method, and Fusion Parameter. "
    caption += "Fidelity measures the percentage of test samples that can be reconstructed with MSE below a given threshold (0.01 or 0.0001). "
    caption += "Higher percentages indicate more successful attacks. "
    caption += "\\textbf{Oracle}: initialization with ground truth data (establishes upper bound); "
    caption += "\\textbf{Zero}: initialization with all zeros (common attack scenario); "
    caption += "\\textbf{Random}: initialization with random values (tests robustness against noise). "
    caption += "Fusion parameter controls the number of features used in the fusion-based privacy-preserving approach."
    
    latex_content.append("\\caption{" + caption + "}")
    latex_content.append("\\label{tab:fidelity_combined}")
    latex_content.append("\\end{table}")
    
    with open(latex_file, 'w') as f:
        f.write("\n".join(latex_content))
    
    print(f"LaTeX table code written to {latex_file}")

def extract_timestamp(filename):
    match = re.search(r'exp\d+-(\d{10,12})\.json$', os.path.basename(filename))
    if match:
        timestamp_str = match.group(1)
        try:
            timestamp = datetime.datetime.strptime(timestamp_str, "%d%m%Y%H%M")
            return timestamp
        except ValueError:
            return datetime.datetime.min
    return datetime.datetime.fromtimestamp(os.path.getmtime(filename))

def format_filename(file_path):
    filename = os.path.basename(file_path)
    match = re.search(r'(exp\d+)-(\d{2})(\d{2})(\d{4})(\d{2})(\d{2})\.json$', filename)
    if match:
        base, day, month, year, hour, minute = match.groups()
        return f"{base} [{day}-{month}-{year} {hour}:{minute}]"
    return filename

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    parser = argparse.ArgumentParser(description='Generate graphs and LaTeX tables from experiment results')
    parser.add_argument('--input_file', type=str, default=None, help='Base filename for input JSON (without .json extension)')
    args = parser.parse_args()
    
    output_dir = os.path.join(project_root, "output", "figures")
    figure_output_dir = output_dir
    table_output_dir = os.path.join(project_root, "output", "tables")
    
    os.makedirs(figure_output_dir, exist_ok=True)
    os.makedirs(table_output_dir, exist_ok=True)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Output directories: {figure_output_dir}, {table_output_dir}")
    
    results_dir = os.path.join(project_root, 'data', 'results', 'exp1')
    
    selected_file = None
    
    if args.input_file:
        json_path = os.path.join(results_dir, f"{args.input_file}.json")
        if os.path.exists(json_path):
            selected_file = json_path
        else:
            print(f"Error: Specified file {json_path} not found!")
            return
    else:
        exp1_files = glob.glob(os.path.join(results_dir, "exp1*.json"))
        
        if not exp1_files:
            runs_json = os.path.join(results_dir, "runs.json")
            if os.path.exists(runs_json):
                print(f"No exp1 specific files found. Using general runs.json")
                selected_file = runs_json
            else:
                print(f"Error: No experiment files found in {results_dir}!")
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
    
    if selected_file:
        print(f"Using file: {selected_file}")
        
        organized_data = load_and_organize_data(selected_file)
        
        if not organized_data:
            print("No valid experiment data found in the selected file.")
            return
        
        graph_info = create_fidelity_graphs(organized_data, figure_output_dir)
        
        generate_latex_figure(graph_info, table_output_dir)
        generate_latex_tables(graph_info, table_output_dir)
        
        print(f"Visualization completed. Output saved to {output_dir}/")

if __name__ == "__main__":
    main()