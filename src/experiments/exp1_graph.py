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
    """Extract key parameters from command line arguments string"""
    params = {}
    
    # Extract dataset
    dataset_match = re.search(r'--dataset=(\w+)', cmd_str)
    if dataset_match:
        params['dataset'] = dataset_match.group(1)
    
    # Extract fusion parameter
    fusion_match = re.search(r'--fusion=(\d+)', cmd_str)
    if fusion_match:
        params['fusion'] = int(fusion_match.group(1))
    
    # Extract idlg_type
    idlg_match = re.search(r'--idlg_type=(\w+)', cmd_str)
    if idlg_match:
        params['idlg_type'] = idlg_match.group(1)
    
    return params

def is_power_of_two(n):
    """Check if n is a power of 2 or is 0 or 1"""
    return n == 0 or n == 1 or (n > 1 and (n & (n-1)) == 0)

def load_and_organize_data(json_path):
    """Load the results JSON file and organize data by dataset and idlg_type"""
    with open(json_path, 'r') as f:
        runs = json.load(f)
    
    # Organize by dataset -> idlg_type -> fusion -> timestamp
    run_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for run in runs:
        # Skip runs without fidelity metrics
        if 'fidelity_easy_test' not in run or 'fidelity_hard_test' not in run:
            continue
        
        params = parse_cli_args(run['cli_command'])
        if 'dataset' not in params or 'idlg_type' not in params or 'fusion' not in params:
            continue
        
        dataset = params['dataset']
        idlg_type = params['idlg_type']
        fusion = params['fusion']
        
        # Use timestamp if available, otherwise use current time
        timestamp = run.get('timestamp', datetime.datetime.now().timestamp())
        
        # Store run data with timestamp
        run_data[dataset][idlg_type][fusion].append({
            'fidelity_easy': run['fidelity_easy_test'],
            'fidelity_hard': run['fidelity_hard_test'],
            'timestamp': timestamp
        })
    
    # Keep only the most recent result for each combination
    organized_data = defaultdict(lambda: defaultdict(dict))
    for dataset in run_data:
        for idlg_type in run_data[dataset]:
            for fusion in run_data[dataset][idlg_type]:
                if is_power_of_two(fusion):  # Only keep powers of 2
                    # Sort by timestamp and get the most recent
                    most_recent = sorted(run_data[dataset][idlg_type][fusion], 
                                        key=lambda x: x['timestamp'])[-1]
                    organized_data[dataset][idlg_type][fusion] = most_recent
    
    return organized_data

def create_fidelity_graphs(organized_data, output_dir):
    """Create graphs of fidelity metrics for each dataset and idlg_type"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Map for better labels
    dataset_names = {'bcd': 'Breast Cancer Diagnostic', 'bco': 'Breast Cancer Original', 'mnist': 'MNIST'}
    idlg_names = {'cheat': 'Cheat Mode', 'zero': 'Zero Mode', 'rand': 'Random Mode'}
    
    # Track all generated files and data for LaTeX output
    graph_info = []
    
    for dataset in organized_data:
        for idlg_type in organized_data[dataset]:
            # Use a smaller figure size for ACM two-column format
            plt.figure(figsize=(3.5, 2.5))
            
            # X-axis values (fidelity thresholds) - REVERSED ORDER
            thresholds = [0.01, 0.0001][::-1]  # Reverse the array
            threshold_labels = ['0.0001', '0.01']
            
            # Sort fusion values in ascending order
            sorted_fusions = sorted(organized_data[dataset][idlg_type].keys())
            
            # Store data for tables
            fusion_data = {}
            
            # Plot a line for each fusion parameter
            for fusion in sorted_fusions:
                run_data = organized_data[dataset][idlg_type][fusion]
                
                # Get fidelity values for each threshold - REVERSED ORDER to match thresholds
                fidelity_values = [run_data['fidelity_easy'], run_data['fidelity_hard']][::-1]
                
                # Store for table output
                fusion_data[fusion] = {
                    'fidelity_easy': run_data['fidelity_easy'],
                    'fidelity_hard': run_data['fidelity_hard']
                }
                
                # Plot this fusion parameter's line
                plt.plot(thresholds, fidelity_values, marker='o', linewidth=1.5, 
                         label=f'$F_{{HE}}={fusion}$')
            
            plt.xlabel('Fidelity Threshold (MSE)', fontsize=8)
            plt.ylabel('\\% of Good Fidelity', fontsize=8)
            plt.title(f'{dataset.upper()}-{idlg_type}', fontsize=9)
            plt.xscale('log')  # Use log scale for better visualization
            plt.xticks(thresholds, threshold_labels, fontsize=7)  # Reversed labels
            plt.gca().invert_xaxis()  # Invert the x-axis
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=7, loc='best')
            plt.tight_layout()
            
            # Save the figure
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
    """Generate LaTeX code for a single figure with subfigures - using figure* for full width"""
    latex_file = os.path.join(output_dir, "fidelity_figure.tex")
    
    # Map for better labels
    dataset_full_names = {'bcd': 'Breast Cancer Diagnostic', 'bco': 'Breast Cancer Original'}
    idlg_explanations = {
        'cheat': 'initial data is set to the ground truth (shows maximum recovery potential)',
        'zero': 'initial data is set to zeros (common attack scenario)',
        'rand': 'initial data is set to random values (represents robustness against noise)'
    }
    
    # Initialize LaTeX content
    latex_content = []
    latex_content.append("\\begin{figure*}[t]")
    latex_content.append("\\centering")
    
    # Sort graphs by dataset and idlg_type for consistent ordering
    graph_info.sort(key=lambda x: (x['dataset'], x['idlg_type']))
    
    # Group by dataset
    datasets = sorted(set(g['dataset'] for g in graph_info))
    
    # First row - first dataset
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
    
    # Second row - second dataset
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
    
    # Comprehensive caption explaining datasets, attack modes, and interpretation
    caption = "iDLG Attack Fidelity across thresholds for different datasets and attack initialization methods. "
    caption += "Top row: BCD (Breast Cancer Diagnostic) dataset; Bottom row: BCO (Breast Cancer Original) dataset. "
    caption += "Columns from left to right show different attack initialization strategies for the homomorphically processed data: "
    caption += "Cheat (initialized with ground truth), Random (initialized with random values), and Zero (initialized with zeros). "
    caption += "Each line represents a different fusion parameter (F). "
    caption += "Higher fidelity percentages indicate more successful data recovery attacks. "
    caption += "The x-axis shows fidelity thresholds (MSE between original and recovered data), "
    caption += "with lower thresholds representing more precise recovery."
    
    latex_content.append("")
    latex_content.append(f"\\caption{{{caption}}}")
    latex_content.append("\\label{fig:fidelity_overall}")
    latex_content.append("\\end{figure*}")
    
    # Write to file
    with open(latex_file, 'w') as f:
        f.write("\n".join(latex_content))
    
    print(f"LaTeX figure code written to {latex_file}")

def generate_latex_tables(graph_info, output_dir):
    """Generate LaTeX code for a single combined table with all fidelity data"""
    latex_file = os.path.join(output_dir, "fidelity_tables.tex")
    
    # Initialize LaTeX content
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\small")
    
    # Add resizebox to make table fit column width
    latex_content.append("\\resizebox{\\columnwidth}{!}{%")
    
    # Create a complex table structure with dataset/idlg groupings
    latex_content.append("\\begin{tabular}{|c|c||c|c||c|c||c|c|}")
    latex_content.append("\\hline")
    latex_content.append("\\multicolumn{2}{|c||}{{\\textbf{Configuration}}} & \\multicolumn{6}{c|}{{\\textbf{Fidelity Thresholds (\\%)}}} \\\\ \\hline")
    latex_content.append("\\textbf{Dataset} & \\textbf{$F_{HE}$} & \\multicolumn{2}{c||}{{\\textbf{Cheat}}} & \\multicolumn{2}{c||}{{\\textbf{Zero}}} & \\multicolumn{2}{c|}{{\\textbf{Random}}} \\\\ \\hline")
    latex_content.append(" &  & \\textbf{0.01} & \\textbf{0.0001} & \\textbf{0.01} & \\textbf{0.0001} & \\textbf{0.01} & \\textbf{0.0001} \\\\ \\hline")
    
    # Sort data by dataset
    datasets = sorted(set(graph['dataset'] for graph in graph_info))
    dataset_names = {g['dataset']: g['dataset_name'] for g in graph_info}
    
    # Get all possible fusion values
    all_fusions = sorted(set(fusion 
                           for graph in graph_info 
                           for fusion in graph['fusion_data'].keys()))
    
    # Group by dataset
    for dataset in datasets:
        dataset_graphs = [g for g in graph_info if g['dataset'] == dataset]
        dataset_name = dataset_names.get(dataset, "Unknown")
        
        # For each fusion value
        for j, fusion in enumerate(all_fusions):
            row_parts = []
            
            # First column - dataset name (only for first row of each dataset)
            if j == 0:
                row_parts.append(f"\\multirow{{{len(all_fusions)}}}{{*}}{{{dataset_name}}}")
            else:
                row_parts.append("")
            
            # Second column - fusion value
            row_parts.append(str(fusion))
            
            # Find data for each idlg type
            for idlg_type in ['cheat', 'zero', 'rand']:
                matching_graph = next((g for g in dataset_graphs if g['idlg_type'] == idlg_type), None)
                
                if matching_graph and fusion in matching_graph['fusion_data']:
                    data = matching_graph['fusion_data'][fusion]
                    # Format with fixed width to align decimals
                    row_parts.append(f"{data['fidelity_easy']:.1f}")
                    row_parts.append(f"{data['fidelity_hard']:.1f}")
                    print(fusion, data['fidelity_easy'])
                else:
                    row_parts.append("-")
                    row_parts.append("-")
            
            # Add the row to the table
            if j > 0:
                latex_content.append("\\cline{2-8}")
            latex_content.append(" & ".join(row_parts) + " \\\\")
        
        if dataset != datasets[-1]:  # Add horizontal line between datasets
            latex_content.append("\\hline")
    
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("}%")  # Close the resizebox
    
    # Add detailed caption explaining the table contents
    caption = "iDLG Attack Fidelity Percentages by Dataset, Attack Method, and Fusion Parameter. "
    caption += "Fidelity measures the percentage of test samples that can be reconstructed with MSE below a given threshold (0.01 or 0.0001). "
    caption += "Higher percentages indicate more successful attacks. "
    caption += "\\textbf{Cheat}: initialization with ground truth data (theoretical maximum recovery); "
    caption += "\\textbf{Zero}: initialization with all zeros (common attack scenario); "
    caption += "\\textbf{Random}: initialization with random values (tests robustness against noise). "
    caption += "Fusion parameter controls the number of features used in the fusion-based privacy-preserving approach."
    
    latex_content.append("\\caption{" + caption + "}")
    latex_content.append("\\label{tab:fidelity_combined}")
    latex_content.append("\\end{table}")
    
    # Write to file
    with open(latex_file, 'w') as f:
        f.write("\n".join(latex_content))
    
    print(f"LaTeX table code written to {latex_file}")

def extract_timestamp(filename):
    """Extract timestamp from filename"""
    match = re.search(r'exp\d+-(\d{10,12})\.json$', os.path.basename(filename))
    if match:
        timestamp_str = match.group(1)
        try:
            # Parse the ddmmyyyyhhmm format
            timestamp = datetime.datetime.strptime(timestamp_str, "%d%m%Y%H%M")
            return timestamp
        except ValueError:
            # If parsing fails, return a minimum date
            return datetime.datetime.min
    # If no timestamp found, sort by file modification time
    return datetime.datetime.fromtimestamp(os.path.getmtime(filename))

def format_filename(file_path):
    """Format filename for display with timestamp if available"""
    filename = os.path.basename(file_path)
    match = re.search(r'(exp\d+)-(\d{2})(\d{2})(\d{4})(\d{2})(\d{2})\.json$', filename)
    if match:
        base, day, month, year, hour, minute = match.groups()
        return f"{base} [{day}-{month}-{year} {hour}:{minute}]"
    return filename

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate graphs and LaTeX tables from experiment results')
    parser.add_argument('--input_file', type=str, default=None, help='Base filename for input JSON (without .json extension)')
    args = parser.parse_args()
    
    # Set up output directories
    output_dir = os.path.join(project_root, "output", "figures")
    figure_output_dir = output_dir  # For clarity
    table_output_dir = os.path.join(project_root, "output", "tables")
    
    os.makedirs(figure_output_dir, exist_ok=True)
    os.makedirs(table_output_dir, exist_ok=True)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Output directories: {figure_output_dir}, {table_output_dir}")
    
    # Path to the results directory
    results_dir = os.path.join(project_root, 'data', 'results', 'exp1')
    
    selected_file = None
    
    # If input_file argument is provided, use it directly
    if args.input_file:
        json_path = os.path.join(results_dir, f"{args.input_file}.json")
        if os.path.exists(json_path):
            selected_file = json_path
        else:
            print(f"Error: Specified file {json_path} not found!")
            return
    else:
        # Look for exp1 result files
        exp1_files = glob.glob(os.path.join(results_dir, "exp1*.json"))
        
        # If no exp1 files found, check for runs.json
        if not exp1_files:
            runs_json = os.path.join(results_dir, "runs.json")
            if os.path.exists(runs_json):
                print(f"No exp1 specific files found. Using general runs.json")
                selected_file = runs_json
            else:
                print(f"Error: No experiment files found in {results_dir}!")
                return
        else:
            # Sort files by timestamp, newest first
            exp1_files.sort(key=extract_timestamp, reverse=True)
            
            # List the files with their timestamps
            print("Available exp1 files (newest first):")
            for i, file in enumerate(exp1_files):
                formatted_name = format_filename(file)
                print(f"[{i}] {formatted_name}")
            
            # Let user select a file
            try:
                selection = int(input(f"\nEnter file number (0-{len(exp1_files) - 1}): "))
                if selection < 0 or selection >= len(exp1_files):
                    raise ValueError("Invalid selection")
                selected_file = exp1_files[selection]
            except ValueError:
                print("Invalid selection, using the most recent file.")
                selected_file = exp1_files[0]  # First file is already the most recent
    
    if selected_file:
        print(f"Using file: {selected_file}")
        
        # Load and organize data
        organized_data = load_and_organize_data(selected_file)
        
        if not organized_data:
            print("No valid experiment data found in the selected file.")
            return
        
        # Create graphs for each dataset and idlg_type and get graph info
        graph_info = create_fidelity_graphs(organized_data, figure_output_dir)
        
        # Generate LaTeX code for figures and tables
        generate_latex_figure(graph_info, table_output_dir)
        generate_latex_tables(graph_info, table_output_dir)
        
        print(f"Visualization completed. Output saved to {output_dir}/")

if __name__ == "__main__":
    main()