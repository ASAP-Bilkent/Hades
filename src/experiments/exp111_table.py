#!/usr/bin/env python3

import json
import numpy as np
import os
import glob
import re
from datetime import datetime
from collections import defaultdict


def parse_cli_args(cmd):
    
    params = {}

    dataset_match = re.search(r'--dataset[= ](\w+)', cmd)
    if dataset_match:
        params['dataset'] = dataset_match.group(1)

    fusion_match = re.search(r'--fusion[= ](\d+)', cmd)
    if fusion_match:
        params['fusion'] = int(fusion_match.group(1))
    else:
        params['fusion'] = 0

    idlg_type_match = re.search(r'--idlg_type[= ](\w+)', cmd)
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
        if 'avg_psnr' not in run or 'avg_ssim' not in run or 'avg_lpips' not in run:
            continue

        params = parse_cli_args(run['cli_command'])
        if 'dataset' not in params or 'idlg_type' not in params or 'fusion' not in params:
            continue

        dataset = params['dataset']
        idlg_type = params['idlg_type']
        fusion = params['fusion']

        organized_data[dataset][idlg_type][fusion] = {
            'rmse': run.get('avg_rmse'),
            'psnr': run['avg_psnr'],
            'ssim': run['avg_ssim'],
            'lpips': run['avg_lpips']
        }

    return organized_data


def generate_latex_table(organized_data):
    
    dataset_names = {
        'bcd': 'BCD',
        'mnist': 'MNIST',
        'svhn': 'SVHN',
    }

    fusion_options = {
        'bcd': [0, 16],
        'mnist': [0, 16, 64, 256],
        'svhn': [0, 16, 64, 256, 1024],
    }

    idlg_types = ['rand']
    metrics = ['rmse', 'psnr', 'ssim', 'lpips']
    metric_display = {
        'rmse': 'RMSE',
        'psnr': 'PSNR (dB)',
        'ssim': 'SSIM',
        'lpips': 'LPIPS'
    }

    datasets = ['bcd', 'mnist', 'svhn']

    all_fusion_values = sorted(set(f for f_list in fusion_options.values() for f in f_list))

    f_minus_one_values = {
        'bcd': 29,
        'mnist': 783,
        'svhn': 3071,
    }

    def format_value(value, metric):
        if value is not None:
            if metric == 'rmse':
                return f"{value:.2f}"
            elif metric == 'psnr':
                if isinstance(value, (int, float)):
                    if value == float('inf') or (not np.isfinite(value) and value > 0):
                        return "$\\infty$"
                    else:
                        return f"{value:.2f}"
                else:
                    return f"{value:.2f}"
            elif metric == 'ssim':
                return f"{value:.2f}"
            elif metric == 'lpips':
                return f"{value:.2f}"
        else:
            if metric == 'psnr':
                return "$\\infty$"
            else:
                return "NA"

    latex_content = []
    latex_content.append("\\resizebox{\\textwidth}{!}{")
    latex_content.append("\\begin{tabular}{p{3.4cm}|c|cccccc}")
    latex_content.append("\\toprule")
    latex_content.append("\\textbf{Dataset} & \\textbf{Metric} & ")
    latex_content.append("\\multicolumn{6}{c}{$|\\mathcal{F}_{HE}|$} \\\\")
    latex_content.append("\\cmidrule(lr){3-8}")

    header_cols = [f"\\textbf{{{f}}}" for f in all_fusion_values] + ["\\textbf{$|F| - 1$}"]
    latex_content.append("& & " + " & ".join(header_cols) + " \\\\")
    latex_content.append("\\midrule")

    for dataset in datasets:
        if dataset not in organized_data:
            continue

        fusion_values = fusion_options[dataset]
        dataset_name = dataset_names.get(dataset, dataset.upper())
        num_rows = len(idlg_types) * len(metrics)
        row_count = 0

        for idlg_type in idlg_types:
            for metric in metrics:
                row_parts = []

                if row_count == 0:
                    row_parts.append(f"\\multirow{{{num_rows}}}{{*}}{{\\parbox{{3.4cm}}{{{dataset_name}}}}}")
                else:
                    row_parts.append("")

                row_parts.append(metric_display[metric])

                for fusion in all_fusion_values:
                    if fusion not in fusion_values:
                        row_parts.append("--")
                    else:
                        if idlg_type in organized_data[dataset] and fusion in organized_data[dataset][idlg_type]:
                            data = organized_data[dataset][idlg_type][fusion]
                            value = data[metric]
                            row_parts.append(format_value(value, metric))
                        else:
                            row_parts.append("NA")

                f_minus_one = f_minus_one_values[dataset]
                if idlg_type in organized_data[dataset] and f_minus_one in organized_data[dataset][idlg_type]:
                    data = organized_data[dataset][idlg_type][f_minus_one]
                    value = data[metric]
                    row_parts.append(format_value(value, metric))
                else:
                    row_parts.append("NA")

                latex_content.append(" & ".join(row_parts) + " \\\\")
                row_count += 1

        if dataset != datasets[-1]:
            latex_content.append("\\midrule")

    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("}")

    caption = "Reconstruction Quality Metrics (RMSE, PSNR, SSIM, LPIPS) for exp111 (Random iDLG) on BCD."
    caption += " $|\\mathcal{F}_{HE}|$ controls the number of features selected via PCA to be encrypted."

    latex_content.append(f"\\caption{{{caption}}}")
    latex_content.append("\\label{tab:recon_quality_exp111}")

    return "\n".join(latex_content)


def _collect_candidates(results_dir, project_root):
    candidates = []
    if os.path.isdir(results_dir):
        candidates.extend(glob.glob(os.path.join(results_dir, "exp111*.json")))

    candidates.extend(glob.glob(os.path.join(project_root, "data", "results", "exp111*.json")))
    return candidates


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))

    results_dir = os.path.join(project_root, "data", "results", "exp111")
    print(f"Looking for files in: {results_dir}")

    exp111_files = _collect_candidates(results_dir, project_root)

    print(f"Found {len(exp111_files)} files")

    if not exp111_files:
        print("No exp111 JSON files found. Checking runs.json for exp111 data...")
        general_runs_file = os.path.join(project_root, "data", "results", "runs.json")
        if os.path.exists(general_runs_file):
            print(f"Found general runs file: {general_runs_file}")
            selected_file = general_runs_file
        else:
            print("No suitable files found for exp111 data.")
            return
    else:
        exp111_files.sort(key=extract_timestamp, reverse=True)

        print("Available exp111 files (newest first):")
        for i, file in enumerate(exp111_files):
            formatted_name = format_filename(file)
            print(f"[{i}] {formatted_name}")

        try:
            selection = int(input(f"\nEnter file number (0-{len(exp111_files) - 1}): "))
            if selection < 0 or selection >= len(exp111_files):
                raise ValueError("Invalid selection")
            selected_file = exp111_files[selection]
        except (ValueError, EOFError):
            print("Invalid selection or no input provided, using the most recent file.")
            selected_file = exp111_files[0]

        print(f"Using file: {format_filename(selected_file)}")

    organized_data = load_and_organize_data(selected_file)

    if not organized_data:
        print("No exp111 data found in the selected file.")
        return

    latex_table = generate_latex_table(organized_data)

    full_table = "\\begin{table}[htbp]\n\\centering\n" + latex_table + "\n\\end{table}"

    output_dir = os.path.join(project_root, "output", "tables")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{os.path.basename(selected_file).replace('.json', '')}_table.tex")
    if selected_file.endswith("runs.json"):
        output_file = os.path.join(output_dir, "exp111_table.tex")

    with open(output_file, 'w') as f:
        f.write(full_table)

    print(f"LaTeX table saved to {output_file}")

    print("\nLaTeX Table:")
    print("===========")
    print(full_table)


if __name__ == "__main__":
    main()
