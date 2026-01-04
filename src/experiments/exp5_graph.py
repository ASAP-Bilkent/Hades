#!/usr/bin/env python3
import json
import os
import glob
import re
from datetime import datetime
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


def extract_timestamp(filename):
    match = re.search(r'exp\d+-(\d{10,12})\.json$', os.path.basename(filename))
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, "%d%m%Y%H%M")
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


def parse_cli_args(cmd):
    params = {}
    dataset_match = re.search(r'--dataset=([a-zA-Z0-9]+)', cmd)
    params['dataset'] = dataset_match.group(1).lower() if dataset_match else None

    pca_match = re.search(r'--pca_ablation=([0-9]+)', cmd)
    params['pca_ablation'] = int(pca_match.group(1)) if pca_match else None

    hades_match = re.search(r'--hades=([0-9]+)', cmd)
    params['hades'] = int(hades_match.group(1)) if hades_match else None

    return params


def organize_data(results):
    data = defaultdict(lambda: {"pca": {}, "hades": {}})
    for result in results:
        params = parse_cli_args(result.get('cli_command', ''))
        dataset = params.get('dataset')
        if not dataset:
            continue
        acc = result.get('acc_test')
        if acc is None:
            continue
        if params.get('pca_ablation') is not None:
            data[dataset]["pca"][params['pca_ablation']] = acc
        if params.get('hades') is not None:
            data[dataset]["hades"][params['hades']] = acc
    return data


def create_exp5_plot(data_by_dataset, output_dir):
    if not data_by_dataset:
        print("No data found")
        return None

    datasets = sorted(data_by_dataset.keys())
    n = len(datasets)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    title_size = 16
    label_size = 14
    tick_size = 12
    legend_size = 11

    for idx, dataset in enumerate(datasets):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        pca_points = data_by_dataset[dataset]["pca"]
        hades_points = data_by_dataset[dataset]["hades"]
        expected_x = sorted(set(list(pca_points.keys()) + list(hades_points.keys())))

        if expected_x:
            pca_y = [
                pca_points[x] * 100 if x in pca_points else float("nan")
                for x in expected_x
            ]
            hades_y = [
                hades_points[x] * 100 if x in hades_points else float("nan")
                for x in expected_x
            ]
            if any(not math.isnan(v) for v in pca_y):
                ax.plot(expected_x, pca_y, marker="o", linewidth=2.2, label="Baseline")
            if any(not math.isnan(v) for v in hades_y):
                ax.plot(expected_x, hades_y, marker="s", linewidth=2.2, label="Hades")

        if expected_x:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(FixedLocator(expected_x))
            ax.set_xticklabels([str(x) for x in expected_x])
        ax.set_title(dataset.upper(), fontsize=title_size)
        ax.set_xlabel("Number of Encrypted Features ($F_{HE}$)", fontsize=label_size)
        ax.set_ylabel("Test Accuracy (%)", fontsize=label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=legend_size)

    total_axes = nrows * ncols
    for idx in range(n, total_axes):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    fig.tight_layout()
    output_file = os.path.join(output_dir, "exp5_pca_hades_accuracy.pdf")
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    return output_file


def generate_latex_figure(plot_path, output_dir):
    latex_file = os.path.join(output_dir, "exp5_figures.tex")
    latex_content = []
    latex_content.append("\\begin{figure}[t]")
    latex_content.append("\\centering")
    if plot_path:
        latex_content.append(f"\\includegraphics[width=0.95\\linewidth]{{{plot_path}}}")
    caption = (
        "Test accuracy for PCA-only and HADES across encrypted feature counts. "
        "Each subplot corresponds to a dataset."
    )
    latex_content.append(f"\\caption{{{caption}}}")
    latex_content.append("\\label{fig:exp5_pca_hades}")
    latex_content.append("\\end{figure}")

    with open(latex_file, "w") as f:
        f.write("\n".join(latex_content))
    print(f"LaTeX figure code written to {latex_file}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    results_dir = os.path.join(project_root, "data", "results", "exp5")
    selected_exp5 = select_results_file(results_dir, "exp5")
    if not selected_exp5:
        return

    with open(selected_exp5, "r") as f:
        results = json.load(f)

    data_by_dataset = organize_data(results)
    if not data_by_dataset:
        print("No valid data found in the selected file.")
        return

    output_dir = os.path.join(project_root, "output", "figures")
    os.makedirs(output_dir, exist_ok=True)

    plot_path = create_exp5_plot(data_by_dataset, output_dir)
    if plot_path:
        print(f"Exp5 PCA vs Hades plot saved to: {plot_path}")

    generate_latex_figure(plot_path, output_dir)


if __name__ == "__main__":
    main()
