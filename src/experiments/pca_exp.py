import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.core.datasets import (
    BreastCancerDiagnosticDataset,
    MNISTDataset,
    SVHNDataset,
)


def collect_pca_data(X, max_feature_count=2048, pca_cache_dir=None, pca_cache_name=None, standardize=False):
    pca_loaded = False
    pca = None
    scaler = None
    
    if pca_cache_dir is not None and pca_cache_name is not None:
        os.makedirs(pca_cache_dir, exist_ok=True)
        base_name, ext = os.path.splitext(pca_cache_name)
        if standardize:
            base_name = f"{base_name}_standardized"
        pca_cache_file = os.path.join(pca_cache_dir, f"{base_name}{ext}")
        
        if os.path.exists(pca_cache_file):
            print(f"Loading saved PCA from cache: {pca_cache_file}")
            with open(pca_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            if isinstance(cached_data, dict):
                cached_pca = cached_data.get('pca')
                cached_scaler = cached_data.get('scaler')
            else:
                cached_pca = cached_data
                cached_scaler = None
            
            if cached_pca.n_features_in_ != X.shape[1]:
                print(f"Warning: Cached PCA expects {cached_pca.n_features_in_} features but data has {X.shape[1]}. Recomputing PCA...")
            else:
                pca = cached_pca
                scaler = cached_scaler
                pca_loaded = True
                print(f"Successfully loaded PCA from cache")
    
    if not pca_loaded:
        X_for_pca = X
        if standardize:
            print(f"Standardizing features before PCA...")
            scaler = StandardScaler()
            X_for_pca = scaler.fit_transform(X)
        
        print(f"Computing PCA on {X_for_pca.shape[0]} samples with {X_for_pca.shape[1]} features...")
        pca = PCA()
        pca.fit(X_for_pca)
        print(f"PCA computation complete")
        
        if pca_cache_dir is not None and pca_cache_name is not None:
            base_name, ext = os.path.splitext(pca_cache_name)
            if standardize:
                base_name = f"{base_name}_standardized"
            pca_cache_file = os.path.join(pca_cache_dir, f"{base_name}{ext}")
            print(f"Saving PCA to {pca_cache_file}...")
            cache_data = {'pca': pca, 'scaler': scaler}
            with open(pca_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"PCA saved to cache")
    
    print(f"Computing variance statistics...")
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    feature_counts = np.arange(1, len(cumulative_variance) + 1)

    max_power = int(np.log2(max_feature_count))
    powers_of_two = [2**i for i in range(max_power + 1)]
    variances = []
    for p in tqdm(powers_of_two, desc="Processing feature counts"):
        if p <= len(cumulative_variance):
            variances.append(cumulative_variance[p - 1] * 100)
        else:
            variances.append(None)

    return powers_of_two, variances, feature_counts, cumulative_variance


def generate_combined_summary(dataset_names, datasets_variances):
    feature_counts = [2**i for i in range(0, 12)]

    latex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\caption{Cumulative variance explained (\\%) by PCA across datasets at different feature selection thresholds. The table demonstrates how increasing the number of selected principal components captures progressively more variance in the data. Each dataset has a different dimensionality: BCO (9 total features), BCD (30 total features), MNIST (784 total features), CIFAR-10 (3072 total features), and SVHN (3072 total features). Missing values (-) indicate cases where the feature count exceeds the maximum available features for that dataset.}",
        "\\label{tab:pca_comparison}",
        "\\begin{tabular}{c|" + "c" * len(dataset_names) + "}",
        "\\toprule",
        "Feature Count & " + " & ".join(dataset_names) + " \\\\",
        "\\midrule",
    ]

    for i, fc in enumerate(feature_counts):
        row = [str(fc)]
        for dataset_variances in datasets_variances:
            if i < len(dataset_variances) and dataset_variances[i] is not None:
                row.append(f"{dataset_variances[i]:.2f}")
            else:
                row.append("-")
        latex_lines.append(" & ".join(row) + " \\\\")

    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    os.makedirs("FiguresTables", exist_ok=True)
    with open("FiguresTables/pca_summary.tex", "w") as f:
        f.write("\n".join(latex_lines))


def plot_datasets(plot_data, dataset_names):
    os.makedirs("FiguresTables", exist_ok=True)

    fig, axes = plt.subplots(len(plot_data), 1, figsize=(12, 4 * len(plot_data)))
    if len(plot_data) == 1:
        axes = [axes]

    for ax, (feature_counts, cumulative_variance, powers_of_two, powers_variances), name in zip(
        axes, plot_data, dataset_names
    ):
        ax.plot(feature_counts, cumulative_variance * 100, label="Cumulative Variance")
        ax.scatter(powers_of_two, powers_variances, color="red", label="Powers of Two")

        for x, y in zip(powers_of_two, powers_variances):
            if y is not None:
                ax.plot([x, x], [0, y], color="red", linestyle="--", alpha=0.6)
                ax.plot([0, x], [y, y], color="red", linestyle="--", alpha=0.6)

        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Cumulative % Variance Explained")
        ax.set_title(f"PCA Explained Variance - {name}")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig("FiguresTables/pca_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def process_dataset(dataset_cls, dataset_name=None, pca_cache_dir="pca_cache", use_cache=True, standardize=False):
    print(f"\nProcessing {dataset_name or dataset_cls.__name__}...")
    dataset = dataset_cls()
    X_train, _ = dataset.get_train_data()
    print(f"Dataset loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    pca_cache_name = None
    if use_cache and pca_cache_dir is not None:
        pca_cache_name = f"{dataset_cls.__name__}.pkl"
    
    return collect_pca_data(X_train, pca_cache_dir=pca_cache_dir if use_cache else None, pca_cache_name=pca_cache_name, standardize=standardize)


def main():
    parser = argparse.ArgumentParser(description='PCA Experiment')
    args = parser.parse_args()
    
    dataset_entries = [

        (BreastCancerDiagnosticDataset, "BCD"),
        (MNISTDataset, "MNIST"),
        (SVHNDataset, "SVHN"),
    ]

    datasets_variances = []
    plot_data = []

    print("Starting PCA experiment...")
    for dataset_cls, name in tqdm(dataset_entries, desc="Processing datasets"):
        powers_of_two, variances, feature_counts, cumulative_variance = process_dataset(dataset_cls, name, standardize=False)
        datasets_variances.append(variances)
        plot_data.append((feature_counts, cumulative_variance, powers_of_two, variances))

    print("\nGenerating LaTeX table...")
    dataset_names = [name for _, name in dataset_entries]
    generate_combined_summary(dataset_names, datasets_variances)
    
    print("Generating plots...")
    plot_datasets(plot_data, dataset_names)
    print("Done!")


if __name__ == "__main__":
    main()

