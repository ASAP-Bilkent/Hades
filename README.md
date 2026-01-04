# HADES

A federated learning research environment leveraging fully homomorphic encryption (FHE) using OpenFHE and Python. This repository includes the implementation of the paper titled "HADES: Privacy-Preserving Federated Learning via Selective Feature Encryption and Hybrid Model Fusion".

In Hades, we address the challenge of privacy-preserving training in federated learning (FL) by introducing a novel framework that selectively encrypts only the most privacy-sensitive features while leaving the remaining data and the corresponding model portion unencrypted. We propose Hades, a hybrid system that identifies and encrypts the most critical features, ensuring both privacy protection and computational efficiency. Unlike fully encrypted FL training pipelines, which suffer from high computational overhead, Hades integrates an encrypted and non-encrypted training pipeline via a fusion mechanism, enabling seamless interaction between encrypted and plaintext model representations. To achieve this, we use PCA to identify and encrypt the most privacy-sensitive features, which significantly reduces reconstruction attack success in FL. Building on this insight, we design a hybrid FL system that trains an end-to-end encrypted network via multiparty homomorphic encryption (MHE) on the selected features while simultaneously training a plaintext network on the remaining features. 

## 💾 Installation

You can run **Hades** on any operating system with Docker, or build its dependencies manually on Linux or macOS.

After either installation method, you can train or test models with:

```bash
python3 main.py <args>
```
or by invoking the targets in the Makefile.

### Docker (recommended)

```bash
./run_and_exec_docker.sh
```
The script builds the image and launches a container with the Hades workspace.

### Manual build (Linux/macOS)
- Build and install [openfhe-development](https://github.com/openfheorg/openfhe-development)
- Build and install [openfhe-python](https://github.com/openfheorg/openfhe-python)
- Create a virtual environmentand link the openfhe-python `.so` as a library.

## Usage

### 🎛️ Main Script Arguments (`main.py`)

You can customize behavior via the following arguments:

**General Parameters**
- `--net_verbosity`: Verbosity of data and network state (default: 1)
- `--verbosity`: Verbosity level for HadesText (default: 0)
- `--no_infer`: Disable inference
- `--idlg`: Enable iDLG
- `--idlg_type`: Type of iDLG attack (`oracle`, `zero`, `rand`) (default: `oracle`)
- `--no_test`: Disable evaluation on test set
- `--pca`: Apply PCA to dataset
- `--pca_to_feat`: Use PCA to select/sort original features instead of transforming to PCA space
- `--single_feat_split`: Get first n_components
- `--exp`: Run a predefined experiment by number
- `--dataset`: Dataset to use (`bcd`, `mnist`, `synth`, `svhn`) (default: `bcd`)
- `--data_amount`: Number of data points to train on (-1 for all data)
- `--n_synth_features`: Features for synthetic dataset (default: 32)
- `--fusion`: Number of elements for fusion
- `--enc`: Enable encrypted network
- `--hades`: Shortcut to enable HE fusion setup with given fusion size
- `--pca_ablation`: Shortcut for PCA ablation with given fusion size
- `--n_clients`: Number of FL clients (default: 1)
- `--fusion_alpha`: Alpha coefficient for loss fusion (default: 0.5)
- `--loss_enc`: Keeps fused loss encrypted
- `--recon`: Test reconstruction without top fusion PCA features
- `--minmax`: Apply MinMax scaling with range (e.g., "0,1" for range [0,1])
- `--mult_types`: Overriding mult_types for HadesText

**Network Parameters**
- `--epochs`: Number of training epochs (default: 10, can be fractional)
- `--lrate`: Learning rate (default: 0.1)
- `--mom`: Nesterov momentum coefficient (default: 0.0)
- `--batch_size`: Training batch size (default: 4)
- `--dims`: Hidden layer dimensions (comma-separated)
- `--act_func`: Activation function (`sigmoid`, `approx_sigmoid`, `approx_sigmoid_ls`, `approx_relu`, `none`, `relu`)
- `--fus_act_real`: Use non-approx version for network_one in fusion networks
- `--approx_params`: Approximation parameters in format 'degree-range' (default: "3-5")
- `--horner`: Use Horner's method for polynomial evaluation
- `--verbose_grad`: Print aggregated gradient stats every iteration
- `--verbose_act_inp`: Print activation inputs per layer
- `--output_file`: Output JSON filename prefix (default: `runs`)

**Testing Parameters**
- `--test`: Run HE tests
- `--testing`: Passed automatically when running HE tests
- `--test_all`: Run select network type tests

---

### 🛠️ Makefile Commands

To streamline common tasks, use the following commands:

```bash
make train-bcd       # Train on Breast Cancer Diagnostic dataset
make train-mnist     # Train on MNIST dataset
make train-synth     # Train on Synthetic dataset
make train-svhn      # Train on SVHN dataset
make exp<N>          # Run experiment N (e.g. make exp1)
make latex<N>        # Generate LaTeX table for experiment N (e.g. make latex1)
make latex<N>_graph  # Generate LaTeX graph for experiment N (e.g. make latex1_graph)
make latex_pca       # Generate PCA experiment table
make test            # Run all HE tests
make test<N>         # Run specific test N (e.g. make test2)
```
