# HADES

A federated learning research environment leveraging fully homomorphic encryption (FHE) using OpenFHE and Python. 

In Hades, we address the challenge of privacy-preserving training in federated learning (FL) by introducing a novel framework that selectively encrypts only the most privacy-sensitive features while leaving the remaining data and the corresponding model portion unencrypted. We propose Hades, a hybrid system that identifies and encrypts the most critical features, ensuring both privacy protection and computational efficiency. Unlike fully encrypted FL training pipelines, which suffer from high computational overhead, Hades integrates an encrypted and non-encrypted training pipeline via a fusion mechanism, enabling seamless interaction between encrypted and plaintext model representations. To achieve this, we use PCA to identify and encrypt the most privacy-sensitive features, which significantly reduces reconstruction attack success in FL. Building on this insight, we design a hybrid FL system that trains an end-to-end encrypted network via multiparty homomorphic encryption (MHE) on the selected features while simultaneously training a plaintext network on the remaining features. These two networks are then integrated using a fusion mechanism, ensuring both privacy preservation and computational efficiency. We introduce a general packing scheme that eliminates redundant rotations by considering the entire neural network architecture. Finally, we demonstrate that Hades matches the accuracy of vanilla FL while preserving privacy and achieving optimized runtime through selective encryption.

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
- `--verbosity`: Verbosity level (default: 0)
- `--no_infer`: Disable inference
- `--no_idlg`: Disable iDLG
- `--idlg_type`: Type of iDLG attack (`cheat`, `zero`, `rand`) (default: `cheat`)
- `--no_test`: Disable evaluation on test set
- `--pca`: Apply PCA to dataset
- `--exp`: Run a predefined experiment by number
- `--dataset`: Dataset to use (`bcd`, `bco`, `mnist`, `synth`) (default: `bcd`)
- `--data_amount`: Number of data points to train on (-1 for all data)
- `--n_synth_features`: Features for synthetic dataset
- `--fusion`: Number of elements for fusion
- `--no_enc`: Run only plaintext model
- `--n_clients`: Number of FL clients (default: 1)
- `--fusion_alpha`: Alpha coefficient for loss fusion (default: 0.5)

**Network Parameters**
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 0.1)
- `--batch_size`: Training batch size (default: 4)
- `--dims`: Hidden layer dimensions (comma-separated)
- `--act_func`: Activation function (`relu`, `sigmoid`, `approx_relu`, `approx_sigmoid`)
- `--output_file`: Output JSON filename prefix (default: `runs`)

---

### 🛠️ Makefile Commands

To streamline common tasks, use the following commands:

```bash
make train-bco       # Train on Breast Cancer Original dataset
make train-bcd       # Train on Breast Cancer Diagnostic dataset
make train-mnist     # Train on MNIST dataset
make exp<N>          # Run experiment N (e.g. make exp1)
make latex<N>        # Generate LaTeX table for experiment N (e.g. make latex1)
make latex<N>_graph  # Generate LaTeX graph for experiment N (e.g. make latex1_graph)
make latex_pca       # Generate PCA experiment table
make test            # Run all HE tests
make test<N>         # Run specific test N (e.g. make test2)
```
