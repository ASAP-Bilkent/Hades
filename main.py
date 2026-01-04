from src.core.datasets import *
from src.core.trainers import *
from src.core.models import *
import subprocess
import argparse
from src.core import utils
import copy
from src.core.data_manager import DataManager
from src.core.model_factory import ModelFactory
from src.core.experiment_runner import ExperimentRunner

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (cmd, result.returncode, result.stdout, result.stderr)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Homomorphic Encryption and Machine Learning Models')

    # General params
    parser.add_argument('--net_verbosity', type=int, default=1, help='Verbosity of data and network state')
    parser.add_argument('--no_infer', action='store_true', help='Disable inference')
    parser.add_argument('--idlg', action='store_true', help='Enable iDLG')
    parser.add_argument('--idlg_type', type=str, default="oracle", choices=['oracle', 'zero', 'rand'],
                        help='iDLG fusion mode for second network')
    parser.add_argument('--no_test', action='store_true', help='Disable test set eval')
    parser.add_argument('--pca', action='store_true', help='Apply PCA to dataset')
    parser.add_argument('--pca_to_feat', action='store_true', help='Use PCA to select/sort original features instead of transforming to PCA space')
    parser.add_argument('--single_feat_split', type=int, help='Get first n_components')
    parser.add_argument('--exp', type=int, help='Run a predefined experiment')
    parser.add_argument('--dataset', type=str, default="bcd", choices=['bcd', 'mnist', 'synth', 'svhn'], 
                        help="Dataset to train and eval: (B)reast(C)ancer(D)iagnostic, MNIST, Synthetic, SVHN")
    parser.add_argument('--data_amount', type=int, default=-1, help='Amount of data to use per client. -1 means all data')
    parser.add_argument('--n_synth_features', type=int, default=32, help='Number of features for synthetic dataset')
    parser.add_argument('--loss_enc', action='store_true', help='Keeps fused loss encrypted')
    parser.add_argument('--recon', dest='recon', action='store_true',
                        help='Test reconstruction without top fusion PCA features')
    parser.add_argument('--minmax', type=str, default=None, help='Apply MinMax scaling with range (e.g., "0,1" for range [0,1])')
    
    # Network params
    parser.add_argument('--epochs', type=float, default=10, help='Number of training epochs (can be fractional, e.g., 1.24)')
    parser.add_argument('--lrate', type=float, default=0.1, help='Learning rate for training')
    parser.add_argument('--l2_reg', type=float, default=0.0, help='L2 regularization (weight decay) lambda')
    parser.add_argument('--mom', type=float, default=0.0, help='Nesterov momentum coefficient')
    parser.add_argument('--batch_size', type=int, default=4, help='Mini-batch size for training')
    parser.add_argument('--dims', type=str, default=None, help='Dimensions of hidden layers (comma separated list of ints)')
    parser.add_argument('--act_func', type=str, choices=['sigmoid', 'approx_sigmoid', 'approx_sigmoid_ls', 'approx_relu', 'none', 'relu'], 
                        help="Activation function for the layers")
    parser.add_argument('--fus_act_real', action='store_true', 
                        help="When set with an approx_ activation, use its non-approx version for network_one in fusion networks")
    parser.add_argument('--approx_params', type=str, default="3-5", 
                        help="Approximation parameters in format 'degree-range' (e.g., '3-5' for degree=3, range=[-5,5])")
    parser.add_argument('--horner', action='store_true', help='Use Horner\'s method for polynomial evaluation')
    parser.add_argument('--verbose_grad', action='store_true', help='Print aggregated gradient stats every iteration')
    parser.add_argument('--verbose_act_inp', action='store_true', help='Print activation inputs per layer')
    parser.add_argument('--output_file', type=str, default="runs", help='Base filename for results output JSON')
    parser.add_argument('--weight_init', type=str, default='default', choices=['default', 'xav_uni', 'xav_norm', 'he_uni', 'he_norm'],
                        help='Weight initialization method: default, xavier uniform/normal, he uniform/normal')
    # HadesText params
    parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level for HadesText')
    parser.add_argument('--mult_types', type=str, default=None, help='Overriding mult_types for HadesText (only PP makes sense?)')

    # Network type params
    parser.add_argument('--test', type=int, help='Run HE tests')
    parser.add_argument('--testing', action='store_true', help='Passed automatically when running HE tests')
    parser.add_argument('--test_all', action='store_true', help='Run select network type tests')
    parser.add_argument('--fusion', type=int, help='Number of elements to use for fusion')
    parser.add_argument('--enc', action='store_true', help='Enable encrypted network')
    parser.add_argument('--hades', type=int, help='Shortcut to enable HE fusion setup with given fusion size')
    parser.add_argument('--pca_ablation', type=int, help='Shortcut for PCA ablation with given fusion size')
    parser.add_argument('--n_clients', type=int, default=1, help='Number of clients for federated learning')
    parser.add_argument('--fusion_alpha', type=float, default=0.5, help='alpha to control networks\' contribution to loss')
    parser.add_argument('--noflpt', action='store_true', help='Do not average gradients of network_one (network_two is averaged as usual)')

    return parser.parse_args()

def setup_dims(args, dataset, X_train=None):
    if args.dims is not None:
        dims = [dataset.feature_count] + [int(x) for x in args.dims.split(",")] + [dataset.get_num_classes()]
    else:
        dims = [dataset.feature_count, dataset.get_num_classes()]

    if args.single_feat_split is not None:
        dims[0] = args.single_feat_split
    utils.vprint(f"Training started with bs {args.batch_size} and dims {dims}")

    if X_train is not None:
        dims_one = copy.deepcopy(dims)
        if args.n_clients > 1 or args.fusion is not None:
            dims_one[0] = X_train[0][0].shape[-1]
        else:
            dims_one[0] = X_train[0].shape[-1]

        dims_two = copy.deepcopy(dims)
        if args.n_clients > 1 or args.fusion is not None:
            dims_two[0] = X_train[0][1].shape[-1]
        else:
            dims_two[0] = X_train[1].shape[-1]

        dims = [dims_one, dims_two]

    return dims

def get_single_network(args):
    if not args.enc:
        net = Network(
            args=args,
            dims=args.dims,
            network_label="PT"
        )
    else:
        net = HENetwork(
            args=args,
            dims=args.dims,
            network_label="CT"
        )

    return net

def get_fusion_network(args):
    if not args.enc:
        net = NoEncFusionNetwork(
            args=args,
            dims_one=args.dims[0],
            dims_two=args.dims[1]
        )
    else:
        net = FusionNetwork(
            args=args,
            dims_one=args.dims[0],
            dims_two=args.dims[1]
        )

    return net

def run_recon(args, dataset):
    fusion_size = args.recon_fusion if hasattr(args, "recon_fusion") else args.fusion
    if fusion_size == 0:
        print(f"\n=== Reconstruction: All features available (fusion=0) ===")
    else:
        print(f"\n=== Reconstruction: Plotting original vs X_one (leftover features after removing top {fusion_size}) ===")

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    X_train_orig = dataset.X_train
    X_train_one = X_train_orig[:, fusion_size:]

    print(f"Original feature count: {X_train_orig.shape[1]}")
    if fusion_size == 0:
        print(f"X_one feature count: {X_train_one.shape[1]} (all features available)")
    else:
        print(f"X_one feature count (leftover after removing top {fusion_size}): {X_train_one.shape[1]}")

    num_samples_to_plot = min(5, len(X_train_orig))
    num_samples_for_fidelity = min(100, len(X_train_orig))
    indices_plot = np.random.choice(len(X_train_orig), size=num_samples_to_plot, replace=False)
    indices_fidelity = np.random.choice(len(X_train_orig), size=num_samples_for_fidelity, replace=False)

    from skimage.metrics import structural_similarity as ssim

    try:
        import lpips
        import torch
        lpips_available = True
        lpips_model = lpips.LPIPS(net='alex')
    except ImportError:
        lpips_available = False
        lpips_model = None
        print("Warning: lpips not available. Install with: pip install lpips")

    mse_all = []
    mse_image_all = []
    ssim_all = []
    psnr_all = []
    mse_normalized_all = []
    lpips_all = []

    def get_image_shape(dataset_name, feature_count):
        if dataset_name == "mnist":
            return (28, 28)
        elif dataset_name == "svhn":
            return (32, 32, 3)
        elif dataset_name == "svhn_gray":
            return (32, 32)
        elif dataset_name == "cifar10":
            return (32, 32, 3)
        else:
            h = int(np.sqrt(feature_count))
            w = feature_count // h
            return (h, w)

    def reconstruct_image_from_pca(pca_features, pca_obj, original_shape):
        if pca_obj is None:
            return None
        try:
            pca_features_full = np.zeros((1, pca_obj.n_components_))
            pca_features_full[0, :len(pca_features)] = pca_features
            reconstructed = pca_obj.inverse_transform(pca_features_full)
            return reconstructed.reshape(original_shape)
        except:
            return None

    def psnr(img1, img2, max_val=1.0):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val / np.sqrt(mse))

    if dataset.pca is not None:
        original_feature_count = dataset.pca.n_features_in_
    else:
        original_feature_count = X_train_orig.shape[1]

    img_shape = get_image_shape(args.dataset, original_feature_count)
    is_color = len(img_shape) == 3

    def fill_missing_features(orig_features, x_one_features, fusion_size, idlg_type):
        x_one_full = np.zeros_like(orig_features)
        x_one_full[fusion_size:] = x_one_features
        if idlg_type == "oracle":
            x_one_full[:fusion_size] = orig_features[:fusion_size]
        elif idlg_type == "rand":
            x_one_full[:fusion_size] = np.random.uniform(low=0, high=1, size=fusion_size)
        elif idlg_type == "zero":
            x_one_full[:fusion_size] = 0.0
        return x_one_full

    for i in indices_fidelity:
        orig_features = X_train_orig[i]
        x_one_features = X_train_one[i]
        x_one_full = fill_missing_features(orig_features, x_one_features, fusion_size, args.idlg_type)

        mse = np.mean((orig_features - x_one_full) ** 2)
        mse_all.append(mse)

        mse_normalized = mse / (np.var(orig_features) + 1e-10)
        mse_normalized_all.append(mse_normalized)

        orig_img = None
        recon_img = None

        if dataset.pca is not None:
            orig_img = reconstruct_image_from_pca(orig_features, dataset.pca, img_shape)
            recon_img = reconstruct_image_from_pca(x_one_full, dataset.pca, img_shape)
        else:
            if len(img_shape) == 2:
                orig_img = orig_features.reshape(img_shape)
                recon_img = x_one_full.reshape(img_shape)
            elif len(img_shape) == 3:
                orig_img = orig_features.reshape(img_shape)
                recon_img = x_one_full.reshape(img_shape)

        if orig_img is not None and recon_img is not None:
            orig_img_clipped = np.clip(orig_img, 0, 1)
            recon_img_clipped = np.clip(recon_img, 0, 1)

            mse_image = np.mean((orig_img_clipped - recon_img_clipped) ** 2)
            mse_image_all.append(mse_image)

            if is_color:
                ssim_score = ssim(orig_img_clipped, recon_img_clipped, data_range=1.0, channel_axis=2, win_size=min(7, min(img_shape[:2])))
            else:
                ssim_score = ssim(orig_img_clipped, recon_img_clipped, data_range=1.0, win_size=min(7, min(img_shape)))
            ssim_all.append(ssim_score)

            psnr_score = psnr(orig_img_clipped, recon_img_clipped)
            if np.isfinite(psnr_score):
                psnr_all.append(psnr_score)

            if lpips_available and lpips_model is not None:
                try:
                    actual_shape = orig_img_clipped.shape

                    if len(actual_shape) == 2:
                        h, w = actual_shape
                        if h < 16 or w < 16:
                            continue
                        orig_tensor = torch.from_numpy(orig_img_clipped).float()
                        recon_tensor = torch.from_numpy(recon_img_clipped).float()
                        orig_tensor = orig_tensor.unsqueeze(0).unsqueeze(0)
                        recon_tensor = recon_tensor.unsqueeze(0).unsqueeze(0)
                        min_size = 64
                        if orig_tensor.shape[2] < min_size or orig_tensor.shape[3] < min_size:
                            orig_tensor = torch.nn.functional.interpolate(orig_tensor, size=(min_size, min_size), mode='bilinear', align_corners=False)
                            recon_tensor = torch.nn.functional.interpolate(recon_tensor, size=(min_size, min_size), mode='bilinear', align_corners=False)
                        orig_tensor = orig_tensor.repeat(1, 3, 1, 1)
                        recon_tensor = recon_tensor.repeat(1, 3, 1, 1)
                    elif len(actual_shape) == 3:
                        h, w, c = actual_shape
                        if h < 16 or w < 16:
                            continue
                        orig_tensor = torch.from_numpy(orig_img_clipped).float()
                        recon_tensor = torch.from_numpy(recon_img_clipped).float()
                        orig_tensor = orig_tensor.transpose(2, 0).transpose(1, 2).unsqueeze(0)
                        recon_tensor = recon_tensor.transpose(2, 0).transpose(1, 2).unsqueeze(0)
                        min_size = 64
                        if orig_tensor.shape[2] < min_size or orig_tensor.shape[3] < min_size:
                            orig_tensor = torch.nn.functional.interpolate(orig_tensor, size=(min_size, min_size), mode='bilinear', align_corners=False)
                            recon_tensor = torch.nn.functional.interpolate(recon_tensor, size=(min_size, min_size), mode='bilinear', align_corners=False)
                    else:
                        continue

                    orig_tensor = orig_tensor * 2.0 - 1.0
                    recon_tensor = recon_tensor * 2.0 - 1.0

                    lpips_score = lpips_model.forward(orig_tensor, recon_tensor)
                    lpips_all.append(lpips_score.item())
                except Exception:
                    pass

    def get_fidelity(threshold, mse_list):
        return float(100 * sum((np.array(mse_list) < threshold).astype(int)) / len(mse_list))

    avg_mse = np.mean(mse_all)
    avg_mse_image = np.mean(mse_image_all) if mse_image_all else None
    avg_ssim = np.mean(ssim_all) if ssim_all else None
    avg_psnr = np.mean(psnr_all) if psnr_all else None
    avg_mse_normalized = np.mean(mse_normalized_all)
    avg_lpips = np.mean(lpips_all) if lpips_all else None

    fidelity_easy = get_fidelity(0.01, mse_all)
    fidelity_hard = get_fidelity(0.0001, mse_all)

    fidelity_easy_image = get_fidelity(0.01, mse_image_all) if mse_image_all else None
    fidelity_hard_image = get_fidelity(0.0001, mse_image_all) if mse_image_all else None

    print(f"\nFidelity Metrics (based on {len(mse_all)} samples):")
    print(f"\nPCA Feature Space Metrics:")
    print(f"  Average MSE: {avg_mse:.6f}")
    print(f"  Average Normalized MSE: {avg_mse_normalized:.6f}")
    print(f"  % Good Fidelity (MSE < 0.01): {fidelity_easy:.2f}%")
    print(f"  % Good Fidelity (MSE < 0.0001): {fidelity_hard:.2f}%")

    if avg_mse_image is not None:
        print(f"\nImage Space Metrics:")
        print(f"  Average Image MSE: {avg_mse_image:.6f}")
        print(f"  % Good Fidelity Image (MSE < 0.01): {fidelity_easy_image:.2f}%")
        print(f"  % Good Fidelity Image (MSE < 0.0001): {fidelity_hard_image:.2f}%")

    if avg_ssim is not None or avg_psnr is not None or (avg_lpips is not None and len(lpips_all) > 0):
        print(f"\nPerceptual Quality Metrics:")

        if avg_ssim is not None:
            print(f"  Average SSIM: {avg_ssim:.4f} (range: 0-1, higher is better)")
            print(f"  % Good SSIM (SSIM > 0.7): {100 * sum((np.array(ssim_all) > 0.7).astype(int)) / len(ssim_all):.2f}%")
            print(f"  % Good SSIM (SSIM > 0.8): {100 * sum((np.array(ssim_all) > 0.8).astype(int)) / len(ssim_all):.2f}%")

        if avg_psnr is not None:
            print(f"  Average PSNR: {avg_psnr:.2f} dB (higher is better)")
            print(f"  % Good PSNR (PSNR > 10 dB): {100 * sum((np.array(psnr_all) > 10).astype(int)) / len(psnr_all):.2f}%")
            print(f"  % Good PSNR (PSNR > 20 dB): {100 * sum((np.array(psnr_all) > 20).astype(int)) / len(psnr_all):.2f}%")

        if avg_lpips is not None and len(lpips_all) > 0:
            print(f"  Average LPIPS: {avg_lpips:.4f} (lower is better, 0 = identical)")
            print(f"  % Good LPIPS (LPIPS < 0.1): {100 * sum((np.array(lpips_all) < 0.1).astype(int)) / len(lpips_all):.2f}%")
            print(f"  % Good LPIPS (LPIPS < 0.2): {100 * sum((np.array(lpips_all) < 0.2).astype(int)) / len(lpips_all):.2f}%")

    if lpips_available and (avg_lpips is None or len(lpips_all) == 0):
        print(f"\nLPIPS Metrics:")
        print(f"  LPIPS computation attempted but no valid scores computed (check errors above)")

    fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(10, 5 * num_samples_to_plot))
    if num_samples_to_plot == 1:
        axes = axes.reshape(1, -1)

    for idx, i in enumerate(indices_plot):
        orig_features = X_train_orig[i]
        x_one_features = X_train_one[i]

        orig_img = None
        x_one_img = None

        if dataset.pca is not None:
            orig_img = reconstruct_image_from_pca(orig_features, dataset.pca, img_shape)
            x_one_full = fill_missing_features(orig_features, x_one_features, fusion_size, args.idlg_type)
            x_one_img = reconstruct_image_from_pca(x_one_full, dataset.pca, img_shape)
        else:
            if len(img_shape) == 2:
                orig_img = orig_features.reshape(img_shape)
                x_one_padded = fill_missing_features(orig_features, x_one_features, fusion_size, args.idlg_type)
                x_one_img = x_one_padded.reshape(img_shape)
            elif len(img_shape) == 3:
                orig_img = orig_features.reshape(img_shape)
                x_one_padded = fill_missing_features(orig_features, x_one_features, fusion_size, args.idlg_type)
                x_one_img = x_one_padded.reshape(img_shape)

        if orig_img is not None:
            if is_color:
                axes[idx, 0].imshow(np.clip(orig_img, 0, 1))
            else:
                axes[idx, 0].imshow(np.clip(orig_img, 0, 1), cmap='gray')
            axes[idx, 0].axis('off')
        else:
            axes[idx, 0].text(0.5, 0.5, 'Cannot reconstruct\nfrom PCA features',
                             ha='center', va='center', transform=axes[idx, 0].transAxes)
            axes[idx, 0].axis('off')

        if x_one_img is not None:
            if is_color:
                axes[idx, 1].imshow(np.clip(x_one_img, 0, 1))
            else:
                axes[idx, 1].imshow(np.clip(x_one_img, 0, 1), cmap='gray')
            axes[idx, 1].axis('off')
        else:
            axes[idx, 1].text(0.5, 0.5, 'Cannot reconstruct\nfrom X_one features',
                             ha='center', va='center', transform=axes[idx, 1].transAxes)
            axes[idx, 1].axis('off')

    plt.tight_layout()

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f'recon_{args.dataset}_fusion{fusion_size}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

    if args.net_verbosity > 0 and args.output_file:
        import json
        import datetime
        import re
        import sys

        exp_match = re.match(r'(exp\d+)-', args.output_file)
        if exp_match:
            exp_name = exp_match.group(1)
            results_dir = os.path.join('data', 'results', exp_name)
            os.makedirs(results_dir, exist_ok=True)
            results_path = os.path.join(results_dir, f"{args.output_file}.json")
        else:
            results_dir = os.path.join('data', 'results')
            os.makedirs(results_dir, exist_ok=True)
            results_path = os.path.join(results_dir, f"{args.output_file}.json")

        logged_argv = []
        i = 0
        while i < len(sys.argv):
            if sys.argv[i] == '--output_file' or sys.argv[i].startswith('--output_file='):
                if sys.argv[i] == '--output_file' and i + 1 < len(sys.argv):
                    i += 2
                    continue
                else:
                    i += 1
                    continue
            logged_argv.append(sys.argv[i])
            i += 1

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cli_command": " ".join(logged_argv),
            "avg_rmse": float(np.sqrt(avg_mse)) if avg_mse is not None else None,
            "avg_psnr": float(avg_psnr) if avg_psnr is not None else None,
            "avg_ssim": float(avg_ssim) if avg_ssim is not None else None,
            "avg_lpips": float(avg_lpips) if avg_lpips is not None else None,
        }

        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                existing = json.load(f)
        else:
            existing = []

        existing.append(entry)
        with open(results_path, "w") as f:
            json.dump(existing, f, indent=4)

    print("\n=== Reconstruction Complete ===\n")


def main(args):
    from src.core.utils import timer_context, global_timer
    global_timer.set_n_clients(args.n_clients)
    
    # Parse approx_params format "degree-range"
    if hasattr(args, 'approx_params') and args.approx_params:
        try:
            degree_str, range_str = args.approx_params.split('-')
            args.approx_degree = int(degree_str)
            range_val = float(range_str)
            args.approx_a = -range_val
            args.approx_b = range_val
        except ValueError:
            raise ValueError(f"Invalid approx_params format '{args.approx_params}'. Expected format: 'degree-range' (e.g., '3-5')")
    else:
        args.approx_degree = 3
        args.approx_a = -5.0
        args.approx_b = 5.0

    def _parse_range_arg(value, flag_name):
        if value is None:
            return None
        try:
            min_val, max_val = map(float, value.split(','))
        except ValueError:
            raise ValueError(f"Invalid format for {flag_name}: '{value}'. Expected 'min,max'")
        if min_val >= max_val:
            raise ValueError(f"{flag_name} requires min < max. Got {min_val} >= {max_val}")
        return (min_val, max_val)

    args.minmax_range = _parse_range_arg(args.minmax, '--minmax')
    
    with timer_context('Total'):
        if args.test is not None:
            experiment_runner = ExperimentRunner(args)
            experiment_runner.run_tests()
            return
        elif args.exp is not None:
            experiment_runner = ExperimentRunner(args)
            experiment_runner.run_experiment()
            return

        if args.dataset == "bcd":
            dataset = BreastCancerDiagnosticDataset(apply_pca=args.pca, pca_to_feat=args.pca_to_feat)
        elif args.dataset == "bco":
            print('Error: BreastCancerOriginalDataset was removed during cleanup')
            return
        elif args.dataset == "mnist":
            dataset = MNISTDataset(apply_pca=args.pca, pca_to_feat=args.pca_to_feat)
        elif args.dataset == "cifar10":
            print('Error: CIFAR10Dataset was removed during cleanup')
            return
        elif args.dataset == "synth":
            dataset = SyntheticDataset(n_features=args.n_synth_features, n_samples=args.n_clients*args.data_amount, apply_pca=args.pca, pca_to_feat=args.pca_to_feat)
        elif args.dataset == "svhn":
            dataset = SVHNDataset(apply_pca=args.pca, pca_to_feat=args.pca_to_feat, data_amount=args.data_amount)
        elif args.dataset == "svhn_gray":
            print('Error: SVHNGrayDataset was removed during cleanup')
            return

        data_manager = DataManager(dataset, args)
        X_train, y_train, X_test, y_test = data_manager.configure_data()

        if args.recon and not args.idlg:
            run_recon(args, dataset)
            return

        with timer_context('Precompute'):
            if args.enc:
                ckks = CKKSContext(args.n_clients)
                HadesText.set_ckks_context(ckks)

            if args.fusion is None:
                args.dims = data_manager.setup_dims(X_train)
                model_factory = ModelFactory(args)
                base_net = model_factory.get_single_network()
            else:
                if args.fusion == 0 or args.fusion == 30:
                    raise ValueError("args.fusion should be in range [1, 30].")
                args.dims = data_manager.setup_dims(X_train)
                model_factory = ModelFactory(args)
                base_net = model_factory.get_fusion_network()

            client_models = [base_net]
            for _ in range(args.n_clients - 1):
                if isinstance(base_net, HENetwork):
                    client_model = HENetwork(base_net.dims, args, network_label="CT")
                elif hasattr(base_net, 'network_two') and isinstance(base_net.network_two, HENetwork):
                    client_model = model_factory.get_fusion_network()
                else:
                    client_model = copy.deepcopy(base_net)
                client_models.append(client_model)

        if args.fusion is not None:
            num_samples = len(X_train[0][0])
        else:
            num_samples = len(X_train[0])
        global_timer.set_batch_info(num_samples, args.batch_size)

        trainer = BaseTrainer(args)
        trainer.run(client_models, X_train, y_train, X_test, y_test)

        if args.recon and args.idlg:
            run_recon(args, dataset)

    # Print the global timing results
    global_timer.print_timing_table()

    from src.core.utils import print_call_stats
    #print_call_stats()

if __name__ == "__main__":
    args = parse_arguments()
    if args.single_feat_split is not None:
        assert args.pca
    if args.hades is not None:
        args.n_clients = 10
        args.pca = True
        args.loss_enc = True
        args.fus_act_real = True
        args.fusion = args.hades
    if args.pca_ablation is not None:
        args.n_clients = 10
        args.pca = True
        args.single_feat_split = args.pca_ablation
    utils.VERBOSITY = args.net_verbosity

    if args.recon:
        args.recon_fusion = args.fusion

    if args.fusion == 0 and (not args.recon or args.idlg):
        args.fusion = None
        
    if args.enc:
        if not args.exp and not args.test:
            assert not args.idlg and args.no_infer
    if args.enc and args.horner:
        assert args.act_func in ('approx_sigmoid_ls', 'approx_relu'), (
            "When using --enc and --horner, --act_func must be approx_sigmoid_ls or approx_relu"
        )
    if args.pca_to_feat:
        assert args.pca, "pca_to_feat requires --pca to be enabled"
    if args.loss_enc:
        assert args.fusion != 0
    if args.recon:
        assert args.pca, "recon requires --pca to be enabled"
        assert args.recon_fusion is not None and args.recon_fusion >= 0, "recon requires --fusion to be set (can be 0 for all features)"

    main(args)
