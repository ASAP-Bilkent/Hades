from sklearn.preprocessing import MinMaxScaler
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
from multiprocessing import set_start_method

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (cmd, result.returncode, result.stdout, result.stderr)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Homomorphic Encryption and Machine Learning Models')

    # General params
    parser.add_argument('--net_verbosity', type=int, default=1, help='Verbosity of data and network state')
    parser.add_argument('--no_infer', action='store_true', help='Disable inference')
    parser.add_argument('--no_idlg', action='store_true', help='Disable iDLG')
    parser.add_argument('--idlg_type', type=str, default="cheat", choices=['cheat', 'zero', 'rand'],
                        help='iDLG fusion mode for second network')
    parser.add_argument('--no_test', action='store_true', help='Disable test set eval')
    parser.add_argument('--pca', action='store_true', help='Apply PCA to dataset')
    parser.add_argument('--pca_to_feat', action='store_true', help='Use PCA to select/sort original features instead of transforming to PCA space')
    parser.add_argument('--single_feat_split', type=int, help='Get first n_components')
    parser.add_argument('--exp', type=int, help='Run a predefined experiment')
    parser.add_argument('--dataset', type=str, default="bcd", choices=['bcd', 'bco', 'mnist', 'synth'], 
                        help="Dataset to train and eval: (B)reast(C)ancer(D)iagnostic, (B)reast(C)ancer(O)riginal, MNIST, Synthetic")
    parser.add_argument('--data_amount', type=int, default=1, help='Amount of data to use per client. -1 means all data')
    parser.add_argument('--n_synth_features', type=int, default=32, help='Number of features for synthetic dataset')
    
    # Network params
    parser.add_argument('--epochs', type=float, default=100, help='Number of training epochs (can be fractional, e.g., 1.24)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=4, help='Mini-batch size for training')
    parser.add_argument('--parallel_batch_size', type=int, default=1, help='Number of parallel batches to process')
    parser.add_argument('--dims', type=str, default=None, help='Dimensions of hidden layers (comma separated list of ints)')
    parser.add_argument('--act_func', type=str, choices=['sigmoid', 'approx_sigmoid', 'none', 'relu'], 
                        help="Activation function for the layers")
    parser.add_argument('--approx_params', type=str, default="3-5", 
                        help="Approximation parameters in format 'degree-range' (e.g., '3-5' for degree=3, range=[-5,5])")
    parser.add_argument('--nest', action='store_true', help='Use momentum-based weight updates')
    parser.add_argument('--output_file', type=str, default="runs", help='Base filename for results output JSON')

    # HadesText params
    parser.add_argument('--fake', action='store_true', help='Enable fake bootstrapping mode')
    parser.add_argument('--fake_loss', action='store_true', help='Enable fake loss mode')
    parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level for HadesText')
    parser.add_argument('--mult_types', type=str, default=None, help='Overriding mult_types for HadesText (only PP makes sense?)')

    # Network type params
    parser.add_argument('--test', type=int, help='Run HE tests')
    parser.add_argument('--testing', action='store_true', help='Passed automatically when running HE tests')
    parser.add_argument('--test_all', action='store_true', help='Run select network type tests')
    parser.add_argument('--fusion', type=int, help='Number of elements to use for fusion')
    parser.add_argument('--no_enc', action='store_true', help='Run only no-enc network')
    parser.add_argument('--n_clients', type=int, default=1, help='Number of clients for federated learning')
    parser.add_argument('--fusion_alpha', type=float, default=0.5, help='alpha to control networks\' contribution to loss')

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

def configure_data(dataset, args):
    def split_features(X, y, name):
        utils.vprint(f"Before: X_{name}.shape", X.shape); 
        X_one = X[:, args.fusion:]
        X_two = X[:, :args.fusion]
        utils.vprint(f"After: X_{name}_one.shape, X_{name}_two.shape:", X_one.shape, X_two.shape)
        X = [[X_one, X_two]]
        y = y

        return X, y
    
    def get_piece(piece_size, data):
        return [data[i * piece_size:(i + 1) * piece_size] for i in range(args.n_clients)]

    scaler = MinMaxScaler()
    
    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()

    # Ensure all data is a multiple of args.batch_size
    utils.vprint(f"Before: X_train length: {len(X_train)}, y_train length: {len(y_train)}, X_test length: {len(X_test)}, y_test length: {len(y_test)}")
    X_train = X_train[:len(X_train) - len(X_train) % args.batch_size]
    y_train = y_train[:len(y_train) - len(y_train) % args.batch_size]
    X_test = X_test[:len(X_test) - len(X_test) % args.batch_size]
    y_test = y_test[:len(y_test) - len(y_test) % args.batch_size]
    utils.vprint(f"After: X_train length: {len(X_train)}, y_train length: {len(y_train)}, X_test length: {len(X_test)}, y_test length: {len(y_test)}")

    if args.data_amount == -1:
        X_train = scaler.fit_transform(X_train)
    else:
        scaler.fit(X_train)
        data_per_client = args.data_amount * args.n_clients
        X_train = scaler.transform(X_train[:data_per_client])
        y_train = y_train[:data_per_client]
    X_test = scaler.transform(X_test)

    if args.n_clients > 1:
        if args.fusion is not None:
            X_train, y_train = split_features(X_train, y_train, "train")
            X_test, y_test = split_features(X_test, y_test, "test")
            
            piece_size = len(X_train[0][0]) // args.n_clients

            network_data_one = get_piece(piece_size, X_train[0][0])
            network_data_two = get_piece(piece_size, X_train[0][1])

            X_train = list(zip(network_data_one, network_data_two))
            y_train = get_piece(piece_size, y_train)

            y_test = [y_test]
        else:
            piece_size = len(X_train) // args.n_clients

            X_train = get_piece(piece_size, X_train)
            y_train = get_piece(piece_size, y_train)
            X_test = [X_test]
            y_test = [y_test]
    else:
        if args.fusion is not None:
            X_train, y_train = split_features(X_train, y_train, "train")
            X_test, y_test = split_features(X_test, y_test, "test")
            y_train = [y_train]
            y_test = [y_test]
        else:
            if args.single_feat_split is not None:
                X_train = X_train[:, :args.single_feat_split]
                X_test = X_test[:, :args.single_feat_split]

            X_train = [X_train]
            y_train = [y_train]
            X_test = [X_test]
            y_test = [y_test]

    return X_train, y_train, X_test, y_test

def get_single_network(args):
    if args.no_enc:
        net = Network(
            args=args,
            dims=args.dims
        )
    else:
        net = HENetwork(
            args=args,
            dims=args.dims
        )

    return net

def get_fusion_network(args):
    if args.no_enc:
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
            dataset = BreastCancerOriginalDataset(apply_pca=args.pca, pca_to_feat=args.pca_to_feat)
        elif args.dataset == "mnist":
            dataset = MNISTDataset(apply_pca=args.pca, pca_to_feat=args.pca_to_feat)
        elif args.dataset == "synth":
            dataset = SyntheticDataset(n_features=args.n_synth_features, n_samples=args.n_clients*args.data_amount, apply_pca=args.pca, pca_to_feat=args.pca_to_feat)

        data_manager = DataManager(dataset, args)
        X_train, y_train, X_test, y_test = data_manager.configure_data()

        with timer_context('Precompute'):
            if not args.no_enc:
                ckks = CKKSContext(args.n_clients)
                HadesText.set_ckks_context(ckks)

            if args.fusion is None:
                args.dims = data_manager.setup_dims()
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
                    client_model = HENetwork(base_net.dims, args)
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

    # Print the global timing results
    global_timer.print_timing_table()

    from src.core.utils import print_call_stats
    #print_call_stats()

if __name__ == "__main__":
    args = parse_arguments()
    if args.single_feat_split is not None:
        assert args.pca

    utils.VERBOSITY = args.net_verbosity

    if args.fusion == 0:
        args.fusion = None
    if args.parallel_batch_size > 1:
        assert args.batch_size == 1
        set_start_method("spawn")
    if not args.no_enc:
        if not args.exp and not args.test:
            assert args.no_idlg and args.no_infer
    if args.pca_to_feat:
        assert args.pca, "pca_to_feat requires --pca to be enabled"

    main(args)