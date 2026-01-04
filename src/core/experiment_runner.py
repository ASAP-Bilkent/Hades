import datetime
import subprocess
import sys
import os
import traceback
from multiprocessing import Process, Pool
import numpy as np
from src.core.data_manager import DataManager
from src.core.model_factory import ModelFactory
from src.core.trainers import BaseTrainer
from src.core import utils
from src.core.models import HENetwork
from src.core.hadestext import HadesText
from src.core.datasets import BreastCancerDiagnosticDataset, MNISTDataset, SyntheticDataset, SVHNDataset

class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.dataset = None
        
    def _create_dataset(self):
        if self.args.dataset == "bcd":
            self.dataset = BreastCancerDiagnosticDataset(apply_pca=self.args.pca)
        elif self.args.dataset == "mnist":
            self.dataset = MNISTDataset(apply_pca=self.args.pca)
        elif self.args.dataset == "synth":
            self.dataset = SyntheticDataset(n_features=self.args.n_synth_features, 
                                          n_samples=self.args.batch_size*self.args.n_clients*self.args.data_amount, 
                                          apply_pca=self.args.pca)
        elif self.args.dataset == "svhn":
            self.dataset = SVHNDataset(apply_pca=self.args.pca, data_amount=self.args.data_amount)
        
    def run_experiment(self):
        now = datetime.datetime.now()
        date_str = now.strftime("%d%m%Y%H%M")
        output_file = f"exp{self.args.exp}-{date_str}"
        
        experiment_method = getattr(self, f"_run_experiment_{self.args.exp}", None)
        if experiment_method:
            experiment_method(output_file)
        else:
            print(f"Experiment {self.args.exp} not defined")
    
    def _run_command(self, cmd):
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return (cmd, result.returncode, result.stdout, result.stderr)
        
    def _run_experiment_1(self, output_file):
        """Q1, reconstruction experiment"""
        additional = f" --epochs=10 --batch_size=1 --data_amount=-1 --act_fun=sigmoid --lrate=0.1 --mom=0.9 --minmax=0,1 --output_file={output_file}"
        datasets = ["bcd", "mnist", "svhn"]
        idlg_types = ["rand"]
        commands = []

        for ds in datasets:
            base = f"python3 main.py --idlg --dataset={ds}" + additional
            if ds == "bcd":
                #options = [0, 8, 16]
                options = [0, 4, 8, 16]
                dim = "16"
            elif ds == "mnist":
                #options = [0, 8, 16, 64, 128, 256]
                options = [0, 16, 64, 256]
                dim = "64"
            elif ds == "svhn":
                #options = [0, 8, 16, 64, 128, 256]
                options = [0, 64, 256, 1024]
                dim = "64,64"

            for fusion in options:
                for idlg in idlg_types:
                    commands.append(base + f" --dims={dim}  --pca --fusion={fusion} --idlg_type={idlg}")

        for cmd in commands:
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)
            
    def _run_experiment_2(self, output_file):
        """Q2, utility experiment"""
        dims = {
            "bcd": "16",
            "mnist": "64",
            "svhn": "64,64",
        }
        fusion_options = {
            "bcd": "16",
            "mnist": "256",
            "svhn": "256",
        }
        learning_rates = {
            "bcd": 0.1,
            "mnist": 0.1,
            "svhn": 0.1,
        }
        #datasets = ["bcd", "mnist", "svhn"]
        datasets = ["mnist"]
        commands = []
        
        base_cmd = f"python3 main.py --data_amount=-1 --epoch=10 --output_file={output_file}"
        
        for ds in datasets:
            lr = str(learning_rates[ds])
            common_args = f" --dims={dims[ds]} --dataset={ds} --mom=0.9 "
            if ds == "bcd":
                common_args += " --minmax=-1,1"
            
            # Centrilized sigmoid no-fusion
            cmd = base_cmd + common_args + f" --batch_size=10 --act_fun=sigmoid --fusion=0 --lrate={lr}"
            commands.append(cmd)
            
            # Centrilized sigmoid fusion
            cmd = base_cmd + common_args + f" --batch_size=10 --act_fun=sigmoid --fusion={fusion_options[ds]} --lrate={lr} "
            commands.append(cmd)
            
            # Centrilized approx-sigmoid no-fusion
            #cmd = base_cmd + common_args + " --act_fun=approx_sigmoid --fusion=0"
            #commands.append(cmd)

            # FL sigmoid fusion 10 party
            cmd = base_cmd + common_args + f" --batch_size=1 --act_fun=sigmoid --fusion={fusion_options[ds]} --n_clients=10 --lrate={lr} "
            commands.append(cmd)

            # FL approx-sigmoid / approx-relu fusion 10 party
            act_fun_last = "approx_relu" if ds == "svhn" else "approx_sigmoid_ls"
            cmd = base_cmd + common_args + f" --batch_size=1 --act_fun={act_fun_last} --hades={fusion_options[ds]} --approx_param=3-1 --lrate={lr} "
            commands.append(cmd)
            print(commands)
        for cmd in commands:
            subprocess.run(cmd, shell=True, check=True)
            
    def _run_experiment_3(self, output_file):
        """Q4 experiment - F_HE vs Time for all layer sizes"""
        fusion_options = [2, 4, 8, 16, 32]
        layer_sizes = [None, 2, 4, 8, 16, 32]  # None = no hidden layers
        
        clients = [1]
        data_amount = 10
        synth_feat_size = 32
        
        commands = []
        for i in range(10):
            for n_clients in clients:
                base_cmd = f"python3 main.py --enc --no_infer --batch_size=1 --lrate=0.1 --epoch=1 --dataset=synth " \
                            f"--n_synth_features={synth_feat_size} --act_fun=approx_sigmoid_ls --approx_params=3-1 --horner --loss_enc " \
                            f"--data_amount={data_amount} --output_file={output_file} --n_clients={n_clients}"
                
                for layer_size in layer_sizes:
                    for fusion in fusion_options:
                        cmd = base_cmd
                        
                        if layer_size is not None:
                            cmd += f" --dims={layer_size}"
                        
                        if fusion != synth_feat_size:
                            cmd += f" --fusion={fusion}"
                        
                        commands.append(cmd)
        
        for cmd in commands:
            subprocess.run(cmd, shell=True, check=True)

    def _run_experiment_32(self, output_file):
        """Q4 experiment - F_HE vs Time for all layer sizes"""
        fusion_options = [2, 4, 8, 16, 32]
        layer_sizes = [None, "16", "16,16", "16,16,16", "16,16,16,16"]

        
        clients = [1]
        data_amount = 10
        synth_feat_size = 32
        
        commands = []
        for i in range(10):
            for n_clients in clients:
                base_cmd = f"python3 main.py --enc --no_infer --epoch=1 --batch_size=1 --lrate=0.1 --epoch=1 --dataset=synth " \
                            f"--n_synth_features={synth_feat_size} --act_fun=approx_sigmoid_ls --approx_params=3-1 --horner --loss_enc " \
                            f"--data_amount={data_amount} --output_file={output_file} --n_clients={n_clients}"
                
                for layer_size in layer_sizes:
                    for fusion in fusion_options:
                        cmd = base_cmd
                        
                        if layer_size is not None:
                            cmd += f" --dims={layer_size}"
                        
                        if fusion != synth_feat_size:
                            cmd += f" --fusion={fusion}"
                        
                        commands.append(cmd)
        
        for cmd in commands:
            subprocess.run(cmd, shell=True, check=True)
            
    def _run_experiment_4(self, output_file):
        """Q5 experiment - One-GI timing analysis"""
        commands = []
        
        base_cmd = (
            "python3 main.py --enc --no_infer --lrate=0.01 --data_amount=10 "
            "--minmax=-1,1 --act_fun=approx_sigmoid_ls "
            "--approx_params=3-5 --horner "
            f"--output_file={output_file}"
        )
        
        datasets = {
            "bcd": {"base": f"{base_cmd} --dataset=bcd --dims=64",
                   "configs": [
                       {"batch_size": 1, "hades": None, "epochs": 1},
                       {"batch_size": 2, "hades": None, "epochs": 2},
                       {"batch_size": 4, "hades": 16, "epochs": 3.5},
                       {"batch_size": 4, "hades": 8, "epochs": 3.5},
                       {"batch_size": 8, "hades": 8, "epochs": 5},
                       {"batch_size": 8, "hades": 4, "epochs": 5},
                       {"batch_size": 16, "hades": 4, "epochs": 10},
                       {"batch_size": 16, "hades": 2, "epochs": 10}
                   ]},
            "mnist": {"base": f"{base_cmd} --dataset=mnist",
                     "configs": [
                         {"batch_size": 1, "hades": 128, "epochs": 1},
                         {"batch_size": 1, "hades": 64, "epochs": 1},
                         {"batch_size": 2, "hades": 128, "epochs": 2},
                         {"batch_size": 2, "hades": 64, "epochs": 2},
                         {"batch_size": 4, "hades": 64, "epochs": 3.5}
                    ]},
            "svhn": {"base": f"{base_cmd} --dataset=svhn --act_fun=approx_relu",
                     "configs": [
                         {"batch_size": 1, "hades": 256, "epochs": 1},
                         {"batch_size": 1, "hades": 128, "epochs": 1},
                         {"batch_size": 2, "hades": 128, "epochs": 2},
                         {"batch_size": 4, "hades": 64, "epochs": 3.5}
                     ]}
        }
        
        for dataset_name, dataset_info in datasets.items():
            for config in dataset_info["configs"]:
                cmd = f"{dataset_info['base']} --epochs={config['epochs']} --batch_size={config['batch_size']}"
                if config['hades'] is not None:
                    cmd += f" --hades={config['hades']}"
                else:
                    cmd += " --n_clients=10"
                commands.append(cmd)
        
        for cmd in commands:
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)
            
    def _run_experiment_5(self, output_file):
        """Q3 experiment - Only PCA vs Hades"""
        dims = {
            "bcd": "16",
            "mnist": "64",
            "svhn": "64,64",
        }
        feat_options = {
            "bcd": [1, 2, 4, 8, 16],
            "mnist": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            "svhn": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        }
        datasets = ["bcd", "mnist", "svhn"]
        act_fun_map = {
            "bcd": "approx_sigmoid_ls",
            "mnist": "approx_sigmoid_ls",
            "svhn": "approx_sigmoid_ls",
        }
        approx_params_map = {
            "bcd": "3-1",
            "mnist": "3-1",
            "svhn": "3-5",
        }
        commands = []
        
        base_cmd = (
            f"python3 main.py --batch_size=1 --data_amount=-1 --epochs=10 "
            f"--lrate=0.1 --mom=0.9 --output_file={output_file}"
        )
        
        for ds in datasets:
            dim = dims[ds]
            act_fun = act_fun_map[ds]
            approx_params = approx_params_map[ds]
            minmax = "-1,1" if ds == "bcd" else None
            for n_feat in feat_options[ds]:
                minmax_arg = f" --minmax={minmax}" if minmax else ""
                commands.append(
                    f"{base_cmd} --dims={dim} --dataset={ds} --act_fun={act_fun} "
                    f"--approx_params={approx_params}{minmax_arg} --pca_ablation={n_feat}"
                )
                commands.append(
                    f"{base_cmd} --dims={dim} --dataset={ds} --act_fun={act_fun} "
                    f"--approx_params={approx_params}{minmax_arg} --hades={n_feat}"
                )
        
        for cmd in commands:
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)
    
    def _run_experiment_11(self, output_file):
        """Reconstruction test experiment - PSNR, SSIM, LPIPS metrics"""
        additional = f" --batch_size=1 --output_file={output_file}"
        datasets = ["bcd", "mnist", "svhn"]
        idlg_types = ["oracle", "zero", "rand"]
        
        fusion_options = {
            "bcd": [0, 16, 29],      # BCD has 30 features, so max fusion is 16, plus |F|-1 = 29
            "mnist": [0, 16, 64, 256, 783],  # MNIST has 784 features, plus |F|-1 = 783
            "svhn": [0, 16, 64, 256, 1024, 3071]  # SVHN has 3072 features, plus |F|-1 = 3071
        }
        
        commands = []

        for ds in datasets:
            base = f"python3 main.py --recon --pca --dataset={ds}" + additional
            fusion_values = fusion_options.get(ds, [0, 16, 64, 256])
            for fusion in fusion_values:
                for idlg in idlg_types:
                    commands.append(base + f" --fusion={fusion} --idlg_type={idlg}")

        for cmd in commands:
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)
    
    def _run_experiment_111(self, output_file):
        """Merged exp1+exp11: iDLG + reconstruction testing with exp11 fusion params"""
        additional = f" --epochs=0.00001 --batch_size=1 --data_amount=-1 --act_fun=sigmoid --lrate=0.1 --mom=0.9 --minmax=0,1 --output_file={output_file}"

        # Use exp1 datasets/idlg types and exp11 fusion parameters.
        datasets = ["bcd", "mnist", "svhn"]
        idlg_types = ["rand"]

        fusion_options = {
            "bcd": [0, 16, 29],      # BCD has 30 features, so max fusion is 16, plus |F|-1 = 29
            "mnist": [0, 16, 64, 256, 783],  # MNIST has 784 features, plus |F|-1 = 783
            "svhn": [0, 16, 64, 256, 1024, 3071]  # SVHN has 3072 features, plus |F|-1 = 3071
        }

        commands = []

        for ds in datasets:
            base = f"python3 main.py --idlg --recon --pca --dataset={ds}" + additional
            fusion_values = fusion_options.get(ds, [0, 16, 64, 256])

            # Use exp1's dimension settings
            if ds == "bcd":
                dim = "16"
            elif ds == "mnist":
                dim = "64"
            elif ds == "svhn":
                dim = "64,64"
            else:
                dim = "64"

            for fusion in fusion_values:
                for idlg in idlg_types:
                    commands.append(base + f" --dims={dim} --fusion={fusion} --idlg_type={idlg}")

        for cmd in commands:
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)

    def run_tests(self):
        if self.args.test > 0:
            self._run_dimensional_tests()
    
    def _run_dimensional_tests(self):
        self.args.epochs = 2
        self.args.no_infer = True
        self.args.net_verbosity = 0
        self.args.dataset = "synth"
        self.args.testing = True
        utils.VERBOSITY = self.args.net_verbosity

        assert self.args.enc
        #assert self.args.dataset != "mnist" and self.args.dataset != "synth"
        
        if self.args.data_amount == -1:
            self.args.data_amount = 1
        
        if self.args.test == 1:
            arg_list = [
                {"batch_size": 1, "dims": None},
                {"batch_size": 1, "dims": "2"},
                {"batch_size": 1, "dims": "4"},
                {"batch_size": 1, "dims": "8"},
                {"batch_size": 1, "dims": "2,2"},
                {"batch_size": 1, "dims": "4,2"},
                {"batch_size": 1, "dims": "4,8"},
                {"batch_size": 1, "dims": "16,8"},
            ]
        elif self.args.test == 2:
            arg_list = [
                {"batch_size": 1, "dims": "8,2,8"},
                {"batch_size": 1, "dims": "2,8,4"},
                {"batch_size": 1, "dims": "4,4,4"},
                {"batch_size": 1, "dims": "4,16,4"},
                {"batch_size": 1, "dims": "2,4,4,4"},
                {"batch_size": 1, "dims": "2,8,4,2"},
                {"batch_size": 1, "dims": "2,8,4,8"},
            ]
        elif self.args.test == 3:
            arg_list = [
                {"n_synth_features": 2, "batch_size": 1, "dims": "4,4"},
                {"n_synth_features": 2, "batch_size": 1, "dims": "16,16"},
                {"n_synth_features": 8, "batch_size": 1, "dims": "8,8"},
                {"n_synth_features": 8, "batch_size": 1, "dims": "32,32"},
                {"n_synth_features": 16, "batch_size": 1, "dims": "16,16"},
                {"n_synth_features": 16, "batch_size": 1, "dims": "8,8"},
                {"n_synth_features": 32, "batch_size": 1, "dims": "8,8"},
                {"n_synth_features": 32, "batch_size": 1, "dims": "64,64"},
            ]
        elif self.args.test == 4:
            arg_list = [
                {"batch_size": 1, "dims": "64,16"},
                {"batch_size": 1, "dims": "128,32"},
                {"batch_size": 1, "dims": "32,128"},
                {"batch_size": 1, "dims": "32,32"},
                {"batch_size": 1, "dims": "64,64"},
                {"batch_size": 1, "dims": "128,32"},
            ]
        elif self.args.test == 5: 
            arg_list = [
                #{"data_amount": 4, "batch_size": 2, "dims": "32,32"},
                {"data_amount": 4, "batch_size": 2, "dims": "64,16"},
                #{"data_amount": 8, "batch_size": 4, "dims": "16,16"},
                #{"data_amount": 8, "batch_size": 2, "dims": "32,64"}
                {"data_amount": 8, "batch_size": 2, "dims": None},
                {"data_amount": 8, "batch_size": 4, "dims": None},
                {"data_amount": 16, "batch_size": 8, "dims": None},
                {"data_amount": 8, "batch_size": 2, "dims": "16"},
                {"data_amount": 8, "batch_size": 4, "dims": "2"},
                {"data_amount": 16, "batch_size": 8, "dims": "4"},
            ]
            
        elif self.args.test == 6:
            self.args.data_amount = 4 # Might need to increase for higher batch_size tests
            arg_list = [
                {"batch_size": 1, "dims": None, "n_clients": 2},
                {"batch_size": 1, "dims": "2", "n_clients": 2},
                {"batch_size": 1, "dims": "4,8", "n_clients": 2},
                {"batch_size": 2, "dims": "4", "n_clients": 2},
                {"batch_size": 4, "dims": "4,4", "n_clients": 2},
                {"batch_size": 4, "dims": "4,8", "n_clients": 4},
            ]
        elif self.args.test == 7: # SlightError
            arg_list = [
                {"batch_size": 1, "dims": "2", "act_func": "approx_sigmoid"},
                {"batch_size": 1, "dims": "4", "act_func": "approx_sigmoid"},
                {"batch_size": 1, "dims": "8,4", "act_func": "approx_sigmoid"},
                {"batch_size": 1, "dims": "8,4,8", "act_func": "approx_sigmoid"}
            ]
        elif self.args.test == 8:
            self.args.n_synth_features = 128
            arg_list = [
                {"batch_size": 1, "dims": "16"},
            ]
        elif self.args.test == 9:
            self.args.n_synth_features = 784
            arg_list = [
                {"batch_size": 1, "dims": None},
                {"batch_size": 1, "dims": "2"}
            ]
        
        self._create_dataset()

        # Force stdout to flush after each print
        sys.stdout = utils.FlushingStdout(sys.stdout)
        
        # Maximum number of concurrent processes (Around 4 GB RAM per process)
        max_concurrent = 2
        
        processes = []
        remaining_tests = [(i, arg) for i, arg in enumerate(arg_list)]
        active_processes = []
        all_results = []
        any_failures = False
        
        while remaining_tests or active_processes:
            while remaining_tests and len(active_processes) < max_concurrent:
                test_idx, arg_entry = remaining_tests.pop(0)
                process = Process(target=self._run_nn_test, args=(arg_entry, test_idx))
                processes.append((test_idx, process))
                active_processes.append((test_idx, process))
                process.start()
                
            still_active = []
            for test_idx, process in active_processes:
                if not process.is_alive():
                    process.join(timeout=0.1)
                    if process.exitcode == 0:
                        print(f"Test {test_idx} (Process {process.pid}) completed successfully")
                        all_results.append((test_idx, process.pid, "success"))
                    elif process.exitcode is None:
                        print(f"Test {test_idx} (Process {process.pid}) was terminated abnormally")
                        all_results.append((test_idx, process.pid, "abnormal termination"))
                        any_failures = True
                    else:
                        print(f"Test {test_idx} (Process {process.pid}) failed with exit code {process.exitcode}")
                        all_results.append((test_idx, process.pid, f"failed: {process.exitcode}"))
                        any_failures = True
                else:
                    still_active.append((test_idx, process))
            
            active_processes = still_active
            
        print("All tests completed:")
        for test_idx, pid, result in all_results:
            print(f"  Test {test_idx} (Process {pid}): {result}")
            
        if any_failures:
            print("Some tests failed. Check the output above for details.")
            sys.exit(1)
    
    def _run_nn_test(self, arg_entry, test_idx):
        pid = os.getpid()
        print(f"Test {test_idx} started (Process {pid}) with args: {arg_entry}")
        try:
            # Set up CKKS context for this process since multiprocessing doesn't share context
            if self.args.enc:
                from src.core.encryption import CKKSContext
                ckks = CKKSContext(self.args.n_clients)
                HadesText.set_ckks_context(ckks)
            
            args_copy = self.args
            args_copy.batch_size = arg_entry["batch_size"]
            args_copy.dims = arg_entry["dims"]

            if "n_synth_features" in arg_entry:
                args_copy.n_synth_features = arg_entry["n_synth_features"]
            if "data_amount" in arg_entry:
                args_copy.data_amount = arg_entry["data_amount"]
            if "act_func" in arg_entry:
                args_copy.act_func = arg_entry["act_func"]
            if "n_clients" in arg_entry:
                args_copy.n_clients = arg_entry["n_clients"]
            
            data_manager = DataManager(self.dataset, args_copy)
            X_train, y_train, X_test, y_test = data_manager.configure_data()
            args_copy.dims = data_manager.setup_dims()
            
            model_factory = ModelFactory(args_copy)
            net = model_factory.get_single_network()
            trainer = BaseTrainer(args_copy)

            client_models = [net]
            for _ in range(args_copy.n_clients - 1):
                if isinstance(net, HENetwork):
                    client_model = HENetwork(net.dims, args_copy, network_label="CT")
                elif hasattr(net, 'network_two') and isinstance(net.network_two, HENetwork):
                    client_model = model_factory.get_fusion_network()
                client_models.append(client_model)
            
            losses_per = trainer.run(client_models, X_train, y_train, X_test, y_test)
            for out in losses_per:
                if not np.isclose(out.data, out.padded_data[0], atol=1e-5):
                    print(f"Test {test_idx} failed (Process {pid})")
                    sys.exit(1)
            print(f"Test {test_idx} completed successfully (Process {pid})")
            sys.exit(0)
        except Exception as e:
            print(f"Test {test_idx} failed (Process {pid}) with error: {str(e)}")
            print(f"Full stack trace for Test {test_idx}:")
            traceback.print_exc()
            sys.exit(1)
