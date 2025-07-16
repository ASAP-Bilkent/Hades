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
from src.core.datasets import BreastCancerDiagnosticDataset, BreastCancerOriginalDataset, MNISTDataset, SyntheticDataset

class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.dataset = None
        
    def _create_dataset(self):
        if self.args.dataset == "bcd":
            self.dataset = BreastCancerDiagnosticDataset(apply_pca=self.args.pca)
        elif self.args.dataset == "bco":
            self.dataset = BreastCancerOriginalDataset(apply_pca=self.args.pca)
        elif self.args.dataset == "mnist":
            self.dataset = MNISTDataset(apply_pca=self.args.pca)
        elif self.args.dataset == "synth":
            self.dataset = SyntheticDataset(n_features=self.args.n_synth_features, 
                                          n_samples=self.args.batch_size*self.args.n_clients*self.args.data_amount, 
                                          apply_pca=self.args.pca)
        
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
        additional = f" --epochs=10 --batch_size=1 --dims=16 --no_enc --data_amount=-1 --act_fun=sigmoid --pca --learning_rate=0.1 --output_file={output_file}"
        datasets = ["mnist"]
        idlg_types = ["cheat", "zero", "rand"]
        commands = []

        for ds in datasets:
            base = f"python3 main.py --dataset={ds}" + additional
            if ds == "bco":
                options = [0, 1, 2, 4, 8]
            elif ds == "bcd":
                options = [0, 1, 2, 4, 8, 16]
            elif ds == "mnist":
                options = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            
            for fusion in options:
                for idlg in idlg_types:
                    commands.append(base + f" --fusion={fusion} --idlg_type={idlg}")

        for cmd in commands:
            subprocess.run(cmd, shell=True, check=True)
            
    def _run_experiment_2(self, output_file):
        """Q2, utility experiment"""
        dims = {
            "bco": "8",
            "bcd": "8",
            "mnist": "64",
        }
        fusion_options = {
            "bco": "8",
            "bcd": "8",
            "mnist": "256",
        }
        learning_rates = {
            "bco": "0.1",
            "bcd": "0.1",
            "mnist": "0.01",
        }
        #datasets = ["bco", "bcd", "mnist"]
        datasets = ["bcd"]
        commands = []
        
        base_cmd = f"python3 main.py --batch_size=1 --no_enc --no_idlg --pca --data_amount=-1 --epoch=10 --output_file={output_file}"
        
        for ds in datasets:
            lr = learning_rates[ds]
            # Centrilized sigmoid no-fusion anam-babam
            commands.append(f"{base_cmd} --dims={dims[ds]} --dataset={ds} --act_fun=sigmoid --fusion=0 --learning_rate={lr}")
            
            # Centrilized sigmoid fusion
            commands.append(f"{base_cmd} --dims={dims[ds]} --dataset={ds} --act_fun=sigmoid --fusion={fusion_options[ds]} --learning_rate={lr}")
            
            # Centrilized approx-sigmoid no-fusion
            commands.append(f"{base_cmd} --dims={dims[ds]} --dataset={ds} --act_fun=approx_sigmoid --fusion=0 --learning_rate={lr}")

            # FL sigmoid fusion 10 party
            commands.append(f"{base_cmd} --dims={dims[ds]} --dataset={ds} --act_fun=sigmoid --fusion={fusion_options[ds]} --n_clients=10 --learning_rate={lr}")

            # FL approx-sigmoid fusion 10 party
            commands.append(f"{base_cmd} --dims={dims[ds]} --dataset={ds} --act_fun=approx_sigmoid --fusion={fusion_options[ds]} --n_clients=10 --learning_rate={lr}")
        
        for cmd in commands:
            subprocess.run(cmd, shell=True, check=True)
            
    def _run_experiment_3(self, output_file):
        """Q3 experiment - F_HE vs Time for all layer sizes"""
        fusion_options = [2, 4, 8, 16, 32]
        layer_sizes = [None, 2, 4, 8, 16, 32]  # None = no hidden layers
        
        clients = [1]
        data_amount = 10
        lr = 0.01
        synth_feat_size = 32
        
        commands = []
        for i in range(10):
            for n_clients in clients:
                base_cmd = f"python3 main.py --batch_size=1 --no_idlg --no_infer --epoch=1 --dataset=synth --act_fun=approx_sigmoid " \
                            f"--learning_rate={lr} --n_synth_features={synth_feat_size} --fake " \
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
        """Q3 experiment - F_HE vs Time for all layer sizes"""
        fusion_options = [2, 4, 8, 16, 32]
        layer_sizes = [None, "16", "16,16", "16,16,16", "16,16,16,16"]
        #layer_sizes = ["16,16,16,16"]
        
        clients = [1]
        data_amount = 10
        lr = 0.01
        synth_feat_size = 32
        
        commands = []
        for i in range(10):
            for n_clients in clients:
                base_cmd = f"python3 main.py --batch_size=1 --no_idlg --no_infer --epoch=1 --dataset=synth --act_fun=approx_sigmoid " \
                            f"--learning_rate={lr} --n_synth_features={synth_feat_size} --fake --approx_params=3-1 " \
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
        """Q4 experiment - One-GI timing analysis"""
        commands = []
        
        base_cmd = f"python3 main.py --no_idlg --no_infer --act_fun=approx_sigmoid --learning_rate=0.01 --data_amount=-1 --fake --n_clients=10 --output_file={output_file}"
        
        datasets = {
            "bco": {"base": f"{base_cmd} --dataset=bco --dims=64 --epoch=2.13", 
                   "configs": [
                       {"batch_size": 1, "fusion": None},
                       {"batch_size": 4, "fusion": None},
                       {"batch_size": 8, "fusion": 8},
                       {"batch_size": 16, "fusion": 4}
                   ]},
            "bcd": {"base": f"{base_cmd} --dataset=bcd --dims=64 --epoch=2.57",
                   "configs": [
                       {"batch_size": 1, "fusion": None},
                       {"batch_size": 2, "fusion": None},
                       {"batch_size": 4, "fusion": 16},
                       {"batch_size": 8, "fusion": 8},
                       {"batch_size": 16, "fusion": 4}
                   ]},
            "mnist": {"base": f"{base_cmd} --dataset=mnist --epoch=0.036",
                     "configs": [
                         {"batch_size": 1, "fusion": 128},
                         {"batch_size": 2, "fusion": 128},
                         {"batch_size": 4, "fusion": 64}
                     ]}
        }
        
        for dataset_name, dataset_info in datasets.items():
            for config in dataset_info["configs"]:
                cmd = f"{dataset_info['base']} --batch_size={config['batch_size']}"
                if config['fusion'] is not None:
                    cmd += f" --fusion={config['fusion']}"
                commands.append(cmd)
        
        for cmd in commands:
            subprocess.run(cmd, shell=True, check=True)
            
    def _run_experiment_10(self, output_file):
        """Q10, MNIST check stable parameters"""
        base_cmd = f"python3 main.py --batch_size=1 --no_enc --no_idlg --data_amount=-1 --epoch=1 --dataset=mnist --act_fun=approx_sigmoid --fusion=0 --output_file={output_file}"
        
        dims_variations = [
            "32", "64", "128", "256", "512", 
            "32,8", "64,8", "64,16", "128,16", "128,32", "256,32", "256,64", "512,64", "512,128"
        ]
        lr_variations = [0.01, 0.03, 0.05, 0.07, 0.1]
        
        commands = []
        for dims in dims_variations:
            for lr in lr_variations:
                commands.append(f"{base_cmd} --dims={dims} --learning_rate={lr}")
        
        with Pool(processes=min(4, len(commands))) as pool:
            results = pool.map(self._run_command, commands)
        
        print("\nExperiment 10 Results:")
        for cmd, returncode, stdout, stderr in results:
            status = "SUCCESS" if returncode == 0 else f"FAILED ({returncode})"
            print(f"{cmd}: {status}")
            if returncode != 0:
                print(f"Error: {stderr}")
    
    def run_tests(self):
        if self.args.test > 0:
            self._run_dimensional_tests()
    
    def _run_dimensional_tests(self):
        self.args.fake = True
        self.args.epochs = 2
        self.args.no_infer = True
        self.args.net_verbosity = 0
        self.args.dataset = "synth"
        self.args.testing = True
        utils.VERBOSITY = self.args.net_verbosity

        assert not self.args.no_enc
        #assert self.args.dataset != "mnist" and self.args.dataset != "synth"
        
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
            if not self.args.no_enc:
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
                    client_model = HENetwork(net.dims, args_copy)
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