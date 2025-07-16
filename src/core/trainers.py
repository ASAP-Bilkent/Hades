from src.layers.activation import *
from src.core.hadestext import *
from src.core.model_factory import ModelFactory
from tqdm import tqdm
from src.core.models import *
import numpy as np
import math
import time
import json
import copy
import sys
import os
import datetime
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import re
from src.core.utils import timer_context, global_timer
from multiprocessing import Pool, cpu_count
import pickle

class BaseTrainer():
    def __init__(self, args):
        self.args = args
        self.results_dir = os.path.join('data', 'results')
        self.n_clients = args.n_clients
        self.is_fusion = self.args.fusion is not None
        self.sigmoid = Activation(args, enc=False).sigmoid

        self.client_models = []
        self.client_data = []

    def run(self, client_models, X_train, y_train, X_test, y_test):
        self.client_models = client_models
        self.client_data = X_train

        start_train_time = time.time()
        self.train_loop(y_train)
        total_train_time = time.time() - start_train_time

        infer_results = None
        if not self.args.no_infer:
            self.client_models[0].eval()
            infer_results = self.inference(X_train, y_train, X_test, y_test)

        if self.args.net_verbosity > 0:
            self._save_results(total_train_time, infer_results)
        
        flattened = []
        for model in self.client_models:
            flattened.extend(model.losses_per)
        return flattened
    
    @staticmethod
    def process_parallel_batch(args):
        client_idx, batch_idx, y_client, shallow_model, client_data, is_fusion, n_clients = args
        import os
        import pickle
        
        from src.core.encryption import CKKSContext
        from src.core.utils import global_timer, timer_context
        worker_ckks = CKKSContext(n_clients)
        HadesText.set_ckks_context(worker_ckks)
        
        worker_timer_data_before = dict(global_timer.timers)
        
        with timer_context('LocalIter'):
            batch_loss, logits, gradients = shallow_model.batch_iteration(client_data, y_client, batch_idx)
        
        with timer_context('UpdateWeights'):
            actual_gradients = []
            for layer in shallow_model.layers:
                if hasattr(layer, 'dW_plain') and layer.dW_plain is not None:
                    actual_gradients.append(layer.dW_plain.copy())
                else:
                    actual_gradients.append(np.zeros_like(layer.W_plain))
        
        worker_timing = {}
        for category in ['LocalIter', 'FeedForward', 'BackProp', 'UpdateWeights', 'Bootstrap']:
            if category in global_timer.timers:
                worker_timing[category] = global_timer.timers[category] - worker_timer_data_before.get(category, 0)
        
        worker_id = f"{client_idx}_{batch_idx}_{os.getpid()}"
        results_file = f"/tmp/worker_results_{worker_id}.pkl"
        
        if isinstance(batch_loss, HadesText):
            if batch_loss.data is not None:
                if hasattr(batch_loss.data, 'shape') and batch_loss.data.shape == ():
                    loss_scalar = float(batch_loss.data)
                else:
                    loss_scalar = float(np.mean(batch_loss.data))
            else:
                loss_scalar = 0.0
        else:
            loss_scalar = float(batch_loss)
        
        results = {
            'batch_loss': loss_scalar,
            'gradients': actual_gradients,
            'worker_id': worker_id,
            'timing': worker_timing
        }
        
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Worker {worker_id}: Saved results to {results_file}")
        
        return results_file

    def average_parameters(self, param_list):
        avg_params = []
        for layer_params in zip(*param_list):
            if isinstance(layer_params[0], HadesText):
                avg_param = HadesText.mean(layer_params)
            else:
                avg_param = np.sum(layer_params, axis=0)
            avg_params.append(avg_param)
        return avg_params
    
    def train_loop(self, y):
        if self.is_fusion:
            num_samples = len(self.client_data[0][0])
        else:
            num_samples = len(self.client_data[0])
        num_batches_per_epoch = math.ceil(num_samples / self.args.batch_size)
        
        total_iterations = int(self.args.epochs * num_batches_per_epoch)
        
        global_timer.set_total_iterations(total_iterations)
        
        iteration_pbar = tqdm(
            total=total_iterations, 
            desc="Iterations", 
            unit="iter",
            disable=self.args.test is not None
        )

        if self.args.parallel_batch_size > 1:
            step_size = self.args.parallel_batch_size
        else:
            step_size = self.args.batch_size

        current_epoch = 0
        epoch_loss = None
        iterations_in_current_epoch = 0

        for iteration in range(total_iterations):
            batch_idx = (iteration % num_batches_per_epoch)
            data_idx = batch_idx * step_size
            
            if batch_idx == 0 and iteration > 0:
                if epoch_loss is not None:
                    epoch_loss = self._finalize_loss(iterations_in_current_epoch, epoch_loss)
                current_epoch += 1
                epoch_loss = None
                iterations_in_current_epoch = 0
            
            client_gradients = []
            for c in range(self.n_clients):
                if self.args.parallel_batch_size > 1:
                    batch_indices = list(range(data_idx, min(data_idx + step_size, num_samples)))
                    
                    with Pool(processes=min(len(batch_indices), cpu_count())) as pool:
                        shallow_model = copy.copy(self.client_models[c])
                        args_list = [(c, j, y[c], shallow_model, self.client_data[c], self.is_fusion, self.n_clients) for j in batch_indices]
                        result_files = pool.map(self.process_parallel_batch, args_list)
                    
                    batch_gradients = []
                    for results_file in result_files:
                        try:
                            with open(results_file, 'rb') as f:
                                results = pickle.load(f)
                            
                            batch_loss = results['batch_loss']
                            gradients = results['gradients']
                            worker_id = results['worker_id']
                            worker_timing = results.get('timing', {})
                            
                            for category, elapsed_time in worker_timing.items():
                                global_timer.timers[category] += elapsed_time
                                global_timer.counts[category] += 1
                            
                            self.client_models[c].losses_per.append(batch_loss)
                            batch_gradients.append(gradients)
                            
                            print(f"Main process: Read results from worker {worker_id}")
                            
                            os.remove(results_file)
                            
                        except Exception as e:
                            print(f"Error reading results file {results_file}: {e}")
                            zero_gradients = [np.zeros_like(layer.W_plain) for layer in self.client_models[c].layers]
                            batch_gradients.append(zero_gradients)
                            self.client_models[c].losses_per.append(0.0)
                    
                    avg_gradients = self.average_parameters(batch_gradients)
                    client_gradients.append(avg_gradients)
                else:
                    batch_loss, logits, gradients = self._process_batch(self.client_models[c], self.client_data[c], y[c], data_idx, self.is_fusion)
                    self.client_models[c].losses_per.append(batch_loss)
                    client_gradients.append(gradients)
            
            self._aggregate_gradients(client_gradients, self.is_fusion, self.args.batch_size)

            batch_loss = self._finalize_loss(self.n_clients)
            epoch_loss = batch_loss if epoch_loss is None else self._add_loss(epoch_loss, batch_loss)
            iterations_in_current_epoch += 1
                
            current_epoch_progress = current_epoch + (iterations_in_current_epoch / num_batches_per_epoch)
            iteration_pbar.set_postfix({
                'Epoch': f"{current_epoch_progress:.2f}", 
                'Loss': batch_loss
            })
            iteration_pbar.update(1)

        if epoch_loss is not None and iterations_in_current_epoch > 0:
            epoch_loss = self._finalize_loss(iterations_in_current_epoch, epoch_loss)

        iteration_pbar.close()

    def inference(self, X_train, y_train, X_test, y_test):
        results = {'acc_train': None, 'acc_test': None, 'idlg_train': None, 'idlg_test': None}
        def infer_loop(X, y, split_name):
            if self.is_fusion:
                num_samples = len(X[0][0])
            else:
                num_samples = len(X[0])

            logits_list = []
            for i in range(0, num_samples, self.args.batch_size):
                _, logits, _ = self._process_batch(self.client_models[0], X[0], y[0], i, self.is_fusion)
                self._append_logits(logits, logits_list)

            preds = self._binary_classifier_criterion(logits_list, threshold=0.5)
            acc = np.mean(preds == np.argmax(y[0], axis=1))
            print(f"Accuracy {split_name}:", acc)
            results['acc_' + split_name] = acc

            if not self.args.no_idlg:
                idlg_mse, fidelity = self._run_idlg(self.client_models[0], X[0], y[0])
                print(f"iDLG MSE {split_name}:", idlg_mse)
                results['idlg_' + split_name] = idlg_mse
                results['fidelity_easy_' + split_name] = fidelity[0]
                results['fidelity_hard_' + split_name] = fidelity[1]

        infer_loop(X_train, y_train, "train")
        if not self.args.no_test:
            infer_loop(X_test, y_test, "test")

        return results

    def _binary_classifier_criterion(self, logits, threshold=0.5):
        logits_array = np.array(logits)
        if logits_array.ndim == 1:
            logits_array = logits_array.reshape(1, -1)
        return np.argmax(logits_array, axis=1)

    def _save_results(self, total_train_time, infer_results):
        exp_match = re.match(r'(exp\d+)-', self.args.output_file)
        
        if exp_match:
            exp_name = exp_match.group(1)
            exp_dir = os.path.join(self.results_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            results_path = os.path.join(exp_dir, f"{self.args.output_file}.json")
        else:
            os.makedirs(self.results_dir, exist_ok=True)
            results_path = os.path.join(self.results_dir, f"{self.args.output_file}.json")
        
        infer_keys = ["acc"]
        if not self.args.no_idlg:
            infer_keys.extend(["idlg", "fidelity_easy", "fidelity_hard"])

        timing_data = global_timer.get_timing_table()
        total_timing = next((row for row in timing_data if row['Category'] == 'LocalIter'), None)
        
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cli_command": " ".join(sys.argv),
            "total_train_time_s": total_train_time,
            "time_per_epoch_train_s": total_train_time / self.args.epochs,
        }
        
        if total_timing:
            entry["one_gi_time_s"] = float(total_timing['Per-Client Per-Iteration Time (s)'])
            entry["total_training_time_s"] = float(total_timing['Per-Client Total Time (s)'])
        else:
            entry["one_gi_time_s"] = 0.0
            entry["total_training_time_s"] = 0.0

        entry["timing_breakdown"] = {row['Category']: {
            'total_time_s': float(row['Total Time (s)']),
            'per_client_total_time_s': float(row['Per-Client Total Time (s)']),
            'per_client_per_iteration_time_s': float(row['Per-Client Per-Iteration Time (s)']),
            'count': row['Count']
        } for row in timing_data}

        if infer_results is not None:
            for key in infer_keys:
                suffix_list = ["train"] if self.args.no_test else ["train", "test"]
                for suffix in suffix_list:
                    entry[f"{key}_{suffix}"] = infer_results[f"{key}_{suffix}"]
        
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                existing = json.load(f)
        else:
            existing = []
            
        existing.append(entry)
        with open(results_path, "w") as f:
            json.dump(existing, f, indent=4)

    def _append_logits(self, logits, logits_list):
        if isinstance(logits, HadesText):
            logits_list.extend(logits.data.flatten())
        else:
            logits_list.extend(logits)
        return logits_list

    def _process_batch(self, model, X_inputs, y, i, is_fusion):
        with timer_context('LocalIter'):
            if is_fusion:
                return model.fusion_batch_iteration(X_inputs, y, i)
            else:
                return model.batch_iteration(X_inputs, y, i)

    def _aggregate_gradients(self, client_gradients, is_fusion, batch_size):
        with timer_context('Combine'):
            if is_fusion:
                all_gradients_one = []
                all_gradients_two = []
                for gradients_one, gradients_two in client_gradients:
                    all_gradients_one.append(gradients_one)
                    all_gradients_two.append(gradients_two)
                
                avg_gradients_one = self.average_parameters(all_gradients_one)
                avg_gradients_two = self.average_parameters(all_gradients_two)
                
                for client in self.client_models:
                    for layer, gradient in zip(client.network_one.layers, avg_gradients_one):
                        layer.update_weights(gradient, client.learning_rate, batch_size)
                    for layer, gradient in zip(client.network_two.layers, avg_gradients_two):
                        layer.update_weights(gradient, client.learning_rate, batch_size)
            else:
                avg_gradients = self.average_parameters(client_gradients)
                        
                for client in self.client_models:
                    for layer, gradient in zip(client.layers, avg_gradients):
                        layer.update_weights(gradient, client.learning_rate, batch_size)

    def _add_loss(self, current_loss, new_loss):
        added_losses = []
        for client in self.client_models:
            added_losses.append(client.add_loss(current_loss, new_loss))
        return np.mean(added_losses, axis=0)

    def _finalize_loss(self, divider, loss=None):
        if self.n_clients > 1: # == 1 still works but needless compute
            finalized_losses = []
            for client in self.client_models:
                if loss is None:
                    current_loss = client.losses_per[-1]
                    if isinstance(current_loss, (int, float)):
                        finalized_losses.append(current_loss / divider)
                    else:
                        finalized_losses.append(client.finalize_loss(current_loss, divider))
                else:
                    finalized_losses.append(loss / divider)
            return np.sum(finalized_losses, axis=0)
        else:
            if loss is None:
                current_loss = self.client_models[0].losses_per[-1]
                if isinstance(current_loss, (int, float)):
                    return current_loss / divider
                else:
                    return self.client_models[0].finalize_loss(current_loss, divider)
            else:
                return loss / divider
    
    def _process_idlg_instance(self, args):
            model, gt_data, gt_label, iteration = args
            model = copy.deepcopy(model)

            if isinstance(model, BaseFusionNetwork):
                gt_data = [np.array([gt_data[0]]), np.array([gt_data[1]])]
                gt_label = np.array([gt_label])
                dummy_shape = gt_data[0].shape
            else:
                gt_data, gt_label = np.array([gt_data]), np.array([gt_label])
                dummy_shape = gt_data.shape

            original_dy_dx = model.compute_idlg_gradients(gt_data, gt_label)

            dummy_data = np.random.uniform(low=0, high=1, size=dummy_shape)
            dummy_data_flat = dummy_data.ravel()

            label_pred = gt_label

            def objective(dummy_data_flat):
                dummy_data = dummy_data_flat.reshape(dummy_shape)
                if isinstance(model, BaseFusionNetwork):
                    if self.args.idlg_type == "zero":
                        dummy_dy_dx = model.compute_idlg_gradients([dummy_data, np.zeros(gt_data[1].shape)], label_pred)
                    elif self.args.idlg_type == "rand":
                        dummy_dy_dx = model.compute_idlg_gradients([dummy_data, np.random.uniform(low=0, high=1, size=gt_data[1].shape)], label_pred)
                    else:  # "cheat" mode
                        dummy_dy_dx = model.compute_idlg_gradients([dummy_data, gt_data[1]], label_pred)
                else:
                    dummy_dy_dx = model.compute_idlg_gradients(dummy_data, label_pred)

                grad_diff = sum(np.sum((dg - og) ** 2) for dg, og in zip(dummy_dy_dx, original_dy_dx))
                return grad_diff

            res = minimize(objective, dummy_data_flat, method='L-BFGS-B', options={'maxiter': iteration})
            optimized_dummy_data = res.x.reshape(dummy_shape)
            if isinstance(model, BaseFusionNetwork):
                mse = np.mean((gt_data[0] - optimized_dummy_data) ** 2)
            else:
                mse = np.mean((gt_data - optimized_dummy_data) ** 2)
            return mse
    
    def _run_idlg(self, model, X, y):
        iteration = 300

        if isinstance(model, BaseFusionNetwork):
            if self.args.dataset == "mnist":
                indices = np.random.choice(len(X[0]), size=min(100, len(X[0])), replace=False)
                X[0] = X[0][indices]
                X[1] = X[1][indices]
                y = y[indices]

                model = ModelFactory(self.args).get_fusion_network()

            args_list = [(model, [gt_data0, gt_data1], gt_label, iteration) for gt_data0, gt_data1, gt_label in zip(X[0], X[1], y)]
        else:
            if self.args.dataset == "mnist":
                indices = np.random.choice(len(X), size=min(100, len(X)), replace=False)
                X = X[indices]
                y = y[indices]

                model = ModelFactory(self.args).get_single_network()
                
            args_list = [(model, gt_data, gt_label, iteration) for gt_data, gt_label in zip(X, y)]

        mse_all = []
        with ProcessPoolExecutor() as executor:
            for mse in tqdm(executor.map(self._process_idlg_instance, args_list), total=len(y), desc="Running iDLG"):
                mse_all.append(mse)

        def get_fidelity(threshold):
            if isinstance(model, BaseFusionNetwork):
                denom = len(X[0])
            else:
                denom = len(X)
            return float(100 * sum((np.array(mse_all) < threshold).astype(int)) / denom)
        
        fidelity_easy = get_fidelity(0.01)
        fidelity_hard = get_fidelity(0.0001)
        print(f"% Good Fidelity for 0.01: {fidelity_easy:.2f}%")
        print(f"% Good Fidelity for 0.0001: {fidelity_hard:.2f}%")
        avg_mse = np.mean(mse_all)
        return avg_mse, [fidelity_easy, fidelity_hard]