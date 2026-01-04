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
        self.train_labels = []
        self.stopped_due_to_nan = False
        self._latest_train_acc = None
        self._epoch_train_accs = []
        self.last_idlg_results = {}

    def run(self, client_models, X_train, y_train, X_test, y_test):
        self.client_models = client_models
        self.client_data = X_train
        self.train_labels = y_train
        self._epoch_train_accs = []

        start_train_time = time.time()
        self.train_loop(y_train)
        total_train_time = time.time() - start_train_time

        infer_results = None
        if not self.args.no_infer:
            if self.stopped_due_to_nan:
                infer_results = {'acc_train': 0.0, 'acc_test': 0.0, 'idlg_train': None, 'idlg_test': None}
            else:
                if self._epoch_train_accs:
                    acc_str = " --> ".join(f"{acc:.4f}" for acc in self._epoch_train_accs)
                    print(f"Train accuracy per epoch: {acc_str}")
                self.client_models[0].eval()
                infer_results = self.inference(X_train, y_train, X_test, y_test)
                if infer_results and 'idlg_original_data' in infer_results:
                    self.last_idlg_results['idlg_original_data'] = infer_results['idlg_original_data']
                    self.last_idlg_results['idlg_reconstructed_data'] = infer_results['idlg_reconstructed_data']
        elif self.stopped_due_to_nan:
            infer_results = {'acc_train': 0.0, 'acc_test': 0.0, 'idlg_train': None, 'idlg_test': None}

        if self.args.net_verbosity > 0:
            self._save_results(total_train_time, infer_results)
        
        flattened = []
        for model in self.client_models:
            flattened.extend(model.losses_per)
        return flattened
    
    def average_parameters(self, param_list):
        avg_params = []
        for layer_params in zip(*param_list):
            if isinstance(layer_params[0], HadesText):
                avg_param = HadesText.mean(layer_params)
            else:
                avg_param = np.mean(layer_params, axis=0)
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
        self._latest_train_acc = None
        
        iteration_pbar = tqdm(
            total=total_iterations, 
            desc="Iterations", 
            unit="iter",
            disable=self.args.test is not None
        )

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
                    self._update_train_accuracy()
                current_epoch += 1
                epoch_loss = None
                iterations_in_current_epoch = 0
            
            client_gradients = []
            batch_losses = []
            
            for c in range(self.n_clients):
                batch_loss, logits, gradients = self._process_batch(self.client_models[c], self.client_data[c], y[c], data_idx, self.is_fusion)
                self.client_models[c].losses_per.append(batch_loss)
                client_gradients.append(gradients)
            
            epoch_loss = batch_loss if epoch_loss is None else self._add_loss(epoch_loss, batch_loss)
            iterations_in_current_epoch += 1
            
            is_last_iteration = (iteration == total_iterations - 1)
            self._aggregate_gradients(client_gradients, self.is_fusion, self.args.batch_size, is_last_iteration)

            batch_loss = self._finalize_loss(self.n_clients)
            
            if isinstance(batch_loss, (int, float)):
                loss_value = batch_loss
            elif isinstance(batch_loss, np.ndarray):
                loss_value = float(np.mean(batch_loss))
            elif hasattr(batch_loss, 'data') and batch_loss.data is not None:
                if hasattr(batch_loss.data, 'shape') and batch_loss.data.shape == ():
                    loss_value = float(batch_loss.data)
                else:
                    loss_value = float(np.mean(batch_loss.data))
            elif hasattr(batch_loss, 'padded_data'):
                loss_value = float(batch_loss.padded_data[0])
            else:
                try:
                    loss_value = float(batch_loss) if not isinstance(batch_loss, (list, tuple)) else float(np.mean(batch_loss))
                except (ValueError, TypeError):
                    loss_value = np.nan
            
            if np.isnan(loss_value) and not self.args.enc:
                print(f"\nWARNING: Detected NaN loss. Stopping training early at iteration {iteration + 1}.")
                self.stopped_due_to_nan = True
                break
            
            current_epoch_progress = current_epoch + (batch_idx / num_batches_per_epoch)
            loss_display = 'nan' if np.isnan(loss_value) else f"{abs(loss_value):.4e}"
            postfix = {
                'Epoch': f"{current_epoch_progress:.2f}",
                'Loss': loss_display
            }
            if self._latest_train_acc is not None:
                postfix['acc'] = f"{self._latest_train_acc:.4f}"
            iteration_pbar.set_postfix(postfix)
            iteration_pbar.update(1)

        if epoch_loss is not None and iterations_in_current_epoch > 0:
            epoch_loss = self._finalize_loss(iterations_in_current_epoch, epoch_loss)
            self._update_train_accuracy()

        iteration_pbar.close()

    def inference(self, X_train, y_train, X_test, y_test):
        results = {'acc_train': None, 'acc_test': None, 'idlg_train': None, 'idlg_test': None}
        def infer_loop(X, y, split_name):
            if self.is_fusion:
                num_samples = len(X[0][0])
            else:
                num_samples = len(X[0])

            logits_list = []
            num_batches = math.ceil(num_samples / self.args.batch_size)
            with tqdm(total=num_batches, desc=f"Inference ({split_name})", unit="batch", leave=False,
                      disable=self.args.test is not None) as infer_pbar:
                for i in range(0, num_samples, self.args.batch_size):
                    _, logits, _ = self._process_batch(self.client_models[0], X[0], y[0], i, self.is_fusion)
                    self._append_logits(logits, logits_list)
                    infer_pbar.update(1)

            preds = self._binary_classifier_criterion(logits_list, threshold=0.5)
            acc = np.mean(preds == np.argmax(y[0], axis=1))
            print(f"Accuracy {split_name}:", acc)
            results['acc_' + split_name] = acc

            if self.args.idlg and split_name == "train":
                idlg_result = self._run_idlg(self.client_models[0], X[0], y[0])
                if len(idlg_result) == 4:
                    idlg_mse, fidelity, original_data_list, reconstructed_data_list = idlg_result
                    results['idlg_original_data'] = original_data_list
                    results['idlg_reconstructed_data'] = reconstructed_data_list
                else:
                    idlg_mse, fidelity = idlg_result
                print(f"iDLG MSE {split_name}:", idlg_mse)
                results['idlg_' + split_name] = idlg_mse
                results['fidelity_easy_' + split_name] = fidelity[0]
                results['fidelity_hard_' + split_name] = fidelity[1]

        infer_loop(X_train, y_train, "train")
        if not self.args.no_test:
            infer_loop(X_test, y_test, "test")

        return results

    def _update_train_accuracy(self):
        acc = self._compute_train_accuracy()
        if acc is not None:
            self._latest_train_acc = acc
            self._epoch_train_accs.append(acc)


    def _compute_train_accuracy(self):
        if self.args.no_infer or self.args.enc:
            return None
        if not self.client_models or not self.client_data or not self.train_labels:
            return None
        model = self.client_models[0]
        was_eval = model.eval_mode
        model.eval()
        acc = self._evaluate_dataset_accuracy(model, self.client_data[0], self.train_labels[0])
        if not was_eval:
            model.train()
        return acc

    def _evaluate_dataset_accuracy(self, model, X_split, y_split):
        if X_split is None or y_split is None:
            return None
        logits_list = []
        if self.is_fusion:
            num_samples = len(X_split[0])
        else:
            num_samples = len(X_split)
        if num_samples == 0:
            return None
        for i in range(0, num_samples, self.args.batch_size):
            _, logits, _ = self._process_batch(model, X_split, y_split, i, self.is_fusion)
            self._append_logits(logits, logits_list)
        preds = self._binary_classifier_criterion(logits_list, threshold=0.5)
        targets = y_split if y_split.ndim == 1 else np.argmax(y_split, axis=1)
        return float(np.mean(preds == targets))

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
        if self.args.idlg:
            infer_keys.extend(["idlg", "fidelity_easy", "fidelity_hard"])

        timing_data = global_timer.get_timing_table()
        total_timing = next((row for row in timing_data if row['Category'] == 'LocalIter'), None)
        
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
                    if self.stopped_due_to_nan and key == "acc":
                        entry[f"{key}_{suffix}"] = 0.0
                    else:
                        entry[f"{key}_{suffix}"] = infer_results.get(f"{key}_{suffix}", None)
        
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

    def _aggregate_gradients(self, client_gradients, is_fusion, batch_size, is_last_iteration=False):
        with timer_context('Combine'):
            if is_fusion:
                all_gradients_one = []
                all_gradients_two = []
                for gradients_one, gradients_two in client_gradients:
                    all_gradients_one.append(gradients_one)
                    all_gradients_two.append(gradients_two)
                
                avg_gradients_one = self.average_parameters(all_gradients_one)
                avg_gradients_two = self.average_parameters(all_gradients_two)

                for client_idx, client in enumerate(self.client_models):
                    gradients_one = avg_gradients_one
                    for i, (layer, gradient) in enumerate(zip(client.network_one.layers, gradients_one)):
                        is_last = (i == len(client.network_one.layers) - 1)
                        layer.update_weights(gradient, client.learning_rate, batch_size, is_last=is_last, is_last_iteration=is_last_iteration)
                    for i, (layer, gradient) in enumerate(zip(client.network_two.layers, avg_gradients_two)):
                        is_last = (i == len(client.network_two.layers) - 1)
                        try:
                            layer.update_weights(gradient, client.learning_rate, batch_size, is_last=is_last, is_last_iteration=is_last_iteration)
                        except TypeError:
                            layer.update_weights(gradient, client.learning_rate, batch_size, is_last_iteration=is_last_iteration)
            else:
                avg_gradients = self.average_parameters(client_gradients)
                label = self._infer_gradient_label(self.client_models[0])

                for client in self.client_models:
                    for i, (layer, gradient) in enumerate(zip(client.layers, avg_gradients)):
                        is_last = (i == len(client.layers) - 1)
                        layer.update_weights(gradient, client.learning_rate, batch_size, is_last=is_last, is_last_iteration=is_last_iteration)

    def _add_loss(self, current_loss, new_loss):
        added_losses = []
        for client in self.client_models:
            added_losses.append(client.add_loss(current_loss, new_loss))
        
        if any(isinstance(x, HadesText) for x in added_losses):
            if len(added_losses) == 1:
                return added_losses[0]
            sum_loss = added_losses[0]
            for loss in added_losses[1:]:
                sum_loss = sum_loss.add(loss, "CC")
            divisor = HadesText(1.0 / len(added_losses))
            return sum_loss.mult(divisor, "CD")
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
                    if isinstance(loss, (int, float)):
                        finalized_losses.append(loss / divider)
                    else:
                        finalized_losses.append(client.finalize_loss(loss, divider))
            return np.sum(finalized_losses, axis=0)
        else:
            if loss is None:
                current_loss = self.client_models[0].losses_per[-1]
                if isinstance(current_loss, (int, float)):
                    return current_loss / divider
                else:
                    return self.client_models[0].finalize_loss(current_loss, divider)
            else:
                if isinstance(loss, (int, float)):
                    return loss / divider
                else:
                    return self.client_models[0].finalize_loss(loss, divider)

    def _infer_gradient_label(self, model):
        label = getattr(model, 'network_label', None)
        if label:
            return label
        if isinstance(model, HENetwork):
            return "CT"
        return "PT"


    
    def _process_idlg_instance(self, args):
            model, gt_data, gt_label, iteration = args
            model = copy.deepcopy(model)
            model.eval()

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
                    else:  # "oracle" mode
                        dummy_dy_dx = model.compute_idlg_gradients([dummy_data, gt_data[1]], label_pred)
                else:
                    dummy_dy_dx = model.compute_idlg_gradients(dummy_data, label_pred)
                grad_diff = sum(np.sum((dg - og) ** 2) for dg, og in zip(dummy_dy_dx, original_dy_dx))
                return grad_diff

            res = minimize(objective, dummy_data_flat, method='L-BFGS-B', options={'maxiter': iteration})
            optimized_dummy_data = res.x.reshape(dummy_shape)
            if isinstance(model, BaseFusionNetwork):
                mse = np.mean((gt_data[0] - optimized_dummy_data) ** 2)
                original_data = gt_data[0].flatten()
            else:
                mse = np.mean((gt_data - optimized_dummy_data) ** 2)
                original_data = gt_data.flatten()
            reconstructed_data = optimized_dummy_data.flatten()
            return mse, original_data, reconstructed_data
    
    def _run_idlg(self, model, X, y):
        iteration = 300

        if isinstance(model, BaseFusionNetwork):
            if self.args.dataset == "mnist" or self.args.dataset == "svhn":
                indices = np.random.choice(len(X[0]), size=min(100, len(X[0])), replace=False)
                X0_subset = X[0][indices]
                X1_subset = X[1][indices]
                y_subset = y[indices]
                X = [X0_subset, X1_subset]
                y = y_subset

                #model = ModelFactory(self.args).get_fusion_network()

            args_list = [(model, [gt_data0, gt_data1], gt_label, iteration) for gt_data0, gt_data1, gt_label in zip(X[0], X[1], y)]
        else:
            if self.args.dataset == "mnist" or self.args.dataset == "svhn":
                indices = np.random.choice(len(X), size=min(100, len(X)), replace=False)
                X = X[indices]
                y = y[indices]

                #model = ModelFactory(self.args).get_single_network()
                
            args_list = [(model, gt_data, gt_label, iteration) for gt_data, gt_label in zip(X, y)]

        mse_all = []
        reconstructed_data_list = []
        original_data_list = []
        with ProcessPoolExecutor() as executor:
            for result in tqdm(executor.map(self._process_idlg_instance, args_list), total=len(y), desc="Running iDLG"):
                if isinstance(result, tuple) and len(result) == 3:
                    mse, original_data, reconstructed_data = result
                    mse_all.append(mse)
                    original_data_list.append(original_data)
                    reconstructed_data_list.append(reconstructed_data)
                else:
                    mse_all.append(result)

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
        
        if original_data_list and reconstructed_data_list:
            return avg_mse, [fidelity_easy, fidelity_hard], original_data_list, reconstructed_data_list
        else:
            return avg_mse, [fidelity_easy, fidelity_hard]
