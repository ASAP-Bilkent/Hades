import numpy as np
import time
import json
import sys
import os

from src.core.encryption import CKKSContext
from src.core.hadestext import HadesText
from src.layers.activation import *
from src.layers.layer import *
from src.core.utils import *

class BaseModel:
    def __init__(self, args):
        self.args = args
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.losses_per = []
        self.eval_mode = False
    
    def fusion_batch_iteration(self, X_inputs, batch_y, i):
        batch_X1, batch_X2 = X_inputs[0], X_inputs[1]
        if i != -1:
            batch_X1 = batch_X1[i:i+self.batch_size]
            batch_X2 = batch_X2[i:i+self.batch_size]
            batch_y = batch_y[i:i+self.batch_size]

        with timer_context('FeedForward'):
            activations1, activations2 = self.forward(batch_X1, batch_X2)
        preds1 = activations1[-1]
        preds2 = activations2[-1]
        
        if isinstance(preds2, HadesText):  # HE fusion
            preds2_decrypted = preds2._return_decrypted()[:preds1.shape[-1]]
            fused_preds = (preds1 + preds2_decrypted) / 2
        else:  # Non-HE fusion
            fused_preds = (preds1 + preds2) / 2
        
        batch_loss = np.mean(fused_preds - batch_y)

        grad_loss = (fused_preds - batch_y) / batch_y.size
        
        if hasattr(preds2, 'validation_mask'):
            grad2 = HadesText(grad_loss, mask=preds2.validation_mask)
            grad2._pad_data(pad_step=len(preds2.validation_mask) // self.batch_size)
        else:
            grad2 = grad_loss
        grad1 = grad_loss

        activations1[-1] = grad1
        activations2[-1] = grad2
        
        with timer_context('BackProp'):
            gradients = self.backward(
                batch_X1, batch_X2,
                activations1, activations2
            )

        return batch_loss, fused_preds, gradients

    def batch_iteration(self, batch_X, batch_y, i):
        if i != -1:
            batch_X = batch_X[i:i+self.batch_size]
            batch_y = batch_y[i:i+self.batch_size]

        with timer_context('FeedForward'):
            activations = self.forward(batch_X)
        error, batch_loss = self.mse_loss(activations, batch_y)

        grad_loss = self.mse_loss_grad(error, activations, batch_y)
        with timer_context('BackProp'):
            gradients = self.backward(batch_X, grad_loss)

        return batch_loss, activations[-1], gradients
    
    def _create_or_load_results(self, results_folder, results_path):
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if os.path.exists(results_path):
            with open(results_path, mode="r") as jsonfile:
                results_data = json.load(jsonfile)
        else:
            results_data = []

        return results_data

    def _save_results(self, path, data, start_time):
        total_time = time.time() - start_time

        data.append({
            "last_epoch_loss": self.losses_per[-1],
            "total_time_s": total_time,
            "time_per_epoch_s": total_time / self.epochs,
            "cli_command": " ".join(sys.argv)
        })

        with open(path, mode="w") as jsonfile:
            json.dump(data, jsonfile, indent=4)
    
    def forward(self, X):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def backward(self, X, activations):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def add_loss(self, current_loss, new_loss):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def finalize_loss(self, total_loss, num_batches):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eval(self):
        self.eval_mode = True

class HENetwork(BaseModel):
    def __init__(self, dims, args):
        super().__init__(args)
        np.random.seed(42)

        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        self.dim_slots = []
        for i in range(len(dims)):
            self.dim_slots.append(next_power_of_two(dims[i]))

        dim_slots_and_bs = [self.batch_size] + self.dim_slots
        if args.verbosity > 0:
            print("Dim slots and bs:", dim_slots_and_bs)

        HadesText.set_verbosity(args.verbosity)
        HadesText.set_fake(args.fake)

        self.dims = dims

        self.matrix_sizes = [self.dim_slots[i] * self.dim_slots[i+1] for i in range(len(self.dim_slots)-1)] + [self.dim_slots[len(self.dim_slots)-1]]
        self.slot_sizes = [max(self.matrix_sizes[i:]) for i in range(len(self.matrix_sizes))]
        self.max_slot_size = max(self.slot_sizes)
        self.slot_multipliers = [max(self.slot_sizes) // matrix_size for matrix_size in self.matrix_sizes]
        HadesText.verbose_print(f"matrix_sizes: {self.matrix_sizes}", 2)
        HadesText.verbose_print(f"slot_sizes: {self.slot_sizes}", 2)
        HadesText.verbose_print(f"slot_multipliers: {self.slot_multipliers}", 2)
        
        activation = Activation(args, enc=True)
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(LayerHE(i + 1, dims, self.dim_slots, self.max_slot_size, self.slot_multipliers, activation, self.args.learning_rate, getattr(self.args, 'nest', False)))
        
        self.cur_data_batch_size = None
        self.ciphertext_size = None

    def forward(self, X):
        n_cols = X.shape[1]
        next_pow2 = 2**int(np.ceil(np.log2(n_cols)))
        pad_width = ((0, 0), (0, next_pow2 - n_cols))
        X = np.pad(X, pad_width, mode='constant', constant_values=0)
        X = HadesText(X)

        self.cur_data_batch_size = X.shape[0]
        
        window_size = 2
        for i in range(len(self.dim_slots) - window_size + 1):
            window = self.dim_slots[i:i+window_size]
            product = window[0] * window[1] * self.cur_data_batch_size
            if product > 4096:
                raise ValueError(f"Product of {self.cur_data_batch_size} * {window[0]} * {window[1]} = {product}, which exceeds 4096.")

        z = [X]
        for i in range(len(self.layers)):
            z.append(self.layers[i].forward(z[-1], self.cur_data_batch_size))
        z = z[1:] # Don't need first for gradient calc

        return z
    
    def get_error(self, activations, targets):
        preds = activations[-1]

        if len(self.dim_slots) % 2 == 0:
            targets = HadesText(targets.reshape(-1, 1), mask=preds.validation_mask)
            targets._pad_data(pad_step=(self.max_slot_size // self.dim_slots[-1]))
        else:
            targets = HadesText(targets, mask=preds.validation_mask)
            targets._pad_data("row", pad_step=2 * (self.max_slot_size // self.dim_slots[-1]))

        targets.dump_info("targets")
        error = preds.subtract(targets, "CP") # Bx1
        error.dump_info("error")

        error.data = preds.data - targets.data.reshape(self.cur_data_batch_size, -1)
        HadesText.verbose_print(f"error_nonenc:{error.data}", 1)
        return error
    
    def mse_loss(self, activations, targets):
        HadesText.verbose_print(f"{'-' * 50}Calculating loss{'-' * 50}", 1)
        error = self.get_error(activations, targets)

        #error = error.mult(error) # Bx1
        if self.args.fake_loss:
            preds = activations[-1]
            loss_divided = np.mean((preds.data - targets) ** 2)
        else:
            #error = error.mult(error) # Bx1
            #error.dump_info("sError")

            #error.data = error.data**2
            #HadesText.verbose_print(f"squared_errors_nonenc:{error.data}", 1)

            if len(self.dim_slots) % 2 == 0:
                errorRotated = error._rotate_ciphertext("left", self.max_slot_size // self.dim_slots[-1], self.dim_slots[-1], clean=False)
            else:
                errorRotated = error._rotate_ciphertext("left", 1, self.dim_slots[-1], clean=False)

            errorRotated.dump_info("sErrorScalar")
            
            divider = HadesText(1 / (self.cur_data_batch_size * self.dim_slots[-1]))
            loss_divided = errorRotated.mult(divider, "CD")
            loss_divided.dump_info("loss_divided")
            loss_divided.data = np.mean(loss_divided.data)

            if self.cur_data_batch_size > 1:
                loss_divided = loss_divided._rotate_ciphertext("left", self.max_slot_size, self.cur_data_batch_size, clean=False)
                loss_divided.dump_info("loss_divided_summed")

            HadesText.verbose_print(f"loss_nonenc:{loss_divided.data}", 1)
            
            #if self.args.test is not None:
            if self.args.data_amount != -1:
                print(loss_divided.data, loss_divided.padded_data[0])
            if not np.isclose(loss_divided.data, loss_divided.padded_data[0], atol=1e-5):
                print("Result mismatch")

        return error, loss_divided

    def mse_loss_grad(self, error, activations, targets):
        HadesText.verbose_print(f"{'-' * 50}Calculating loss grad{'-' * 50}", 1)
        z = activations
        #error = self.get_error(z, targets)

        #divider = HadesText(2 / (self.cur_data_batch_size * self.dim_slots[-1]))
        divider = HadesText(1 / (self.cur_data_batch_size * self.dim_slots[-1]))
        grad = error.mult(divider, "CD") # Bx1 (2ceil(N*B)x1 cipher)
        grad.dump_info("grad")

        #grad.data = 2 * error.data / targets.size
        grad.data = error.data / targets.size
        HadesText.verbose_print(f"grad_nonenc:{grad.data}", 1)
        
        z[-1] = grad
        return z

    def backward(self, X, g): 
        z = g
        g = z[-1]
        gradients = []

        for i in range(len(self.layers) - 1, 0, -1):
            g, dW = self.layers[i].backward(z[i-1], g, self.cur_data_batch_size)
            gradients.append(dW)
            
        XT = HadesText(X.T) # NxB
        XT._pad_data("col", pad_step=self.dim_slots[0], pad_batch= -1 * self.slot_sizes[0] // self.dim_slots[0])
        XT.dump_info("XT")
        _, dW = self.layers[0].backward(XT, g, self.cur_data_batch_size)
        gradients.append(dW)

        return list(reversed(gradients))

    def add_loss(self, current_loss, new_loss):
        return current_loss + new_loss

    def finalize_loss(self, total_loss, num_batches):
        if self.args.fake_loss:
            return total_loss / num_batches
        else:
            total_loss.dump_info("total_loss")
            return total_loss.padded_data[0] / num_batches
    
    def get_gradients(self):
        gradients = []
        for layer in self.layers:
            gradients.append(layer.get_gradients())
        return gradients

    def set_gradients(self, gradients):
        if len(gradients) != len(self.layers):
            raise ValueError("Number of gradient matrices must match number of layers.")
        for layer, gradient in zip(self.layers, gradients):
            layer.dW = gradient

class Network(BaseModel):
    def __init__(self, dims, args):
        super().__init__(args)
        np.random.seed(42)
        activation = Activation(args, enc=False)
        self.layers = [Layer(activation, dims[i], dims[i + 1], getattr(args, 'nest', False)) for i in range(len(dims) - 1)]
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size

    def forward(self, X):
        activations = [X]  # Include input X in activations
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                X = layer.forward(X, apply_activation=False)
            else:
                X = layer.forward(X)
            activations.append(X)
        return activations  # Return all activations including input

    def backward(self, X, activations):
        grad_loss = activations[-1]
        gradients = []
        
        for i in reversed(range(len(self.layers))):
            input_activation = activations[i]
            if i == len(self.layers) - 1:
                grad_loss, dW = self.layers[i].backward(grad_loss, input_activation, self.learning_rate, apply_activation=False, update_weight=not self.eval_mode)
            else:
                grad_loss, dW = self.layers[i].backward(grad_loss, input_activation, self.learning_rate, update_weight=not self.eval_mode)
            gradients.append(dW)
        return list(reversed(gradients))

    def mse_loss(self, activations, targets):
        #return np.mean((activations[-1] - targets) ** 2)
        return activations[-1] - targets, np.mean((activations[-1] - targets))

    def mse_loss_grad(self, error, activations, targets):
        z = activations.copy()
        #grad = (2 / targets.size) * (activations[-1] - targets)
        grad = (1 / targets.size) * (error)
        z[-1] = grad
        return z
    
    def add_loss(self, current_loss, new_loss):
        return current_loss + new_loss

    def finalize_loss(self, total_loss, num_batches):
        return total_loss / num_batches

    def get_gradients(self):
        gradients = []
        for layer in self.layers:
            gradients.append(layer.get_gradients())
        return gradients
    
    def compute_idlg_gradients(self, data, target):
        _, _, gradients = self.batch_iteration(data, target, -1)
        return gradients

class BaseFusionNetwork(BaseModel):
    def __init__(self, args, dims_one, dims_two):
        super().__init__(args)
        self.network_one = Network(dims_one, args)
        self.network_two = Network(dims_two, args)

    def forward(self, X_one, X_two):
        activations_one = self.network_one.forward(X_one)
        activations_two = self.network_two.forward(X_two)
        return activations_one, activations_two

    def backward(self, X_one, X_two, activations_one, activations_two):
        gradients_one = self.network_one.backward(X_one, activations_one)
        gradients_two = self.network_two.backward(X_two, activations_two)
        return gradients_one, gradients_two

    def mse_loss(self, preds, targets):
        loss_one = self.network_one.mse_loss(preds[0], targets)
        loss_two = self.network_two.mse_loss(preds[1], targets)
        return loss_one, loss_two

    def mse_loss_grad(self, preds, targets):
        grad_loss_one = self.network_one.mse_loss_grad(preds[0], targets)
        grad_loss_two = self.network_two.mse_loss_grad(preds[1], targets)
        return grad_loss_one, grad_loss_two

    def get_gradients(self):
        return (self.network_one.get_gradients(), 
                self.network_two.get_gradients())

    def set_gradients(self, gradients):
        self.network_one.set_gradients(gradients[0])
        self.network_two.set_gradients(gradients[1])

    def compute_idlg_gradients(self, data, target):
        _, _, gradients = self.fusion_batch_iteration(data, target, -1)
        return gradients[0]  # Return only network_one gradients

    def eval(self):
        self.eval_mode = True
        self.network_one.eval()
        self.network_two.eval()

class FusionNetwork(BaseFusionNetwork):
    def __init__(self, args, dims_one, dims_two):
        super().__init__(args, dims_one, dims_two)
        self.network_two = HENetwork(dims_two, args) # Replace network_two with HENetwork

    def finalize_loss(self, total_loss, num_batches):
        return total_loss / num_batches
    
    def add_loss(self, current_loss, new_loss):
        return current_loss + new_loss

class NoEncFusionNetwork(BaseFusionNetwork):
    def __init__(self, args, dims_one, dims_two):
        super().__init__(args, dims_one, dims_two)

    def finalize_loss(self, total_loss, num_batches):
        return total_loss / num_batches
    
    def add_loss(self, current_loss, new_loss):
        return current_loss + new_loss
