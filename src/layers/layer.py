import numpy as np
from src.core.hadestext import HadesText
from src.core.utils import *

class LayerHE():
    
    def __init__(self, rank, dims, dim_slots, max_slot_size, slot_multipliers, activation, lr, momentum=0.0, network_label=None):
        self.rank = rank
        self.dim_slots = dim_slots
        self.max_slot_size = max_slot_size
        self.slot_multipliers = slot_multipliers
        self.activation = activation
        self.lr = HadesText(lr)
        self.network_label = network_label
        self.momentum = momentum
        
        self.in_dim = dims[rank - 1]
        self.out_dim = dims[rank]
        self.padded_in_dim = dim_slots[rank - 1]
        self.padded_out_dim = dim_slots[rank]
        
        std = np.sqrt(2.0 / (self.in_dim + self.out_dim))
        W = np.random.randn(self.in_dim, self.out_dim) * std
        self.W_plain = np.pad(W, ((0, self.padded_in_dim - self.in_dim), (0, self.padded_out_dim - self.out_dim)), mode='constant')
        self.W = HadesText(self.W_plain)
        self.velocity_plain = np.zeros_like(self.W_plain)
        self.velocity = HadesText(np.zeros_like(self.W_plain))
        
        self.last = True if self.rank == len(self.dim_slots) - 1 else False

    @profile
    def forward(self, inp, cur_data_batch_size):
        HadesText.verbose_print(f"{'-' * 50}Calculating z{self.rank}{'-' * 50}", 1)
        layer_rank = self.rank - 1

        slot_multiplier = self.slot_multipliers[self.rank - 1]
        HadesText.verbose_print(f"slot_multiplier: {slot_multiplier}", 2)
        pad_step = slot_multiplier * self.W.shape[0] if self.rank % 2 == 1 else slot_multiplier * self.W.shape[1]
        HadesText.verbose_print(f"pad_step: {pad_step}", 2)
        
        if self.W._padded_data is None:
            order = "col" if self.rank % 2 == 1 else "row"
            try:
                self.W._pad_data(order, pad_step=pad_step, pad_batch=cur_data_batch_size)
            except:
                pass
            self.W.dump_info(f"W{self.rank}")

        mult_types = "PC" if self.rank == 1 else "CC"

        if self.rank % 2 == 0:
            rotate = [pad_step, self.dim_slots[self.rank - 1]]
        else:
            rotate = [1, self.dim_slots[self.rank - 1]]

        z = inp.multiply(self.W, mult_types, rotate)

        inp.dump_info(f"inp{self.rank}")
        self.W.dump_info(f"W{self.rank}")
        z.dump_info(f"z{self.rank}")

        z.data = np.matmul(inp.data, self.W_plain)
        HadesText.verbose_print(f"z{self.rank}_noenc:{z.data}", 1)

        if not self.last:
            z.data = self.activation.plain_forward(
                z.data,
                layer_id=layer_rank,
                network_label=self.network_label,
                layer_rank=layer_rank
            )
            z = self.activation.forward(
                z,
                layer_id=layer_rank,
                network_label=self.network_label,
                layer_rank=layer_rank
            )

            z.dump_info(f"z{self.rank}_activated")
            HadesText.verbose_print(f"z{self.rank}_activated_noenc:{z.data}", 1)            

            if self.rank % 2 == 1:
                # Rotate so its rows align with the next layer's(rank+1) columns
                z = z._rotate_ciphertext("right", 1, self.dim_slots[self.rank+1]) 
            else:
                next_n_rows = self.dim_slots[self.rank]
                next_n_cols = self.dim_slots[self.rank + 1]
                next_slot_multiplier = self.slot_multipliers[self.rank]
                HadesText.verbose_print(f"next_slot_multiplier: {next_slot_multiplier}", 2)
                HadesText.verbose_print(f"next_n_rows: {next_n_rows}", 2)
                HadesText.verbose_print(f"next_n_cols: {next_n_cols}", 2)

                if (self.rank + 1) % 2 == 1:
                    next_n_rows = next_n_rows * next_slot_multiplier
                else:
                    next_n_cols = next_n_cols * next_slot_multiplier

                HadesText.verbose_print(f"rank: {self.rank}, next_n_rows: {next_n_rows}, next_n_cols: {next_n_cols}", 2)
                z = z._rotate_ciphertext("right", next_n_rows, next_n_cols)
            z.dump_info(f"z{self.rank}_rotated")

            self.preZ = z
        return z
    
    @profile
    def backward(self, z, g, cur_data_batch_size):
        HadesText.verbose_print(f"{'-' * 50}Calculating z{self.rank-1}g{'-' * 50}", 1)
        z.dump_info(f"z{self.rank-1}")
        layer_rank = self.rank - 1

        if not self.last:
            # Apply activation derivative
            if self.activation.args.act_func is not None and self.activation.args.act_func != 'none':
                g = g.mult(self.activation.backward(
                    self.preZ,
                    layer_id=layer_rank,
                    network_label=self.network_label,
                    layer_rank=layer_rank
                ))
                g.data = g.data * self.activation.plain_backward(
                    self.preZ.data,
                    layer_id=layer_rank,
                    network_label=self.network_label,
                    layer_rank=layer_rank
                )
        
        # ------------------------------------------------------------
        # Prep input grad
        # ------------------------------------------------------------
        g.dump_info(f"gin(gW{self.rank+1})")
        if self.rank % 2 == 1:
            g = g._rotate_ciphertext("right", 1, self.dim_slots[self.rank - 1])
        else:
            g = g._rotate_ciphertext("right", self.max_slot_size // self.dim_slots[self.rank - 1], self.dim_slots[self.rank - 1])
        g.dump_info(f"gW{self.rank+1}Rotated")

        dz_plain = g.data # dZ = dA
        HadesText.verbose_print(f"dz{self.rank-1}_plain_noenc:{dz_plain}", 1)

        # ------------------------------------------------------------
        # Prep weight grad
        # ------------------------------------------------------------
        zg = z.mult(g)
        if cur_data_batch_size > 1:
            zg = zg.mult(HadesText(1/cur_data_batch_size), "CD")
            zg = zg._rotate_ciphertext("left", self.max_slot_size, cur_data_batch_size)
        zg.dump_info(f"z{self.rank-1}g")

        try:
            dW_plain = np.matmul(z.data.T, dz_plain) / cur_data_batch_size
        except:
            dW_plain = np.matmul(z.data, dz_plain) / cur_data_batch_size
        dW_plain = np.pad(dW_plain, ((0, self.padded_in_dim - self.in_dim), (0, self.padded_out_dim - self.out_dim)), mode='constant')
        HadesText.verbose_print(f"dW{self.rank}_plain_noenc:{dW_plain}", 1)
        zg.data = dW_plain

        # ------------------------------------------------------------
        # Prep output grad
        # ------------------------------------------------------------
        gW = g
        if self.rank > 1:
            HadesText.verbose_print(f"{'-' * 50}Calculating gW{self.rank}{'-' * 50}", 1)
            g.dump_info(f"gW{self.rank+1}")

            self.W.dump_info(f"W{self.rank}_old")

            if self.rank % 2 == 0:
                rotate_params = [1, self.dim_slots[self.rank]]
            else:
                rotate_params = [self.max_slot_size // self.dim_slots[self.rank], self.dim_slots[self.rank]]
            
            gW = g.matrix_multiply(self.W, "CC", rotate=rotate_params)
            gW.dump_info(f"gW{self.rank}Summed")

            dz_plain = np.matmul(dz_plain, self.W_plain.T)
            gW.data = dz_plain
            HadesText.verbose_print(f"gW{self.rank}_plain_noenc:{dz_plain}", 1)

        return gW, zg

    def update_weights(self, zg, learning_rate, cur_data_batch_size, is_last=False, is_last_iteration=False):
        zg.dump_info(f"zg{self.rank}_debug")
        with timer_context('UpdateWeights'):
            HadesText.verbose_print(f"{'-' * 50}Calculating W{self.rank}{'-' * 50}", 1)
            self.lr = HadesText(learning_rate)
            update_W = zg.mult(self.lr, "CD")
            update_W.data = self.lr.data * zg.data

            self.W.dump_info(f"W{self.rank}_old")
            update_W.dump_info(f"update_W{self.rank}")

            if self.rank > 2:
                rotate_step = self.slot_multipliers[self.rank] * self.dim_slots[self.rank]
                update_W = update_W._rotate_ciphertext("right", rotate_step, cur_data_batch_size)
            else:
                update_W = update_W._rotate_ciphertext("right", self.dim_slots[0] * self.dim_slots[1], cur_data_batch_size)
            update_W.dump_info(f"update_W{self.rank}_rotated")
            HadesText.verbose_print(f"W{self.rank}_plain_update_noenc:{update_W.data}", 1)

            if self.momentum > 0.0:
                prev_velocity_plain = self.velocity_plain.copy()
                self.velocity_plain = self.momentum * self.velocity_plain - learning_rate * zg.data
                self.W_plain = self.W_plain - self.momentum * prev_velocity_plain + (1 + self.momentum) * self.velocity_plain
                
                momentum_ht = HadesText(self.momentum)
                one_plus_momentum_ht = HadesText(1 + self.momentum)
                prev_velocity = self.velocity.copy()
                self.velocity = momentum_ht.mult(self.velocity, "CD").subtract(update_W, "CC")
                self.W = self.W.subtract(momentum_ht.mult(prev_velocity, "CD"), "CC").add(one_plus_momentum_ht.mult(self.velocity, "CD"), "CC")
            else:
                self.W_plain = self.W_plain - self.lr.data * zg.data
                self.W = self.W.subtract(update_W)

            if not is_last_iteration:
                self.W.apply_bootstrapping_if_needed()
            self.W.shape = (self.dim_slots[self.rank - 1], self.dim_slots[self.rank])
            self.W.dump_info(f"W{self.rank}_updated")

        HadesText.verbose_print(f"W{self.rank}_plain_updated_noenc:{self.W_plain}", 1)
        

    def get_plain_weights(self):
        return self.W_plain
    
    def set_plain_weights(self, weights):
        self.W_plain = weights
    
    def update_activation_range(self, new_range):
        if hasattr(self, 'activation') and hasattr(self.activation, 'update_approx_range'):
            try:
                return self.activation.update_approx_range(new_range)
            except ValueError:
                return False
        return False

class Layer:
    _layer_counter = 0
    
    def __init__(self, activation, input_dim, output_dim, network_label=None, momentum=0.0, layer_rank=None):
        std = np.sqrt(2.0 / (input_dim + output_dim))
        self.weights = np.random.randn(input_dim, output_dim) * std
        self.activation = activation
        self.layer_id = Layer._layer_counter
        self.network_label = network_label
        self.momentum = momentum
        self.velocity = np.zeros_like(self.weights)
        self.layer_rank = layer_rank if layer_rank is not None else self.layer_id
        Layer._layer_counter += 1

    def forward(self, X, apply_activation=True):
        self.Z = X @ self.weights
        if apply_activation:
            layer_kwargs = {
                'layer_id': self.layer_id,
                'network_label': self.network_label,
                'layer_rank': self.layer_rank
            }
            if hasattr(self.activation.forward, '__name__') and self.activation.forward.__name__ == 'sigmoid':
                return self.activation.sigmoid(self.Z, **layer_kwargs)
            elif hasattr(self.activation.forward, '__name__') and self.activation.forward.__name__ == 'relu':
                return self.activation.relu(self.Z, **layer_kwargs)
            else:
                return self.activation.forward(self.Z, **layer_kwargs)
        return self.Z

    def backward(self, dA, X, learning_rate, apply_activation=True, update_weight=True):
        m = X.shape[0]
        if apply_activation:
            if hasattr(self.activation.backward, '__name__') and self.activation.backward.__name__ == 'sigmoid_derivative':
                dZ = dA * self.activation.sigmoid_derivative(self.Z, layer_id=self.layer_id, network_label=self.network_label, layer_rank=self.layer_rank)
            elif hasattr(self.activation.backward, '__name__') and self.activation.backward.__name__ == 'relu_derivative':
                dZ = dA * self.activation.relu_derivative(self.Z, layer_id=self.layer_id, network_label=self.network_label, layer_rank=self.layer_rank)
            elif hasattr(self.activation.backward, '__name__') and self.activation.backward.__name__ == 'approx_derivative':
                dZ = dA * self.activation.backward(self.Z, layer_id=self.layer_id, network_label=self.network_label, layer_rank=self.layer_rank)
            else:
                dZ = dA * self.activation.backward(self.Z, layer_id=self.layer_id, network_label=self.network_label, layer_rank=self.layer_rank)
        else:
            dZ = dA

        self.dW = X.T @ dZ# / m
        return dZ @ self.weights.T, self.dW

    def update_weights(self, gradient, learning_rate, cur_data_batch_size, is_last=False, is_last_iteration=False):
        with timer_context('UpdateWeights'):
            if self.momentum > 0.0:
                prev_velocity = self.velocity.copy()
                self.velocity = self.momentum * self.velocity - learning_rate * gradient
                self.weights = self.weights - self.momentum * prev_velocity + (1 + self.momentum) * self.velocity
            else:
                self.weights = self.weights - learning_rate * gradient


    def __repr__(self):
        return f"<Layer object at {hex(id(self))}, weights shape: {self.weights.shape}>"

    def update_activation_range(self, new_range):
        if hasattr(self.activation, 'update_approx_range'):
            try:
                return self.activation.update_approx_range(new_range)
            except ValueError:
                return False
        return False
