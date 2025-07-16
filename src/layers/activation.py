import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as T
from src.core.utils import *
from src.core.hadestext import HadesText
import os
from pathlib import Path

class Activation:
    def __init__(self, args, enc=True):
        self.args = args
        self.enc = enc
        self.coeff_path = "approx_coeffs/"
        Path(self.coeff_path).mkdir(parents=True, exist_ok=True)

        if args.act_func is None or args.act_func == 'none':
            self.plain_forward = self.none
            self.plain_backward = self.none_derivative

            if self.enc:
                self.forward = self.none
                self.backward = self.none_derivative
            else:
                self.forward = self.none
                self.backward = self.none_derivative
        elif args.act_func == 'sigmoid':
            if self.enc:
                raise ValueError("Sigmoid is not defined for encrypted inputs")
            else:
                self.forward = self.sigmoid
                self.backward = self.sigmoid_derivative
        elif args.act_func == 'approx_sigmoid':
            self.a = getattr(args, 'approx_a', -5.0)
            self.b = getattr(args, 'approx_b', 5.0)
            self.degree = getattr(args, 'approx_degree', 3)
            
            if self.enc:
                self.forward = self.sigmoid_he
                self.backward = self.sigmoid_he_derivative

                self.plain_forward = self.approx
                self.plain_backward = self.approx_derivative
                self.coefficients = self.load_approximation_coefficients(self.degree, abs(self.a))
            else:
                self.forward = self.approx
                self.backward = self.approx_derivative
                self.coefficients = self.load_approximation_coefficients(self.degree, abs(self.a))
        elif args.act_func == 'relu':
            if self.enc:
                raise ValueError("ReLU is not defined for encrypted inputs")
            else:
                self.forward = self.relu
                self.backward = self.relu_derivative

    def eval_chebyshev_coefficients(self, func, a, b, degree):
        if degree == 0:
            raise ValueError("The degree of approximation cannot be zero")
        
        if abs(a) != abs(b):
            raise ValueError("Absolute values of a and b must be equal")
            
        coeff_total = degree + 1
        b_minus_a = 0.5 * (b - a)
        b_plus_a = 0.5 * (b + a)
        pi_by_deg = np.pi / coeff_total
        
        j_vals = np.arange(coeff_total)
        x_vals = np.cos(pi_by_deg * (j_vals + 0.5)) * b_minus_a + b_plus_a
        # Scale sigmoid to [-1,1] range for approximation
        function_points = 2 * func(x_vals) - 1
        
        mult_factor = 2.0 / coeff_total
        i_vals = np.arange(coeff_total)[:, np.newaxis]
        j_vals = np.arange(coeff_total)[np.newaxis, :]
        cos_matrix = np.cos(pi_by_deg * i_vals * (j_vals + 0.5))
        coefficients = mult_factor * np.sum(function_points * cos_matrix, axis=1)
        
        return coefficients

    def load_approximation_coefficients(self, degree, range_val):
        csv_path = os.path.join(self.coeff_path, f'sigmoid_chebyshev_k{degree}_r{int(abs(range_val))}.csv')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            coeffs = df[df['Method'] == 'Chebyshev'].iloc[0, 1:].dropna().values
        else:
            func = lambda x: 1 / (1 + np.exp(-x))
            coeffs = self.eval_chebyshev_coefficients(func, -range_val, range_val, degree)
            
            df = pd.DataFrame({
                'Method': ['Chebyshev'],
                **{f'c{i}': [c] for i, c in enumerate(coeffs)}
            })
            df.to_csv(csv_path, index=False)
            
        return coeffs
    
    def none(self, x):
        return x
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def none_derivative(self, z):
        return 1

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def _chebyshev_recurrence(self, x, n, base_0, base_1):
        if n == 0:
            return base_0
        if n == 1:
            return base_1
        val_n_2 = base_0
        val_n_1 = base_1
        for k in range(2, n+1):
            val_n = 2*x*val_n_1 - val_n_2
            val_n_2, val_n_1 = val_n_1, val_n
        return val_n

    def Tfunc(self, x, n):
        return self._chebyshev_recurrence(x, n, 1, x)

    def Ufunc(self, x, n):
        return self._chebyshev_recurrence(x, n, 1, 2*x)

    @count_calls
    def approx(self, x):
        s = 0
        for n, a in enumerate(self.coefficients):
            s += a * self.Tfunc(x, n)
        return 0.5 * (s + 1)  # Convert back from [-1,1] to [0,1]

    @count_calls
    def approx_derivative(self, x):
        s = 0
        for n, a in enumerate(self.coefficients):
            if n:
                s += a * n * self.Ufunc(x, n-1)
        return s

    @count_calls
    def sigmoid_he(self, x):
        if isinstance(x, HadesText):
            return x.logistic(self.a, self.b, self.degree)
        else:
            raise ValueError("Sigmoid_he is not defined for non-HadesText inputs")
            
    @count_calls
    def sigmoid_he_derivative(self, x):
        if isinstance(x, HadesText):
            return x.logistic_derivative(self.a, self.b, self.degree)
        else:
            raise ValueError("Sigmoid_he_derivative is not defined for non-HadesText inputs")
