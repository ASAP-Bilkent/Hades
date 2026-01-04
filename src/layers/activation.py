import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as T
from src.core.utils import *
from src.core.hadestext import HadesText
import os
from pathlib import Path

class Activation:
    APPROX_RELU_POLYS = {
        3: {
            1: {0: 0.6936, 1: 0.5, 2: 0.1208},
            3: {0: 0.7146, 1: 1.5, 2: 0.8793},
            5: {0: 0.7865, 1: 2.5, 2: 1.88},
            7: {0: 0.9003, 1: 3.5, 2: 2.9013},
            10: {0: 1.1155, 1: 5.0, 2: 4.4003},
            20: {0: 1.87593797, 1: 10.0, 2: 9.37031484},
            30: {0: 2.81390695, 1: 15.0, 2: 14.05547226},
        },
        5: {
            1: {0: 0.4082, 1: 0.5, 2: 0.6609, 4: -0.2526},
            3: {0: 0.1760, 1: 1.5, 2: 2.4585, 4: -1.2268},
            5: {0: 0.2933, 1: 2.5, 2: 4.0975, 4: -2.0447},
            7: {0: 0.7521, 1: 3.5, 2: 4.3825, 4: -1.7281},
            10: {0: 0.5859667749, 1: 5.0, 2: 8.2027150283, 4: -4.1009475219},
            20: {0: 1.3127, 1: 10.0, 2: 15.7631, 4: -7.6296},
            30: {0: 1.75869185, 1: 15.0, 2: 24.59708876, 4: -12.28627041},
        },
        7: {
            1: {0: 0.3841, 1: 0.5, 2: 0.9204, 4: -0.8479, 6: 0.3662},
            10: {0: 0.4272674261, 1: 5.0, 2: 11.5350682168, 4: -14.0970077479, 6: 7.3297113656},
            20: {0: 0.8545348523, 1: 10.0, 2: 23.0701364336, 4: -28.1940154959, 6: 14.6594227313},
            30: {0: 1.2818022784, 1: 15.0, 2: 34.6052046504, 4: -42.2910232438, 6: 21.9891340969},
            50: {0: 2.1363371307, 1: 25.0, 2: 57.6753410840, 4: -70.4850387397, 6: 36.6485568282},
        },
        9: {
            5: { 0: 0.0343550549, 1: 0.5, 2: 0.2135653000, 4: -0.00845769038, 6: 0.000202706899, 8: -0.00000193209947, },
            10: { 0: 0.3364730838, 1: 5.0, 2: 14.8033378944, 4: -32.0706945346, 6: 38.4809881738, 8: -16.6865161036, },
            30: { 0: 1.0094192514, 1: 15.0, 2: 44.4100136831, 4: -96.2120836038, 6: 115.4429645214, 8: -50.0595483109, },
        },
        13: {
            5: { 0: 0.1181410937, 1: 2.5, 2: 10.6316389995, 4: -50.1999512563, 6: 152.5926168198, 8: -245.2136490924, 10: 194.9405211329, 12: -60.4094180327 },
            10: { 0: 0.2362821875, 1: 5.0, 2: 21.2632779990, 4: -100.3999025127, 6: 305.1852336397, 8: -490.4272981847, 10: 389.8810422657, 12: -120.8188360654 },
            30: { 0: 0.7088465624, 1: 15.0, 2: 63.7898339971, 4: -301.1997075381, 6: 915.5557009190, 8: -1471.2818945541, 10: 1169.6431267972, 12: -362.4565081962 },
            50: { 0: 1.1814109374, 1: 25.0, 2: 106.3163899952, 4: -501.9995125635, 6: 1525.9261681983, 8: -2452.1364909236, 10: 1949.4052113286, 12: -604.0941803270 },
        }
    }
    APPROX_SIGMOID_LS_POLYS = {
        3: {
            1:  {0: 0.5, 1: 0.449977813018, 3: -0.091311037095},
            3:  {0: 0.5, 1: 0.699714658337, 3: -0.264800954257},
            5:  {0: 0.5, 1: 0.991579111827, 3: -0.55901392223},
            7:  {0: 0.5, 1: 1.15095770387, 3: -0.751360991824},
            10: {0: 0.5, 1: 1.266714569922, 3: -0.902176393317},
            20: {0: 0.5, 1: 1.368413322791, 3: -1.040825918364},
        },
        5: {
            1:  {0: 0.5, 1: 0.469756669019, 3: -0.168675001664, 5: 0.034346188229},
            3:  {0: 0.5, 1: 0.739166110236, 3: -0.448846408574, 5: 0.165585732047},
            5:  {0: 0.5, 1: 1.14659107817, 3: -1.282162156262, 5: 0.650616610747},
            7:  {0: 0.5, 1: 1.428917772961, 3: -2.048075934784, 5: 1.166654693448},
            10: {0: 0.5, 1: 1.676888589851, 3: -2.815684267287, 5: 1.721583416801},
            20: {0: 0.5, 1: 1.939465253904, 3: -3.704847317919, 5: 2.396820585828},
        },
    }
    
    def __init__(self, args, enc=True):
        self.args = args
        self.enc = enc
        self.use_horner = getattr(args, 'horner', False)
        self.coeff_path = "approx_coeffs/"
        Path(self.coeff_path).mkdir(parents=True, exist_ok=True)
        self._sigmoid_coeff_cache = {}
        self._approx_relu_scaled_cache = {}
        self._approx_sigmoid_ls_scaled_cache = {}

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
        elif args.act_func == 'approx_relu':
            self.approx_degree = getattr(args, 'approx_degree', 3)
            self.approx_interval = abs(getattr(args, 'approx_b', 1.0)) or 1.0
            self.forward = self.approx_relu
            self.backward = self.approx_relu_derivative
        elif args.act_func == 'approx_sigmoid':
            self.a = getattr(args, 'approx_a', -5.0)
            self.b = getattr(args, 'approx_b', 5.0)
            self.degree = getattr(args, 'approx_degree', 3)
            
            if self.enc:
                self.forward = self.sigmoid_he
                self.backward = self.sigmoid_he_derivative

                self.plain_forward = self.approx
                self.plain_backward = self.approx_derivative
                base_range = abs(self.a)
                self.coefficients = self.load_approximation_coefficients(self.degree, base_range)
                self._sigmoid_coeff_cache[(self.degree, base_range)] = self.coefficients
            else:
                self.forward = self.approx
                self.backward = self.approx_derivative
                base_range = abs(self.a)
                self.coefficients = self.load_approximation_coefficients(self.degree, base_range)
                self._sigmoid_coeff_cache[(self.degree, base_range)] = self.coefficients
        elif args.act_func == 'approx_sigmoid_ls':
            if self.enc and not self.use_horner:
                raise ValueError("approx_sigmoid_ls is not defined for encrypted inputs without horner")
            self.approx_ls_degree = int(round(getattr(args, 'approx_degree', 3)))
            self.approx_ls_range = abs(float(getattr(args, 'approx_b', 3.0))) or 3.0
            if self.enc:
                self.forward = self.sigmoid_he
                self.backward = self.sigmoid_he_derivative
                self.plain_forward = self.approx_sigmoid_ls
                self.plain_backward = self.approx_sigmoid_ls_derivative
            else:
                self.forward = self.approx_sigmoid_ls
                self.backward = self.approx_sigmoid_ls_derivative
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
    
    def _get_sigmoid_coefficients(self, degree, range_val):
        key = (int(round(degree)), float(abs(range_val)))
        cached = self._sigmoid_coeff_cache.get(key)
        if cached is not None:
            return cached
        coeffs = self.load_approximation_coefficients(degree, range_val)
        self._sigmoid_coeff_cache[key] = coeffs
        return coeffs
    
    def _ensure_no_nan(self, value, func_name):
        if isinstance(value, np.ndarray):
            has_nan = np.isnan(value).any()
        elif np.isscalar(value):
            has_nan = np.isnan(value)
        else:
            return value
        if has_nan:
            raise ValueError(f"NaN detected in {func_name} output")
        return value
    
    def none(self, x, **_):
        return self._ensure_no_nan(x, 'none')
    
    def sigmoid(self, x, layer_id=None, network_label=None, layer_rank=None, skip_tracking=False, **_):
        result = 1 / (1 + np.exp(-x))
        return self._ensure_no_nan(result, 'sigmoid')
    
    def relu(self, x, layer_id=None, network_label=None, layer_rank=None, **_):
        result = np.maximum(0, x)
        return self._ensure_no_nan(result, 'relu')
    
    def relu_derivative(self, x, **_):
        result = np.where(x > 0, 1, 0)
        return self._ensure_no_nan(result, 'relu_derivative')
    
    def none_derivative(self, z, **_):
        return self._ensure_no_nan(1, 'none_derivative')

    def sigmoid_derivative(self, z, layer_id=None, network_label=None, layer_rank=None, **_):
        s = 1 / (1 + np.exp(-z))
        result = s * (1 - s)
        return self._ensure_no_nan(result, 'sigmoid_derivative')
    
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
    def approx(self, x, layer_id=None, network_label=None, layer_rank=None, **_):
        result = self._chebyshev_approx(x, self.approx_degree, self.approx_a, self.approx_b)
        return self._ensure_no_nan(result, 'approx')

    @count_calls
    def approx_derivative(self, x, layer_id=None, network_label=None, layer_rank=None, **_):
        result = self._chebyshev_approx_derivative(x, self.approx_degree, self.approx_a, self.approx_b)
        return self._ensure_no_nan(result, 'approx_derivative')

    def _max_abs_from_input(self, x):
        if isinstance(x, HadesText):
            data = getattr(x, 'data', None)
            if data is None:
                return 0.0
            x = data
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return 0.0
            return float(np.max(np.abs(x)))
        try:
            return float(abs(x))
        except (TypeError, ValueError):
            arr = np.asarray(x)
            return float(np.max(np.abs(arr))) if arr.size else 0.0

    def _determine_interval(self, x, default_interval):
        return default_interval

    def _eval_polynomial_horner(self, x, coeffs_dict):
        sorted_powers = sorted([p for p in coeffs_dict.keys() if p > 0], reverse=True)
        if not sorted_powers:
            return coeffs_dict.get(0, 0)
        
        result = coeffs_dict[sorted_powers[0]]
        
        for i in range(1, len(sorted_powers)):
            power = sorted_powers[i]
            prev_power = sorted_powers[i-1]
            power_diff = prev_power - power
            x_power_diff = np.power(x, power_diff)
            result = result * x_power_diff + coeffs_dict[power]
        
        min_power = sorted_powers[-1]
        if min_power > 0:
            x_min_power = np.power(x, min_power)
            result = result * x_min_power
        
        if 0 in coeffs_dict:
            result = result + coeffs_dict[0]
        
        return result

    def _eval_polynomial_derivative_horner(self, x, coeffs_dict):
        derivative_coeffs = {}
        for power, coeff in coeffs_dict.items():
            if power > 0:
                derivative_coeffs[power - 1] = coeff * power
        return self._eval_polynomial_horner(x, derivative_coeffs)

    def _eval_polynomial_horner_he(self, x, coeffs_dict):
        sorted_powers = sorted([p for p in coeffs_dict.keys() if p > 0], reverse=True)
        if not sorted_powers:
            const = coeffs_dict.get(0, 0)
            const_ht = HadesText(const, mask=x.validation_mask)
            if x.shape:
                pad_step = len(x.validation_mask) // x.shape[0]
                const_ht._pad_data(pad_step=pad_step, pad_batch=x.shape[0])
            const_ht.shape = x.shape
            return const_ht
        
        coeff0 = coeffs_dict[sorted_powers[0]]
        result = HadesText(coeff0, mask=x.validation_mask)
        if x.shape:
            pad_step = len(x.validation_mask) // x.shape[0]
            result._pad_data(pad_step=pad_step, pad_batch=x.shape[0])
        result.shape = x.shape
        
        for i in range(1, len(sorted_powers)):
            power = sorted_powers[i]
            prev_power = sorted_powers[i-1]
            power_diff = prev_power - power
            
            x_power = x
            for _ in range(power_diff - 1):
                x_power = x_power.mult(x, "CC")
            result = result.mult(x_power, "CC")
            
            coeff_val = coeffs_dict[power]
            coeff_ht = HadesText(coeff_val, mask=x.validation_mask)
            if x.shape:
                pad_step = len(x.validation_mask) // x.shape[0]
                coeff_ht._pad_data(pad_step=pad_step, pad_batch=x.shape[0])
            coeff_ht.shape = x.shape
            result = result.add(coeff_ht, "CP")
        
        min_power = sorted_powers[-1]
        if min_power > 0:
            x_power = x
            for _ in range(min_power - 1):
                x_power = x_power.mult(x, "CC")
            result = result.mult(x_power, "CC")
        
        if 0 in coeffs_dict:
            const_ht = HadesText(coeffs_dict[0], mask=x.validation_mask)
            if x.shape:
                pad_step = len(x.validation_mask) // x.shape[0]
                const_ht._pad_data(pad_step=pad_step, pad_batch=x.shape[0])
            const_ht.shape = x.shape
            result = result.add(const_ht, "CP")
        
        return result

    def _eval_polynomial_derivative_horner_he(self, x, coeffs_dict):
        derivative_coeffs = {}
        for power, coeff in coeffs_dict.items():
            if power > 0:
                derivative_coeffs[power - 1] = coeff * power
        return self._eval_polynomial_horner_he(x, derivative_coeffs)

    def _get_approx_relu_coeffs(self, interval_override=None):
        degree = int(round(self.approx_degree))
        interval = float(interval_override if interval_override is not None else self.approx_interval)
        poly_by_interval = self.APPROX_RELU_POLYS.get(degree)
        if not poly_by_interval:
            raise ValueError(f"Approx ReLU degree {degree} unsupported")
        interval_key = None
        for candidate in poly_by_interval:
            if np.isclose(candidate, interval):
                interval_key = candidate
                break
        if interval_key is None:
            supported = sorted(poly_by_interval.keys())
            raise ValueError(f"Approx ReLU interval {interval} unsupported for degree {degree}. Supported: {supported}")
        return interval_key, poly_by_interval[interval_key]

    @count_calls
    def approx_relu(self, x, layer_id=None, network_label=None, layer_rank=None, **_):
        interval = self._determine_interval(x, self.approx_interval)
        interval_key, coeffs = self._get_approx_relu_coeffs(interval_override=interval)
        
        cache_key = (int(round(self.approx_degree)), float(interval_key))
        scaled_coeffs = self._approx_relu_scaled_cache.get(cache_key)
        if scaled_coeffs is None:
            inv_powers = {}
            for power in coeffs:
                if power == 0:
                    continue
                inv_powers.setdefault(power, interval_key**power)
            scaled_coeffs = {}
            for power, coeff in coeffs.items():
                if power == 0:
                    scaled_coeffs[power] = coeff
                else:
                    scaled_coeffs[power] = coeff / inv_powers[power]
            self._approx_relu_scaled_cache[cache_key] = scaled_coeffs
        
        if isinstance(x, HadesText):
            if not self.use_horner:
                raise ValueError("approx_relu for encrypted inputs requires --horner")
            result = self._eval_polynomial_horner_he(x, scaled_coeffs)
            result.data = self._eval_polynomial_horner(x.data, scaled_coeffs) if x.data is not None else None
            result.shape = x.shape
            return self._ensure_no_nan(result, 'approx_relu')

        if self.use_horner:
            result = self._eval_polynomial_horner(x, scaled_coeffs)
        else:
            result = 0
            for power, coeff in sorted(scaled_coeffs.items()):
                x_power = np.power(x, power)
                term = coeff * x_power
                result += term
        
        return self._ensure_no_nan(result, 'approx_relu')

    @count_calls
    def approx_relu_derivative(self, x, layer_id=None, network_label=None, layer_rank=None, **_):
        interval = self._determine_interval(x, self.approx_interval)
        interval_key, coeffs = self._get_approx_relu_coeffs(interval_override=interval)
        cache_key = (int(round(self.approx_degree)), float(interval_key))
        scaled_coeffs = self._approx_relu_scaled_cache.get(cache_key)
        if scaled_coeffs is None:
            inv_powers = {}
            for power in coeffs:
                if power == 0:
                    continue
                inv_powers.setdefault(power, interval_key**power)
            scaled_coeffs = {}
            for power, coeff in coeffs.items():
                if power == 0:
                    scaled_coeffs[power] = coeff
                else:
                    scaled_coeffs[power] = coeff / inv_powers[power]
            self._approx_relu_scaled_cache[cache_key] = scaled_coeffs
        
        if isinstance(x, HadesText):
            if not self.use_horner:
                raise ValueError("approx_relu_derivative for encrypted inputs requires --horner")
            derivative = self._eval_polynomial_derivative_horner_he(x, scaled_coeffs)
            derivative.data = self._eval_polynomial_derivative_horner(x.data, scaled_coeffs) if x.data is not None else None
            derivative.shape = x.shape
            return self._ensure_no_nan(derivative, 'approx_relu_derivative')

        if self.use_horner:
            derivative = self._eval_polynomial_derivative_horner(x, scaled_coeffs)
        else:
            derivative = 0
            for power, coeff in sorted(scaled_coeffs.items()):
                if power == 0:
                    continue
                x_power_minus_1 = np.power(x, power - 1)
                term = coeff * power * x_power_minus_1
                derivative += term
        
        return self._ensure_no_nan(derivative, 'approx_relu_derivative')
    
    def _get_approx_sigmoid_ls_coeffs(self, interval_override=None):
        degree = int(round(self.approx_ls_degree))
        interval = float(interval_override if interval_override is not None else self.approx_ls_range)
        poly_by_interval = self.APPROX_SIGMOID_LS_POLYS.get(degree)
        if not poly_by_interval:
            raise ValueError(f"Approx sigmoid LS degree {degree} unsupported")
        interval_key = None
        for candidate in poly_by_interval:
            if np.isclose(candidate, interval):
                interval_key = candidate
                break
        if interval_key is None:
            supported = sorted(poly_by_interval.keys())
            raise ValueError(f"Approx sigmoid LS interval {interval} unsupported for degree {degree}. Supported: {supported}")
        return interval_key, poly_by_interval[interval_key]

    @count_calls
    def approx_sigmoid_ls(self, x, layer_id=None, network_label=None, layer_rank=None, **_):
        interval = self._determine_interval(x, self.approx_ls_range)
        interval_key, coeffs = self._get_approx_sigmoid_ls_coeffs(interval_override=interval)
        cache_key = (int(round(self.approx_ls_degree)), float(interval_key))
        scaled_coeffs = self._approx_sigmoid_ls_scaled_cache.get(cache_key)
        if scaled_coeffs is None:
            inv_powers = {}
            for power in coeffs:
                if power == 0:
                    continue
                inv_powers.setdefault(power, interval_key**power)
            scaled_coeffs = {}
            for power, coeff in coeffs.items():
                if power == 0:
                    scaled_coeffs[power] = coeff
                else:
                    scaled_coeffs[power] = coeff / inv_powers[power]
            self._approx_sigmoid_ls_scaled_cache[cache_key] = scaled_coeffs
        if self.use_horner:
            result = self._eval_polynomial_horner(x, scaled_coeffs)
        else:
            result = 0
            for power, coeff in sorted(scaled_coeffs.items()):
                result += coeff * np.power(x, power)
        return self._ensure_no_nan(result, 'approx_sigmoid_ls')

    @count_calls
    def approx_sigmoid_ls_derivative(self, x, layer_id=None, network_label=None, layer_rank=None, **_):
        interval = self._determine_interval(x, self.approx_ls_range)
        interval_key, coeffs = self._get_approx_sigmoid_ls_coeffs(interval_override=interval)
        cache_key = (int(round(self.approx_ls_degree)), float(interval_key))
        scaled_coeffs = self._approx_sigmoid_ls_scaled_cache.get(cache_key)
        if scaled_coeffs is None:
            inv_powers = {}
            for power in coeffs:
                if power == 0:
                    continue
                inv_powers.setdefault(power, interval_key**power)
            scaled_coeffs = {}
            for power, coeff in coeffs.items():
                if power == 0:
                    scaled_coeffs[power] = coeff
                else:
                    scaled_coeffs[power] = coeff / inv_powers[power]
            self._approx_sigmoid_ls_scaled_cache[cache_key] = scaled_coeffs
        if self.use_horner:
            derivative = self._eval_polynomial_derivative_horner(x, scaled_coeffs)
        else:
            derivative = 0
            for power, coeff in scaled_coeffs.items():
                if power == 0:
                    continue
                derivative += coeff * power * np.power(x, power - 1)
        return self._ensure_no_nan(derivative, 'approx_sigmoid_ls_derivative')

    def update_approx_range(self, new_range):
        act_func = getattr(self.args, 'act_func', None)
        if act_func not in ('approx_sigmoid', 'approx_sigmoid_ls', 'approx_relu'):
            return False
        new_range = float(abs(new_range))
        if act_func == 'approx_sigmoid':
            self.a = -new_range
            self.b = new_range
            self.args.approx_a = self.a
            self.args.approx_b = self.b
            self.coefficients = self.load_approximation_coefficients(self.degree, new_range)
            return True
        if act_func == 'approx_sigmoid_ls':
            self.approx_ls_range = new_range or self.approx_ls_range
            self._get_approx_sigmoid_ls_coeffs()
            return True
        if act_func == 'approx_relu':
            self.approx_interval = new_range or self.approx_interval
            self.args.approx_b = self.approx_interval
            # Validate availability
            self._get_approx_relu_coeffs()
            return True
        return False

    @count_calls
    def sigmoid_he(self, x, layer_id=None, network_label=None, layer_rank=None, **_):
        if isinstance(x, HadesText):
            if self.use_horner:
                interval = self._determine_interval(x, self.approx_ls_range)
                interval_key, coeffs = self._get_approx_sigmoid_ls_coeffs(interval_override=interval)
                cache_key = (int(round(self.approx_ls_degree)), float(interval_key))
                scaled_coeffs = self._approx_sigmoid_ls_scaled_cache.get(cache_key)
                if scaled_coeffs is None:
                    inv_powers = {}
                    for power in coeffs:
                        if power == 0:
                            continue
                        inv_powers.setdefault(power, interval_key**power)
                    scaled_coeffs = {}
                    for power, coeff in coeffs.items():
                        if power == 0:
                            scaled_coeffs[power] = coeff
                        else:
                            scaled_coeffs[power] = coeff / inv_powers[power]
                    self._approx_sigmoid_ls_scaled_cache[cache_key] = scaled_coeffs
                result = self._eval_polynomial_horner_he(x, scaled_coeffs)
                result.data = self._eval_polynomial_horner(x.data, scaled_coeffs) if x.data is not None else None
                result.shape = x.shape
                return result
            else:
                return x.logistic(self.a, self.b, self.degree)
        else:
            raise ValueError("Sigmoid_he is not defined for non-HadesText inputs")
            
    @count_calls
    def sigmoid_he_derivative(self, x, **_):
        if isinstance(x, HadesText):
            if self.use_horner:
                interval = self._determine_interval(x, self.approx_ls_range)
                interval_key, coeffs = self._get_approx_sigmoid_ls_coeffs(interval_override=interval)
                cache_key = (int(round(self.approx_ls_degree)), float(interval_key))
                scaled_coeffs = self._approx_sigmoid_ls_scaled_cache.get(cache_key)
                if scaled_coeffs is None:
                    inv_powers = {}
                    for power in coeffs:
                        if power == 0:
                            continue
                        inv_powers.setdefault(power, interval_key**power)
                    scaled_coeffs = {}
                    for power, coeff in coeffs.items():
                        if power == 0:
                            scaled_coeffs[power] = coeff
                        else:
                            scaled_coeffs[power] = coeff / inv_powers[power]
                    self._approx_sigmoid_ls_scaled_cache[cache_key] = scaled_coeffs
                derivative = self._eval_polynomial_derivative_horner_he(x, scaled_coeffs)
                derivative.data = self._eval_polynomial_derivative_horner(x.data, scaled_coeffs) if x.data is not None else None
                derivative.shape = x.shape
                return derivative
            else:
                return x.logistic_derivative(self.a, self.b, self.degree)
        else:
            raise ValueError("Sigmoid_he_derivative is not defined for non-HadesText inputs")
