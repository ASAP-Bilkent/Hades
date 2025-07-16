from sklearn.preprocessing import MinMaxScaler
import copy
from src.core import utils

class DataManager:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
        self.scaler = MinMaxScaler()
    
    def setup_dims(self, X_train=None):
        if self.args.dims is not None:
            dims = [self.dataset.feature_count] + [int(x) for x in self.args.dims.split(",")] + [self.dataset.get_num_classes()]
        else:
            dims = [self.dataset.feature_count, self.dataset.get_num_classes()]

        if self.args.single_feat_split is not None:
            dims[0] = self.args.single_feat_split
        utils.vprint(f"Training started with bs {self.args.batch_size} and dims {dims}")

        if X_train is not None:
            dims_one = copy.deepcopy(dims)
            if self.args.n_clients > 1 or self.args.fusion is not None:
                dims_one[0] = X_train[0][0].shape[-1]
            else:
                dims_one[0] = X_train[0].shape[-1]

            dims_two = copy.deepcopy(dims)
            if self.args.n_clients > 1 or self.args.fusion is not None:
                dims_two[0] = X_train[0][1].shape[-1]
            else:
                dims_two[0] = X_train[1].shape[-1]

            dims = [dims_one, dims_two]

        return dims
        
    def configure_data(self):
        X_train, y_train = self.dataset.get_train_data()
        X_test, y_test = self.dataset.get_test_data()

        # Ensure all data is a multiple of args.batch_size
        utils.vprint(f"Before: X_train length: {len(X_train)}, y_train length: {len(y_train)}, X_test length: {len(X_test)}, y_test length: {len(y_test)}")
        X_train = X_train[:len(X_train) - len(X_train) % self.args.batch_size]
        y_train = y_train[:len(y_train) - len(y_train) % self.args.batch_size]
        X_test = X_test[:len(X_test) - len(X_test) % self.args.batch_size]
        y_test = y_test[:len(y_test) - len(y_test) % self.args.batch_size]
        utils.vprint(f"After: X_train length: {len(X_train)}, y_train length: {len(y_train)}, X_test length: {len(X_test)}, y_test length: {len(y_test)}")


        if self.args.data_amount == -1:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        else:
            data_per_client = self.args.data_amount * self.args.n_clients
            if self.args.data_amount > 1 and self.args.dataset != "synth": # or norm is 0
                self.scaler.fit(X_train)
                X_train = self.scaler.transform(X_train[:data_per_client])
                X_test = self.scaler.transform(X_test)
            else:
                if self.args.dataset == "synth":
                    X_train = X_train[:data_per_client]
                    if self.args.testing:
                        X_train = X_train / 10
                else:
                    X_train = X_train[:data_per_client] / 1000
                    X_test = X_test / 1000
            y_train = y_train[:data_per_client]

        if self.args.n_clients > 1:
            X_train, y_train, X_test, y_test = self._prepare_multi_client_data(X_train, y_train, X_test, y_test)
        else:
            X_train, y_train, X_test, y_test = self._prepare_single_client_data(X_train, y_train, X_test, y_test)

        return X_train, y_train, X_test, y_test
    
    def _prepare_multi_client_data(self, X_train, y_train, X_test, y_test):
        if self.args.fusion is not None:
            X_train, y_train = self._split_features(X_train, y_train, "train")
            X_test, y_test = self._split_features(X_test, y_test, "test")
            
            piece_size = len(X_train[0][0]) // self.args.n_clients
            # Trim data so all clients get equal samples (ensures equal batch counts)
            total_samples_to_use = piece_size * self.args.n_clients
            X_train[0][0] = X_train[0][0][:total_samples_to_use]
            X_train[0][1] = X_train[0][1][:total_samples_to_use]
            y_train = y_train[:total_samples_to_use]
            utils.vprint(f"Trimmed fusion data to {total_samples_to_use} samples for equal client distribution")

            network_data_one = self._get_piece(piece_size, X_train[0][0])
            network_data_two = self._get_piece(piece_size, X_train[0][1])

            X_train = list(zip(network_data_one, network_data_two))
            y_train = self._get_piece(piece_size, y_train)

            y_test = [y_test]
        else:
            piece_size = len(X_train) // self.args.n_clients
            # Trim data so all clients get equal samples (ensures equal batch counts)
            total_samples_to_use = piece_size * self.args.n_clients
            X_train = X_train[:total_samples_to_use]
            y_train = y_train[:total_samples_to_use]
            utils.vprint(f"Trimmed data to {total_samples_to_use} samples for equal client distribution")

            X_train = self._get_piece(piece_size, X_train)
            y_train = self._get_piece(piece_size, y_train)
            X_test = [X_test]
            y_test = [y_test]
            
        return X_train, y_train, X_test, y_test
    
    def _prepare_single_client_data(self, X_train, y_train, X_test, y_test):
        if self.args.fusion is not None:
            X_train, y_train = self._split_features(X_train, y_train, "train")
            X_test, y_test = self._split_features(X_test, y_test, "test")
            y_train = [y_train]
            y_test = [y_test]
        else:
            if self.args.single_feat_split is not None:
                X_train = X_train[:, :self.args.single_feat_split]
                X_test = X_test[:, :self.args.single_feat_split]

            X_train = [X_train]
            y_train = [y_train]
            X_test = [X_test]
            y_test = [y_test]
            
        return X_train, y_train, X_test, y_test
    
    def _split_features(self, X, y, name):
        utils.vprint(f"Before: X_{name}.shape", X.shape)
        X_one = X[:, self.args.fusion:]
        X_two = X[:, :self.args.fusion]
        utils.vprint(f"After: X_{name}_one.shape, X_{name}_two.shape:", X_one.shape, X_two.shape)
        X = [[X_one, X_two]]
        return X, y
    
    def _get_piece(self, piece_size, data):
        return [data[i * piece_size:(i + 1) * piece_size] for i in range(self.args.n_clients)] 