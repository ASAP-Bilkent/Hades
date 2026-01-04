from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import numpy as np
import pickle
import os
from urllib.request import urlretrieve
from tqdm import tqdm

class BaseDataset:
    def __init__(self, apply_pca=False, pca_to_feat=False, test_size=0.3, random_state=42, 
                 data_amount=1, pca_cache_dir=None, pca_cache_name=None):
        self.pca = None
        self.scaler = None
        self.pca_to_feat = False
        self.feature_count = 0
        self.num_classes = 0
        self.test_size = test_size
        self.random_state = random_state
        
        X_train, X_test, y_train, y_test, feature_count, num_classes = self._load_data()
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_count = feature_count
        self.num_classes = num_classes
        
        pca_cache_dir = getattr(self, 'pca_cache_dir', pca_cache_dir)
        pca_cache_name = getattr(self, 'pca_cache_name', pca_cache_name)
        self._apply_pca(apply_pca, pca_to_feat, data_amount, pca_cache_dir, pca_cache_name)
    
    def _load_data(self):
        raise NotImplementedError("Subclasses must implement _load_data()")
    
    def _apply_pca(self, apply_pca, pca_to_feat, data_amount=1, pca_cache_dir=None, pca_cache_name=None):
        self.pca = PCA() if apply_pca else None
        self.scaler = None
        self.pca_to_feat = pca_to_feat
        
        if self.pca is not None:
            pca_loaded = False
            
            if data_amount == -1 and pca_cache_dir is not None and pca_cache_name is not None:
                base_name, ext = os.path.splitext(pca_cache_name)
                pca_cache_file = os.path.join(pca_cache_dir, f"{base_name}{ext}")
                if os.path.exists(pca_cache_file):
                    print("Loading saved PCA from cache...")
                    with open(pca_cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    if isinstance(cached_data, dict):
                        cached_pca = cached_data.get('pca')
                        cached_scaler = cached_data.get('scaler')
                    else:
                        cached_pca = cached_data
                        cached_scaler = None
                    
                    if cached_pca.n_features_in_ != self.feature_count:
                        print(f"Warning: Cached PCA expects {cached_pca.n_features_in_} features but data has {self.feature_count}. Recomputing PCA...")
                        pca_loaded = False
                    else:
                        self.pca = cached_pca
                        self.scaler = cached_scaler
                        pca_loaded = True
            
            if not pca_loaded:
                print("Doing PCA")
                
                n_samples = self.X_train.shape[0]
                if n_samples < self.feature_count:
                    print(f"Warning: Only {n_samples} samples for {self.feature_count} features. PCA will reduce dimensions to {n_samples}.")
                
                X_train_for_pca = self.X_train
                
                self.pca.fit(X_train_for_pca)
                
                if data_amount == -1 and pca_cache_dir is not None and pca_cache_name is not None:
                    base_name, ext = os.path.splitext(pca_cache_name)
                    pca_cache_file = os.path.join(pca_cache_dir, f"{base_name}{ext}")
                    print(f"Saving PCA to {pca_cache_file}...")
                    cache_data = {'pca': self.pca}
                    with open(pca_cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)
            
            if self.pca_to_feat:
                self._select_features_by_pca()
            else:
                X_train_for_transform = self.X_train
                X_test_for_transform = self.X_test
                
                self.X_train = self.pca.transform(X_train_for_transform)
                self.X_test = self.pca.transform(X_test_for_transform)
                
                self.feature_count = self.X_train.shape[1]
                print(f"After PCA - X_train: min={self.X_train.min():.6f}, max={self.X_train.max():.6f}, mean={self.X_train.mean():.6f}")
                print(f"After PCA - X_test: min={self.X_test.min():.6f}, max={self.X_test.max():.6f}, mean={self.X_test.mean():.6f}")
                
    
    def _select_features_by_pca(self):
        n_components = min(self.pca.n_components_, self.feature_count)
        feature_importance = np.sum(self.pca.components_[:n_components]**2, axis=0)
        feature_indices = np.argsort(feature_importance)[::-1]
        self.X_train = self.X_train[:, feature_indices]
        self.X_test = self.X_test[:, feature_indices]
        print(f"Features reordered by PCA importance. Top 5 feature indices: {feature_indices[:5]}")
        print(f"After PCA feature selection - X_train: min={self.X_train.min():.6f}, max={self.X_train.max():.6f}, mean={self.X_train.mean():.6f}")
        print(f"After PCA feature selection - X_test: min={self.X_test.min():.6f}, max={self.X_test.max():.6f}, mean={self.X_test.mean():.6f}")
    
    def get_train_data(self):
        y_normalized = self.y_train.astype(int) - self.y_train.min()
        one_hot_y = np.zeros((self.y_train.size, self.num_classes))
        one_hot_y[np.arange(self.y_train.size), y_normalized] = 1
        return self.X_train, one_hot_y
    
    def get_test_data(self):
        y_normalized = self.y_test.astype(int) - self.y_test.min()
        one_hot_y = np.zeros((self.y_test.size, self.num_classes))
        one_hot_y[np.arange(self.y_test.size), y_normalized] = 1
        return self.X_test, one_hot_y
    
    def get_num_classes(self):
        return self.num_classes


class BreastCancerDiagnosticDataset(BaseDataset):
    def _load_data(self):
        data = load_breast_cancer()
        X = data.data
        y = data.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test, X.shape[1], 2

class MNISTDataset(BaseDataset):
    def _load_data(self):
        mnist = fetch_openml('mnist_784', as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
        X = X.astype(np.float32) / 255.0
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test, X.shape[1], 10


class SyntheticDataset(BaseDataset):
    def __init__(self, n_features=10, n_samples=100, apply_pca=False, pca_to_feat=False, 
                 test_size=0.3, random_state=42, **kwargs):
        self.n_features = n_features
        self.n_samples = n_samples
        super().__init__(apply_pca=apply_pca, pca_to_feat=pca_to_feat, test_size=test_size, 
                        random_state=random_state, **kwargs)
    
    def _load_data(self):
        X = np.random.rand(self.n_samples, self.n_features)
        y = np.random.randint(0, 2, size=self.n_samples)
        return X, X, y, y, self.n_features, 2

class SVHNDataset(BaseDataset):
    def _load_data(self):
        from scipy.io import loadmat
        
        data_dir = os.path.join(os.path.expanduser('~'), '.svhn')
        os.makedirs(data_dir, exist_ok=True)
        
        train_file = os.path.join(data_dir, 'train_32x32.mat')
        test_file = os.path.join(data_dir, 'test_32x32.mat')
        
        train_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
        test_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
        
        def load_mat_file(filepath, url):
            if not os.path.exists(filepath):
                print(f"Downloading SVHN dataset from {url}...")
                urlretrieve(url, filepath)
            
            mat = loadmat(filepath)
            X = mat['X']
            y = mat['y'].flatten()
            y[y == 10] = 0
            
            X = np.transpose(X, (3, 0, 1, 2))
            X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
            
            return X, y
        
        X_train, y_train = load_mat_file(train_file, train_url)
        X_test, y_test = load_mat_file(test_file, test_url)
        
        self.pca_cache_dir = data_dir
        self.pca_cache_name = 'svhn_pca.pkl'
        
        return X_train, X_test, y_train.astype(int), y_test.astype(int), X_train.shape[1], 10

