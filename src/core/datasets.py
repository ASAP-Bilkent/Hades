from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo
import numpy as np

class BaseDataset:
    def __init__(self):
        self.pca = None
        self.pca_to_feat = False
        self.feature_count = 0
        self.num_classes = 0
    
    def _apply_pca(self, apply_pca, pca_to_feat):
        self.pca = PCA() if apply_pca else None
        self.pca_to_feat = pca_to_feat
        
        if self.pca is not None:
            print("Doing PCA")
            
            # Check if we have enough samples for meaningful PCA
            n_samples = self.X_train.shape[0]
            if n_samples < self.feature_count:
                print(f"Warning: Only {n_samples} samples for {self.feature_count} features. PCA will reduce dimensions to {n_samples}.")
            
            self.pca.fit(self.X_train)
            if self.pca_to_feat:
                self._select_features_by_pca()
            else:
                self.X_train = self.pca.transform(self.X_train)
                self.X_test = self.pca.transform(self.X_test)
                self.feature_count = self.X_train.shape[1]
    
    def _select_features_by_pca(self):
        n_components = min(self.pca.n_components_, self.feature_count)
        feature_importance = np.sum(self.pca.components_[:n_components]**2, axis=0)
        feature_indices = np.argsort(feature_importance)[::-1]
        self.X_train = self.X_train[:, feature_indices]
        self.X_test = self.X_test[:, feature_indices]
        print(f"Features reordered by PCA importance. Top 5 feature indices: {feature_indices[:5]}")
    
    def get_train_data(self):
        one_hot_y = np.zeros((self.y_train.size, self.num_classes))
        one_hot_y[np.arange(self.y_train.size), self.y_train.astype(int)] = 1
        return self.X_train, one_hot_y
    
    def get_test_data(self):
        one_hot_y = np.zeros((self.y_test.size, self.num_classes))
        one_hot_y[np.arange(self.y_test.size), self.y_test.astype(int)] = 1
        return self.X_test, one_hot_y
    
    def get_num_classes(self):
        return self.num_classes

class BreastCancerOriginalDataset(BaseDataset):
    def __init__(self, apply_pca=False, pca_to_feat=False, test_size=0.3, random_state=42):
        super().__init__()
        breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
        
        X = np.array(breast_cancer_wisconsin_original.data.features)
        y = np.array(breast_cancer_wisconsin_original.data.targets)
        y = np.where(y == 2, 0, 1).flatten()

        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.feature_count = X.shape[1]
        self.num_classes = 2

        self._apply_pca(apply_pca, pca_to_feat)

class BreastCancerDiagnosticDataset(BaseDataset):
    def __init__(self, apply_pca=False, pca_to_feat=False, test_size=0.3, random_state=42):
        super().__init__()
        data = load_breast_cancer()
        X = data.data
        y = data.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.feature_count = X.shape[1]
        self.num_classes = 2

        self._apply_pca(apply_pca, pca_to_feat)

class MNISTDataset(BaseDataset):
    def __init__(self, apply_pca=False, pca_to_feat=False, test_size=0.3, random_state=42):
        super().__init__()
        mnist = fetch_openml('mnist_784', as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size * 2, random_state=random_state
        )
        self.feature_count = X.shape[1]
        self.num_classes = 16

        self._apply_pca(apply_pca, pca_to_feat)

class SyntheticDataset(BaseDataset):
    def __init__(self, n_features=10, n_samples=100, apply_pca=False, pca_to_feat=False, test_size=0.3, random_state=42):
        super().__init__()
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, size=n_samples)
        
        self.X_train = X
        self.X_test = X
        self.y_train = y
        self.y_test = y
        self.feature_count = n_features
        self.num_classes = 2
        
        self._apply_pca(apply_pca, pca_to_feat)

