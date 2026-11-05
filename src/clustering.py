import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def scale_features(data: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def run_kmeans(X: np.ndarray, n_clusters: int = 4, random_state: int = 42) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, labels

def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> tuple[DBSCAN, np.ndarray]:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return model, labels

def reduce_dimensions(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)
