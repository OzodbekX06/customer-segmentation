import json
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    unique_labels = set(labels)
    num_clusters = len(unique_labels - {-1})
    num_noise = int(np.sum(labels == -1))

    if num_clusters > 1 and len(unique_labels) < len(X):
        valid_indices = labels != -1
        if np.sum(valid_indices) > num_clusters:
            sil_score = round(float(silhouette_score(X[valid_indices], labels[valid_indices])), 4)
            db_score = round(float(davies_bouldin_score(X[valid_indices], labels[valid_indices])), 4)
        else:
            sil_score, db_score = None, None
    else:
        sil_score, db_score = None, None

    return {
        "num_clusters": num_clusters,
        "noise_points": num_noise,
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score
    }

def save_segmentation_results(report: dict, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
