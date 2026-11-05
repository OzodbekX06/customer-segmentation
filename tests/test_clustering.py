import unittest
import numpy as np
from src.clustering import run_kmeans, run_dbscan, scale_features, reduce_dimensions
from src.evaluation import evaluate_clustering

class TestClusteringPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        c1 = np.random.randn(50, 3) + np.array([5, 5, 5])
        c2 = np.random.randn(50, 3) + np.array([-5, -5, -5])
        self.X = np.vstack([c1, c2])

    def test_scaling_and_kmeans(self):
        scaled_X, _ = scale_features(self.X)
        model, labels = run_kmeans(scaled_X, n_clusters=2)
        self.assertEqual(len(labels), 100)
        self.assertEqual(len(set(labels)), 2)

    def test_clustering_evaluation(self):
        scaled_X, _ = scale_features(self.X)
        _, labels = run_kmeans(scaled_X, n_clusters=2)
        metrics = evaluate_clustering(scaled_X, labels)
        self.assertGreater(metrics["silhouette_score"], 0.7)

    def test_pca_reduction(self):
        pca_X = reduce_dimensions(self.X, n_components=2)
        self.assertEqual(pca_X.shape, (100, 2))

if __name__ == "__main__":
    unittest.main()
