import os
import numpy as np
import pandas as pd
from src.eda import generate_eda_summary
from src.clustering import scale_features, run_kmeans, run_dbscan, reduce_dimensions
from src.evaluation import evaluate_clustering, save_segmentation_results

def run_pipeline():
    np.random.seed(42)
    sample_size = 300

    cluster_1 = np.random.normal(loc=[25, 30, 80], scale=[3, 5, 6], size=(75, 3))
    cluster_2 = np.random.normal(loc=[50, 85, 20], scale=[5, 8, 5], size=(75, 3))
    cluster_3 = np.random.normal(loc=[35, 90, 85], scale=[4, 7, 5], size=(75, 3))
    cluster_4 = np.random.normal(loc=[45, 35, 30], scale=[6, 6, 8], size=(75, 3))

    features = np.vstack([cluster_1, cluster_2, cluster_3, cluster_4])
    df = pd.DataFrame(features, columns=["Age", "AnnualIncome_k", "SpendingScore_1_100"])
    df["CustomerID"] = [f"CUST_{i+1:03d}" for i in range(sample_size)]
    cols = ["CustomerID", "Age", "AnnualIncome_k", "SpendingScore_1_100"]
    df = df[cols]

    data_path = os.path.join(os.path.dirname(__file__), "data", "customer_data.csv")
    df.to_csv(data_path, index=False)

    eda_summary = generate_eda_summary(df)

    raw_features = df[["Age", "AnnualIncome_k", "SpendingScore_1_100"]].values
    scaled_features, _ = scale_features(raw_features)

    kmeans_model, kmeans_labels = run_kmeans(scaled_features, n_clusters=4)
    dbscan_model, dbscan_labels = run_dbscan(scaled_features, eps=0.65, min_samples=5)

    kmeans_metrics = evaluate_clustering(scaled_features, kmeans_labels)
    dbscan_metrics = evaluate_clustering(scaled_features, dbscan_labels)

    df["KMeans_Cluster"] = kmeans_labels
    df["DBSCAN_Cluster"] = dbscan_labels

    cluster_profiles = {}
    for cluster_id in range(4):
        cluster_data = df[df["KMeans_Cluster"] == cluster_id]
        cluster_profiles[f"Cluster_{cluster_id}"] = {
            "size": int(len(cluster_data)),
            "mean_age": round(float(cluster_data["Age"].mean()), 1),
            "mean_income_k": round(float(cluster_data["AnnualIncome_k"].mean()), 1),
            "mean_spending_score": round(float(cluster_data["SpendingScore_1_100"].mean()), 1)
        }

    pca_2d = reduce_dimensions(scaled_features, n_components=2)
    df["PCA1"] = pca_2d[:, 0]
    df["PCA2"] = pca_2d[:, 1]

    report = {
        "dataset_size": sample_size,
        "eda_summary": eda_summary,
        "kmeans_evaluation": kmeans_metrics,
        "dbscan_evaluation": dbscan_metrics,
        "segment_interpretation": cluster_profiles
    }

    results_path = os.path.join(os.path.dirname(__file__), "results", "segmentation_report.json")
    save_segmentation_results(report, results_path)

    print("Customer Segmentation Analysis Complete:")
    print("-" * 65)
    print(f"K-Means (k=4) Silhouette Score: {kmeans_metrics['silhouette_score']}")
    print(f"K-Means Davies-Bouldin Index:  {kmeans_metrics['davies_bouldin_score']}")
    print(f"DBSCAN Clusters:               {dbscan_metrics['num_clusters']} (Noise: {dbscan_metrics['noise_points']})")
    print("-" * 65)
    print("Identified Customer Segments (K-Means):")
    for cid, profile in cluster_profiles.items():
        print(f"  {cid}: Size={profile['size']} | Age={profile['mean_age']} | Income=${profile['mean_income_k']}k | Spending={profile['mean_spending_score']}")
    print("-" * 65)

if __name__ == "__main__":
    run_pipeline()
