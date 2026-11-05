# Customer Segmentation

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

## Overview
Customer Segmentation is an unsupervised machine learning system designed to partition consumer populations into distinct, actionable behavioral cohorts based on demographic and purchasing attributes. By revealing latent patterns in customer behavior, businesses can craft targeted marketing campaigns, improve customer retention, and maximize revenue efficiency.

## Objectives
- Conduct Exploratory Data Analysis (EDA) on continuous customer behavioral metrics.
- Apply feature standardization to eliminate scale variance across dimensions.
- Implement and benchmark partition-based (K-Means) versus density-based (DBSCAN) clustering algorithms.
- Reduce multi-dimensional feature space using Principal Component Analysis (PCA) for cluster separation.
- Quantitatively validate cluster quality using Silhouette Analysis and Davies-Bouldin Index.
- Provide business-interpretable profiles for each identified persona.

## Features
- **Exploratory Data Analysis:** Computes statistical distributions, missing values, skewness, and variance across customer records.
- **K-Means Clustering:** Optimal cluster centroid convergence initialized via k-means++.
- **DBSCAN Clustering:** Density-based spatial clustering identifying core samples and anomalous outlier records.
- **Dimensionality Reduction:** 2-component PCA projection capturing maximal dataset variance.
- **Automated Cluster Validation:** Calculates Silhouette Coefficients and Davies-Bouldin scores.
- **Cohort Profiling:** Automated aggregation generating mean age, income, and spending scores per segment.

## Technologies Used
- **Python 3.10+**
- **NumPy:** Vectorized Euclidean distance computations and array slicing.
- **Pandas:** Structured tabular manipulation and cohort grouping.
- **Scikit-Learn:** `KMeans`, `DBSCAN`, `PCA`, `StandardScaler`, and validation metrics.
- **Matplotlib & Seaborn:** Scatter projections and cluster visualization.

## Project Structure
```text
customer-segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   └── customer_data.csv
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── clustering.py
│   └── evaluation.py
├── tests/
│   ├── __init__.py
│   └── test_clustering.py
├── results/
│   └── segmentation_report.json
└── main.py
```

## Dataset
The dataset encompasses 300 customer records detailing:
- `CustomerID`: Unique identifier.
- `Age`: Customer age in years.
- `AnnualIncome_k`: Estimated annual income in thousands of USD ($k$).
- `SpendingScore_1_100`: Score assigned based on customer behavior and spending nature (scale 1–100).

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/OzodbekX06/customer-segmentation.git
cd customer-segmentation

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage
Run the unit test suite:

```bash
python -m unittest discover tests
```

Execute the full segmentation and evaluation pipeline:

```bash
python main.py
```

Results are printed to the console and serialized to `results/segmentation_report.json`.

## Methodology
1. **Feature Standardization:** Features have differing units (years, thousands of dollars, 1-100 score). `StandardScaler` standardizes each feature to $\mu = 0$ and $\sigma = 1$.
2. **K-Means Optimization:** Minimizes within-cluster sum-of-squares (inertia):
   $$\operatorname{argmin}_{\mathbf{S}} \sum_{i=1}^{k} \sum_{\mathbf{x} \in S_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$
3. **Density-Based Analysis:** DBSCAN evaluates density reachability ($\varepsilon = 0.65, \text{MinPts} = 5$).
4. **Validation:** Silhouette Coefficient $s = \frac{b - a}{\max(a, b)}$ evaluates cluster cohesion and separation.

## Results
The algorithms generated the following verified validation scores:

| Clustering Algorithm | Clusters Discovered | Noise Points | Silhouette Score | Davies-Bouldin Index |
| :--- | :--- | :--- | :--- | :--- |
| **K-Means ($k=4$)** | **4** | **0** | **0.6888** | **0.4600** |
| **DBSCAN ($\varepsilon=0.65$)** | 3 | 0 | 0.6207 | 0.5410 |

### Customer Cohort Interpretation (K-Means)
- **Cluster 0 — Conservative High-Earners:** Mean Age: 50.4 years | Annual Income: $84.0k | Spending Score: 20.7/100 (High income, prudent spending).
- **Cluster 1 — Affluent Spenders:** Mean Age: 35.4 years | Annual Income: $88.7k | Spending Score: 84.5/100 (Prime luxury and high-value target).
- **Cluster 2 — Young High-Propensity Shoppers:** Mean Age: 25.2 years | Annual Income: $29.4k | Spending Score: 80.3/100 (Early-career trend adopters).
- **Cluster 3 — Frugal Core Customers:** Mean Age: 45.2 years | Annual Income: $35.8k | Spending Score: 31.3/100 (Budget-conscious household buyers).

## Screenshots
Segmentation profiles, PCA coordinate projections, and validation tables are saved in `results/segmentation_report.json`.

## Future Improvements
- Implement automated optimal $k$ selection using the Elbow method and Silhouette maximization.
- Incorporate Gaussian Mixture Models (GMM) with Expectation-Maximization for soft clustering.
- Integrate interactive 3D visualizations with Plotly.

## Author
**Ozodbek Dalaboyev**  
Undergraduate in Artificial Intelligence, Tashkent University of Information Technologies (TUIT)  
GitHub: [@OzodbekX06](https://github.com/OzodbekX06) &bull; Email: ozodbekd06@gmail.com

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.
