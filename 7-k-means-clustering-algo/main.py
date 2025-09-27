"""
K-Means experiments: run K-Means on two datasets (synthetic blobs + Iris)
for multiple values of K, compute metrics, and visualize results.

Requirements:
  numpy
  matplotlib
  scikit-learn
  pandas (optional, used for nicer table printing)

Run:
  python kmeans_experiments.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def run_kmeans_on_data(X, dataset_name, ks=(2,3,4,5,6), random_state=42):
    """
    Run KMeans for each k in ks, compute inertia and silhouette score,
    and plot 2D PCA visualizations of the clusterings.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    results = []
    pca = PCA(n_components=2, random_state=random_state)
    X2 = pca.fit_transform(Xs)

    plt.figure(figsize=(15, 3 * len(ks)))  # one row per k
    for i, k in enumerate(ks, start=1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(Xs)
        inertia = kmeans.inertia_

        # Silhouette is only defined for k >= 2 and when not all points assigned same label
        sil = silhouette_score(Xs, labels) if k > 1 and len(np.unique(labels)) > 1 else float('nan')
        results.append({'dataset': dataset_name, 'k': k, 'inertia': inertia, 'silhouette': sil})

        # Plotting
        ax = plt.subplot(len(ks), 1, i)
        for lab in np.unique(labels):
            mask = labels == lab
            ax.scatter(X2[mask, 0], X2[mask, 1], s=30, label=f'cluster {lab}', alpha=0.7)
        # plot centroids in PCA space
        centroids_pca = pca.transform(kmeans.cluster_centers_)
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=150, c='k', label='centroids')
        ax.set_title(f"{dataset_name} — KMeans k={k} — inertia={inertia:.2f} — silhouette={sil:.3f}")
        ax.legend(loc='best', fontsize='small')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results)

def main():
    random_state = 42

    # 1) Synthetic dataset: make_blobs (easy to see cluster changes with k)
    X_blobs, y_blobs = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=random_state)
    ks = [2, 3, 4, 5, 6, 8]

    print("Running KMeans on synthetic blobs dataset...")
    df_blobs = run_kmeans_on_data(X_blobs, dataset_name="Synthetic Blobs", ks=ks, random_state=random_state)
    print("\nResults (Synthetic Blobs):")
    print(df_blobs.to_string(index=False))

    # 2) Real dataset: Iris (4 features) — KMeans can discover the three species approximately
    iris = load_iris()
    X_iris = iris.data
    ks_iris = [2, 3, 4, 5, 6]

    print("\n\nRunning KMeans on Iris dataset...")
    df_iris = run_kmeans_on_data(X_iris, dataset_name="Iris", ks=ks_iris, random_state=random_state)
    print("\nResults (Iris):")
    print(df_iris.to_string(index=False))

    # Combine and show summary
    print("\n\nSummary combined:")
    print(pd.concat([df_blobs, df_iris], ignore_index=True))

if __name__ == "__main__":
    main()
