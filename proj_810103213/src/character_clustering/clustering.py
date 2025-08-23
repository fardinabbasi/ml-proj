import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def perform_pca_analysis(features, n_comps = None):

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_comps, random_state=42)
    features_reduced = pca.fit_transform(features_scaled)
    
    print(f"PCA selected {pca.n_components_} components.")

    return features_reduced, pca

def kmeans_clustering(features, n_clusters):
    
    print(f"Running K-Means with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(features)
    print("K-Means clustering complete.")
    return labels

def dbscan_clustering(features, eps, min_samples):

    print(f"Running DBSCAN with eps={eps} and min_samples={min_samples}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    print("DBSCAN clustering complete.")
    return labels

def agglomerative_clustering(features, n_clusters):

    print(f"Running Agglomerative Clustering with n_clusters={n_clusters}...")
    agg = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    labels = agg.fit_predict(features)
    print("Agglomerative clustering complete.")
    return labels


def calculate_cluster_metrics(features, labels):

    metrics = {}

    if len(set(labels)) > 1:
        
        sil_score = silhouette_score(features, labels)
        metrics['silhouette_score'] = sil_score
        print(f"  - Silhouette Score: {sil_score:.4f}")
        
        db_score = davies_bouldin_score(features, labels)
        metrics['davies_bouldin_score'] = db_score
        print(f"  - Davies-Bouldin Score: {db_score:.4f}")
        
    else:
        print("Only one cluster found. Metrics cannot be calculated.")

    return metrics
