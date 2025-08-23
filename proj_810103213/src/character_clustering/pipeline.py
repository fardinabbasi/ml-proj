import numpy as np

from . import config
from PIL import Image
from .data import load_characters, extract_characters
from .features import extract_hog_features, extract_raw_pixel_features, extract_geometric_features
from .plotting import plot_kmeans_elbow, plot_clusters_tsne, plot_cluster_samples, plot_dbscan_k_distance, plot_pca_variance
from .clustering import kmeans_clustering, calculate_cluster_metrics, dbscan_clustering, agglomerative_clustering, perform_pca_analysis


def extract_features ():
    characters_path = config.RESULTS_DIR / "characters"
    figs_part2_path = config.FIGS_PART2

    figs_part2_path.mkdir(parents=True, exist_ok=True)

    if characters_path.exists() and any(characters_path.iterdir()):
        all_chars = load_characters(characters_path)
    else:
        all_chars = extract_characters()

    # Extract features
    hog_features = extract_hog_features(all_chars)
    pixel_features = extract_raw_pixel_features(all_chars)
    geometric_features = extract_geometric_features(all_chars)

    print("\nFeature Extraction Summary:")
    print(f"Shape of HOG features matrix: {hog_features.shape}")
    print(f"Shape of Raw Pixel features matrix: {pixel_features.shape}")
    print(f"Shape of Geometric features matrix: {geometric_features.shape}")
    
    return hog_features, pixel_features, geometric_features, all_chars


def kmeans(in_features, method, all_characters):
    figs_part2_path = config.FIGS_PART2
    elbow_plot_path = figs_part2_path / f"elbow_{method}.png"
    input_features = in_features

    input_tile = f"Elbow Method for {method} Features"

    plot_kmeans_elbow(
        features=input_features,
        title=input_tile,
        save_path=elbow_plot_path,
        max_k=30
    )

    if method == "HOG":
        optimal_k = 11

    elif method == "PIXELS":
        optimal_k = 9
    
    elif method == "FUSED":
        optimal_k = 5
    
    elif method == "GEOMETRIC":
        optimal_k = 6
        
    kmeans_plot_path = figs_part2_path / f"kmeans_{method}_k{optimal_k}.png"

    # get labels for k-means
    k_labels = kmeans_clustering(
        features=input_features,
        n_clusters=optimal_k
    )

    # reduced dimension into 2 to visualize clusters
    plot_clusters_tsne(
        features=input_features,
        labels=k_labels,
        title=f"K-Means on {method} Features (k={optimal_k})",
        save_path=kmeans_plot_path
    )

    print("---------------------------------------------------------------------------")
    calculate_cluster_metrics(features=input_features, labels=k_labels)
    print("---------------------------------------------------------------------------")

    # show the results
    # Image.open(elbow_plot_path)
    # Image.open(kmeans_plot_path)

    all_chars = all_characters
    plot_cluster_samples(
        all_character_images=all_chars,
        labels=k_labels,
        title_prefix=f"K-Means (k={optimal_k})",
        n_samples=8
    )

def dbscan(in_features, method, all_characters):

    figs_part2_path = config.FIGS_PART2
    input_features = in_features
    all_chars = all_characters

    if method == "HOG":
        min_samples = 10
        eps_m = 1.69

    elif method == "PIXELS":
        min_samples = 10
        eps_m = 1340

    elif method == "GEOMETRIC":
        min_samples = 10
        eps_m = 0.05

    kdist_plot_path = figs_part2_path / f"kdist_{method}_minsamp{min_samples}.png"


    plot_dbscan_k_distance(
        features=input_features,
        min_samples=min_samples,
        title=f"k-distance Graph for {method} Features (min_samples={min_samples})",
        save_path=kdist_plot_path
    )

    dbscan_plot_path = figs_part2_path / f"dbscan_{method}_eps{eps_m}.png"

    dbscan_labels = dbscan_clustering(
        features=input_features,
        eps=eps_m,
        min_samples=min_samples
    )

    n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise_ = list(dbscan_labels).count(-1)
    print(f"\nDBSCAN Results:")
    print(f"Estimated number of clusters: {n_clusters_}")
    print(f"Estimated number of noise points: {n_noise_}")

    plot_clusters_tsne(
        features=input_features,
        labels=dbscan_labels,
        title=f"DBSCAN on {method} Features (eps={eps_m})",
        save_path=dbscan_plot_path
    )

    print("---------------------------------------------------------------------------")
    calculate_cluster_metrics(features=input_features, labels=dbscan_labels)
    print("---------------------------------------------------------------------------")

    plot_cluster_samples(
        all_character_images=all_chars,
        labels=dbscan_labels,
        title_prefix=f"DBSCAN (eps={eps_m})",
        n_samples=8
    )


def Agglomerative(in_features, method, all_characters):
    
    figs_part2_path = config.FIGS_PART2
    input_features = in_features
    all_chars = all_characters

    if method == "FUESED":
        n_clusters_agg = 7

    else:
        n_clusters_agg = 16

    # run agglomerative clustering
    agg_labels = agglomerative_clustering(
        features=input_features,
        n_clusters=n_clusters_agg
    )

    print(f"\nAgglomerative Clustering Results:")
    print(f"Number of clusters created: {len(set(agg_labels))}")


    # reduced dimension to visualize clusters
    agg_tsne_path = figs_part2_path / f"agg_{method}_k{n_clusters_agg}_tsne.png"
    plot_clusters_tsne(
        features=input_features,
        labels=agg_labels,
        title=f"Agglomerative on {method} Features (k={n_clusters_agg})",
        save_path=agg_tsne_path
    )

    print("---------------------------------------------------------------------------")
    calculate_cluster_metrics(features=input_features, labels=agg_labels)
    print("---------------------------------------------------------------------------")

    plot_cluster_samples(
        all_character_images=all_chars,
        labels=agg_labels,
        title_prefix=f"Agglomerative (k={n_clusters_agg})",
        n_samples=8
    )

def create_fused_feaures(feature1, feature2, name1, name2):
    
    print(f"Performing PCA analysis on {name1} features...")
    fitted_pca1 = perform_pca_analysis(features=feature1)[1]

    plot_path1 = config.FIGS_PART2 / f"pca_variance_{name1}.png"

    plot_pca_variance(
        pca_model=fitted_pca1,
        title=f"Cumulative Explained Variance ({name1} Features)",
        save_path=plot_path1
    )

    print(f"Performing PCA analysis on {name2} features...")
    fitted_pca2 = perform_pca_analysis(features=feature2)[1]

    plot_path2 = config.FIGS_PART2 / f"pca_variance_{name2}.png"

    plot_pca_variance(
        pca_model=fitted_pca2,
        title=f"Cumulative Explained Variance ({name2} Features)",
        save_path=plot_path2
    )

    feature_reduced1, _ = perform_pca_analysis(
        features=feature1,
        n_comps=90
    )

    feature_reduced2, _ = perform_pca_analysis(
        features=feature2,
        n_comps=100
    )

    print("\nFusing feature sets...")
    fused_features = np.concatenate([feature_reduced1, feature_reduced2], axis=1)
    print(f"Final fused features created with shape: {fused_features.shape}")

    return fused_features