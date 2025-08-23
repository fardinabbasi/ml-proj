import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# plots elbow for choosing k
def plot_kmeans_elbow(features, title, save_path, max_k = 20):

    inertia_values = []
    k_range = range(2, max_k + 1)

    print(f"Calculating inertia for k from 2 to {max_k}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(features)
        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(12, 7))
    plt.plot(k_range, inertia_values, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title(title)
    plt.xticks(k_range)
    plt.grid(True)

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    # plt.close()

    print(f"Elbow method plot saved to: {save_path}")

# visualizes clusters
def plot_clusters_tsne(features, labels, title, save_path):

    print("Visualizing clusters with t-SNE...")

    tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=2000)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(12, 8))
    unique_labels = sorted(list(set(labels)))
    n_clusters = len(unique_labels)

    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=labels,
        palette=sns.color_palette("hsv", n_clusters),
        legend="full",
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    # plt.close()

    print(f"Cluster plot saved to: {save_path}")

# saving clustering results
def save_all_cluster_images(all_character_images, labels, base_save_path):

    print(f"Saving all cluster images to: {base_save_path}")
    base_save_path.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(list(set(labels)))

    for cluster_id in unique_labels:
        cluster_dir = base_save_path / f"cluster_{cluster_id}"
        cluster_dir.mkdir(exist_ok=True)

        indices = np.where(labels == cluster_id)[0]

        for i, img_index in enumerate(indices):
            img = all_character_images[img_index]
            img.save(cluster_dir / f"image_{i}.png")

    print("All images for each cluster saved successfully.")


def plot_cluster_samples(all_character_images, labels, title_prefix, n_samples = 8):

    unique_labels = sorted(list(set(labels)))

    for cluster_id in unique_labels:
        indices = np.where(labels == cluster_id)[0]

        if len(indices) == 0:
            continue

        sample_indices = random.sample(list(indices), min(n_samples, len(indices)))

        plt.figure(figsize=(10, 4))
        title = f'{title_prefix} - Cluster {cluster_id} ({len(indices)} members)'
        plt.suptitle(title, fontsize=16)

        for i, img_index in enumerate(sample_indices):
            ax = plt.subplot(2, (n_samples + 1) // 2, i + 1)
            ax.imshow(all_character_images[img_index], cmap='gray')
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def plot_dbscan_k_distance(features, min_samples, title, save_path):

    print(f"Calculating k-distances for min_samples={min_samples}...")

    # Calculate the distance for each point to its k-th nearest neighbor
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(features)
    distances, _ = neighbors_fit.kneighbors(features)

    # Sort the distances
    sorted_distances = np.sort(distances[:, min_samples-1])

    # Create the plot
    plt.figure(figsize=(12, 7))
    plt.plot(sorted_distances)
    plt.xlabel("Points (sorted by distance)")
    plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance (eps)")
    plt.title(title)

    # Add detailed grid
    ax = plt.gca()
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.grid(which='major', linestyle='-', linewidth='0.8', color='gray')

    # Save and close
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    # plt.close()


# plots pca variance for dimension reduction
def plot_pca_variance(pca_model, title, save_path):

    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='--')

    plt.axhline(y=0.95, color='g', linestyle=':', label='95% Explained Variance')
    plt.axhline(y=0.90, color='r', linestyle=':', label='90% Explained Variance')
    plt.title(title)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend(loc='best')
    plt.ylim(0, 1.01)

    ax = plt.gca()
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.grid(which='major', linestyle='-', linewidth='0.8', color='gray')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"PCA variance plot saved to: {save_path}")


def plot_2d_features(features, labels, title, save_path, xlabel = "Feature 1", ylabel = "Feature 2"):

    print(f"Generating 2D scatter plot and saving to: {save_path}")
    
    plt.figure(figsize=(12, 8))
    unique_labels = sorted(list(set(labels)))
    n_colors = len(unique_labels)
    
    sns.scatterplot(
        x=features[:, 0],
        y=features[:, 1],
        hue=labels,
        palette=sns.color_palette("hsv", n_colors),
        legend="full",
        alpha=0.7
    )
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
