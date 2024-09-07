
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from PIL import Image

from pandas.plotting import parallel_coordinates
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from skimage.feature import hog
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform

def load_image(path):
    return np.array(Image.open(path).convert('L'))


def compute_similarity(img1, img2):
    return ssim(img1, img2)


def compute_mse(img1, img2):
    return mean_squared_error(img1, img2)


def compute_hog_similarity(img1, img2):
    hog1 = hog(img1)
    hog2 = hog(img2)
    return np.dot(hog1, hog2) / (np.linalg.norm(hog1) * np.linalg.norm(hog2))


def get_glyph_paths(root_dir):
    glyph_paths = {}
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            for dataset in os.listdir(class_path):
                dataset_path = os.path.join(class_path, dataset)
                if os.path.isdir(dataset_path):
                    for letter in os.listdir(dataset_path):
                        letter_path = os.path.join(dataset_path, letter)
                        if os.path.isdir(letter_path):
                            for img in os.listdir(letter_path):
                                if img.endswith('.png'):
                                    key = f"{class_name}_{dataset}_{letter}_{img}"
                                    glyph_paths[key] = os.path.join(letter_path, img)
    return glyph_paths


def compute_similarity_matrices(glyph_paths):
    keys = list(glyph_paths.keys())
    n = len(keys)
    ssim_matrix = np.zeros((n, n))
    mse_matrix = np.zeros((n, n))
    hog_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            img1 = load_image(glyph_paths[keys[i]])
            img2 = load_image(glyph_paths[keys[j]])

            if img1.shape != img2.shape:
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(img1.shape, img2.shape))
                img1 = np.array(Image.fromarray(img1).resize(min_shape[::-1]))
                img2 = np.array(Image.fromarray(img2).resize(min_shape[::-1]))

            ssim_similarity = compute_similarity(img1, img2)
            mse_similarity = compute_mse(img1, img2)
            #hog_similarity = compute_hog_similarity(img1, img2)

            ssim_matrix[i, j] = ssim_matrix[j, i] = ssim_similarity
            mse_matrix[i, j] = mse_matrix[j, i] = mse_similarity
            #hog_matrix[i, j] = hog_matrix[j, i] = hog_similarity

    return ssim_matrix, mse_matrix, keys


def compute_similarity_matrix(glyph_paths):
    keys = list(glyph_paths.keys())
    n = len(keys)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            img1 = load_image(glyph_paths[keys[i]])
            img2 = load_image(glyph_paths[keys[j]])

            if img1.shape != img2.shape:
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(img1.shape, img2.shape))
                img1 = np.array(Image.fromarray(img1).resize(min_shape[::-1]))
                img2 = np.array(Image.fromarray(img2).resize(min_shape[::-1]))

            similarity = compute_similarity(img1, img2)
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    return similarity_matrix, keys


def plot_similarity_heatmap(similarity_matrix, labels, title):
    plt.figure(figsize=(20, 16))
    sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels, cmap='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def perform_clustering(similarity_matrix, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(similarity_matrix)


# def plot_tsne(similarity_matrix, labels, clusters):
#     tsne = TSNE(n_components=2, random_state=42)
#     tsne_results = tsne.fit_transform(similarity_matrix)
#
#     plt.figure(figsize=(12, 8))
#     scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='viridis')
#     plt.colorbar(scatter)
#     plt.title('t-SNE visualization of glyph similarities')
#     plt.tight_layout()
#     plt.show()


def plot_tsne(similarity_matrix, labels):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(similarity_matrix)

    plt.figure(figsize=(16, 12))

    # Extract class and letter information from labels
    classes = [label.split('_')[0] for label in labels]
    letters = [label.split('_')[2] for label in labels]

    # Create a scatter plot
    for class_name in set(classes):
        mask = np.array(classes) == class_name
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], label=class_name, alpha=0.7)

    plt.title('t-SNE visualization of glyph similarities')
    plt.legend(title="Classes", loc="center left", bbox_to_anchor=(1, 0.5))

    # Annotate points with letters
    for i, letter in enumerate(letters):
        plt.annotate(letter, (tsne_results[i, 0], tsne_results[i, 1]), fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.show()


# def plot_tsne_2(similarity_matrix, labels):
#     tsne = TSNE(n_components=2, metric="precomputed", init='random', random_state=42)
#     tsne_results = tsne.fit_transform(1 - similarity_matrix)
#
#     plt.figure(figsize=(12, 10))
#     plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
#
#     for i, label in enumerate(labels):
#         plt.text(tsne_results[i, 0], tsne_results[i, 1], label, fontsize=9)
#
#     plt.title('Glyph Similarity - t-SNE Plot')
#     plt.show()


def plot_tsne_2(similarity_matrix, labels):
    tsne = TSNE(n_components=2, metric="precomputed", init='random', random_state=42)
    tsne_results = tsne.fit_transform(1 - similarity_matrix)

    plt.figure(figsize=(16, 14))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)

    for i, label in enumerate(labels):
        plt.annotate(label, (tsne_results[i, 0], tsne_results[i, 1]), fontsize=9, alpha=0.8)

    plt.title('Glyph Similarity - t-SNE Plot')
    plt.tight_layout()
    plt.show()


def analyze_specific_letter(root_dir, target_letter='A'):
    letter_paths = {}
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            for dataset in os.listdir(class_path):
                dataset_path = os.path.join(class_path, dataset)
                if os.path.isdir(dataset_path):
                    letter_path = os.path.join(dataset_path, target_letter)
                    if os.path.isdir(letter_path):
                        for img in os.listdir(letter_path):
                            if img.endswith('.png'):
                                key = f"{class_name}_{dataset}"
                                letter_paths[key] = os.path.join(letter_path, img)

    if not letter_paths:
        print(f"No images found for letter '{target_letter}'. Please check the dataset structure and letter name.")
        return

    similarity_matrix, keys = compute_similarity_matrix(letter_paths)

    # Convert similarity matrix to distance matrix
    distance_matrix = 1 - similarity_matrix

    # Perform hierarchical clustering
    linked = linkage(squareform(distance_matrix), 'ward')

    plt.figure(figsize=(20, 10))
    dendrogram(linked, labels=keys, leaf_rotation=90, leaf_font_size=8)
    plt.title(f'Hierarchical Clustering of Letter {target_letter} Across Datasets and Classes')
    plt.tight_layout()
    plt.show()

    # Perform t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(distance_matrix)

    plt.figure(figsize=(16, 12))
    classes = [key.split('_')[0] for key in keys]
    datasets = [key.split('_')[1] for key in keys]

    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[hash(c) for c in classes], cmap='tab20')

    # Create a legend for classes
    class_handles = [plt.scatter([], [], c=hash(c), label=c) for c in set(classes)]
    plt.legend(handles=class_handles, title="Classes", loc="center left", bbox_to_anchor=(1, 0.5))

    # Add labels for datasets
    for i, dataset in enumerate(datasets):
        plt.annotate(dataset, (tsne_results[i, 0], tsne_results[i, 1]), fontsize=8, alpha=0.7)

    plt.title(f't-SNE Visualization of Letter {target_letter} Similarities')
    plt.tight_layout()
    plt.show()


# def plot_dendrogram(similarity_matrix, labels):
#     linkage_matrix = linkage(similarity_matrix, method='ward')
#     plt.figure(figsize=(20, 10))
#     dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=8)
#     plt.title('Hierarchical Clustering Dendrogram')
#     plt.tight_layout()
#     plt.show()


# def plot_dendrogram(similarity_matrix, labels):
#     linked = linkage(1 - similarity_matrix, method='ward')  # Using 1 - similarity as the distance metric
#
#     plt.figure(figsize=(20, 10))
#     dendrogram(linked,
#                orientation='top',
#                labels=labels,
#                distance_sort='descending',
#                show_leaf_counts=True)
#     plt.title('Dendrogram of Glyph Similarities')
#     plt.show()


def plot_mds(similarity_matrix, labels):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    mds_results = mds.fit_transform(1 - similarity_matrix)

    plt.figure(figsize=(12, 10))
    plt.scatter(mds_results[:, 0], mds_results[:, 1])

    for i, label in enumerate(labels):
        plt.text(mds_results[i, 0], mds_results[i, 1], label, fontsize=9)

    plt.title('Glyph Similarity - MDS Plot')
    plt.show()





def plot_clustered_heatmap(similarity_matrix, labels):
    sns.clustermap(similarity_matrix, xticklabels=labels, yticklabels=labels, cmap='viridis', method='ward')
    plt.title('Clustered Heatmap of Glyph Similarities')
    plt.show()


def plot_pairplot(similarity_matrix, labels):
    df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    sns.pairplot(df)
    plt.suptitle('Pairwise Glyph Similarity Plot')
    plt.show()


def plot_parallel_coordinates(similarity_matrix, labels):
    df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    df = df.reset_index()
    df = pd.melt(df, id_vars=['index'], var_name='Glyph', value_name='Similarity')

    plt.figure(figsize=(12, 10))
    parallel_coordinates(df, class_column='index', cols=['Similarity'])
    plt.title('Parallel Coordinates Plot of Glyph Similarities')
    plt.show()


def plot_scatter_of_selected_glyphs(similarity_matrix, labels, selected_labels):
    indices = [labels.index(label) for label in selected_labels]
    selected_matrix = similarity_matrix[np.ix_(indices, indices)]
    selected_labels = [labels[i] for i in indices]

    plt.figure(figsize=(8, 6))
    plt.scatter(selected_matrix[0], selected_matrix[1])

    for i, label in enumerate(selected_labels):
        plt.text(selected_matrix[0][i], selected_matrix[1][i], label, fontsize=12)

    plt.title('Scatter Plot of Selected Glyphs')
    plt.xlabel('Glyph 1')
    plt.ylabel('Glyph 2')
    plt.show()


# Usage
root_dir = '../data/paleography'  # paleography dataset contains classes like 'Classical', 'Hellenistic', etc.

glyph_paths = get_glyph_paths(root_dir)

ssim_matrix, mse_matrix, labels = compute_similarity_matrices(glyph_paths)

# Plot heatmaps
# plot_similarity_heatmap(ssim_matrix, labels, 'SSIM Similarity Heatmap')
# plot_similarity_heatmap(1 / (mse_matrix + 1), labels, 'Inverse MSE Similarity Heatmap')

# Perform clustering
clusters = perform_clustering(ssim_matrix, 4)

# Plot t-SNE visualization
plot_tsne(ssim_matrix, labels, clusters)

plot_tsne_2(ssim_matrix, labels)

# Analyze specific letter (e.g., 'A')
analyze_specific_letter(root_dir, 'A')


# Plot dendrogram
# plot_dendrogram(ssim_matrix, labels)

# Print cluster information
for i, label in enumerate(labels):
    print(f"Glyph: {label}, Cluster: {clusters[i]}")


plot_mds(ssim_matrix, labels)



# plot_clustered_heatmap(mse_matrix, labels)

# plot_pairplot(ssim_matrix, labels)

# plot_parallel_coordinates(ssim_matrix, labels)

selected_labels = ['A']
plot_scatter_of_selected_glyphs(ssim_matrix, labels, selected_labels)
