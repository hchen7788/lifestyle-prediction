import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sklearn.cluster import KMeans, SpectralClustering

def load_data(partition):
    
    df = pd.read_csv('./data/Wellbeing_and_lifestyle_data_Kaggle.csv')
    df = df[df['DAILY_STRESS'] != '1/1/00']
    df["AGE"] = df['AGE'].map({"Less than 20": 0, "21 to 35": 1, "36 to 50": 2, "51 or more": 3}).fillna(0)
    df["GENDER"] = df["GENDER"].map({"Female": 0, "Male": 1}).fillna(0)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df = df.drop(columns=['Timestamp'])

    # condition on partition to get different partitions of data
    print("Dataset loaded successfully.")
    print("data partition: " + partition)

    return df

def get_similarity_graph(embeddings):
    return cosine_similarity(embeddings)
    

def get_clusters(similarity_graph=None, threshold=0.5, resolution=1.0, method="louvain", embeddings=None, k=3, n_clusters=5):
    if method == "louvain":
        return louvain_method(similarity_graph, threshold, resolution)
    elif method == "faiss":
        return faiss_method(embeddings, k)
    elif method == "spectral":
        return spectral_method(embeddings, n_clusters)
    elif method == "girvan_newman":
        return girvan_newman_method(embeddings, threshold, max_communities=n_clusters)



def louvain_method(similarity_graph, threshold, resolution):
    print('inside louvain_method')
    
    assert similarity_graph.shape[0] == similarity_graph.shape[1], 'Similarity graph should be a square matrix!'
    
    def is_symmetric(graph_sim):
        return np.allclose(graph_sim, graph_sim.T)

    assert is_symmetric(similarity_graph), 'Similarity graph has to be symmetric!'
    
    graph = nx.from_numpy_array(similarity_graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    edges_to_remove = [(u, v) for u, v, weight in graph.edges(data="weight") if weight < threshold]
    graph.remove_edges_from(edges_to_remove)

    best_partition = community_louvain.best_partition(graph, resolution=resolution)

    clusters = {}
    for node, community in best_partition.items():
        clusters.setdefault(community, []).append(node)
    
    return clusters

def kmeans(similarity_graph, number_clusters):
    similarity_graph_np = np.array(similarity_graph)
    # We want similar nodes to be closer
    distance_graph = 1 - similarity_graph_np
    kmeans = KMeans(n_clusters=number_clusters)
    labels = kmeans.fit_predict(distance_graph)

def faiss_method(embeddings, k=3):
    kmeans = faiss.Kmeans(d=embeddings.shape[1], k=k, niter=20, verbose=True)
    kmeans.train(embeddings)
    centroids = kmeans.centroids
    _, cluster_assignments = kmeans.index.search(embeddings, 1)
    # print("Cluster assignments shape:", cluster_assignments.shape)  # (3195, 1)
    cluster_assignments = [cluster_assignments[i][0] for i in range(len(cluster_assignments))]

    return cluster_assignments

def spectral_method(embeddings, n_clusters=5):
    print('Using spectral clustering...')
    similarity_graph = get_similarity_graph(embeddings)

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=1
    )
    cluster_assignments = spectral.fit_predict(similarity_graph)
    return cluster_assignments

def girvan_newman_method(embeddings, threshold=0.5, max_communities=5):
    print('Using Girvan-Newman clustering...')
    similarity_graph = get_similarity_graph(embeddings)
    
    graph = nx.from_numpy_array(similarity_graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    edges_to_remove = [(u, v) for u, v, weight in graph.edges(data="weight") if weight < threshold]
    graph.remove_edges_from(edges_to_remove)

    communities_generator = nx.algorithms.community.girvan_newman(graph)

    communities = []
    for i, partition in enumerate(communities_generator):
        communities.append(partition)
        print(f"Partition {i + 1}: {partition}")

        if len(partition) >= max_communities:
            break

    clusters = {i: list(community) for i, community in enumerate(communities[-1])}
    
    return clusters


def calculate_cluster_averages(clusters, target_values):
    cluster_averages = {}
    for cluster, nodes in clusters.items():
        scores = target_values.iloc[nodes]
        cluster_averages[cluster] = scores.mean()
    
    return cluster_averages


def plot_clusters_vs_scores(clusters, scores, filename="clusters_vs_scores.png"):

    cluster_averages = {cluster: scores.iloc[nodes].mean() for cluster, nodes in clusters.items()}

    sorted_clusters = sorted(cluster_averages.items(), key=lambda x: x[1])
    sorted_cluster_indices = {cluster: idx for idx, (cluster, _) in enumerate(sorted_clusters)}

    unique_clusters = list(sorted_cluster_indices.keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    cluster_color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

    plt.figure(figsize=(10, 6))

    for cluster, nodes in clusters.items():
        cluster_scores = scores.iloc[nodes]
        sorted_y = sorted_cluster_indices[cluster]
        plt.scatter(cluster_scores, [sorted_y] * len(cluster_scores), 
                    color=cluster_color_map[cluster], label=f"Cluster {cluster}", alpha=0.7)

    plt.xlabel("Work-Life Balance Score")
    plt.ylabel("Cluster Index (Sorted by Avg Work-life Balance Score)")
    plt.title("Work-Life Balance Scores by Sorted Clusters")
    # plt.legend(loc="best", fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()

def plot_faiss_clusters(cluster_assignments=None, target=None):
    cluster_assignments = np.array(cluster_assignments)
    target = np.array(target)
    
    unique_clusters = np.unique(cluster_assignments)
    cluster_means = {cluster: target[cluster_assignments == cluster].mean() for cluster in unique_clusters}
    
    sorted_clusters = sorted(cluster_means.keys(), key=lambda x: cluster_means[x])
    
    sorted_mapping = {cluster: i for i, cluster in enumerate(sorted_clusters)}
    sorted_cluster_assignments = np.array([sorted_mapping[cluster] for cluster in cluster_assignments])
    
    plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(
    #     sorted_cluster_assignments, target, 
    #     alpha=0.7, c=sorted_cluster_assignments, cmap='viridis'
    # )
    unique_clusters = sorted(set(sorted_cluster_assignments))  # Unique cluster indices
    grouped_targets = [
        [target[i] for i in range(len(sorted_cluster_assignments)) if sorted_cluster_assignments[i] == cluster]
        for cluster in unique_clusters
    ]
    plt.boxplot(grouped_targets, positions=unique_clusters, widths=0.6, patch_artist=True)
    plt.title("Clusters by Girvan-Newman (Sorted by Mean Work-Life Balance)")
    plt.xlabel("Cluster Index (Sorted by Mean)")
    plt.ylabel("Work-Life Balance Score")
    plt.xticks(range(len(sorted_clusters)), labels=[f"{i}" for i in range(len(sorted_clusters))])
    plt.savefig("faiss_clusters_5.png")

def plot_clusters_vs_features(cluster_assignments, features, target=None, output_dir="plots", filename_prefix="cluster_vs_feature"):
    os.makedirs(output_dir, exist_ok=True)

    cluster_assignments = np.array(cluster_assignments)

    if isinstance(features, pd.DataFrame):
        feature_names = features.columns  # Use DataFrame column names
        features = features.values.astype(float)  # Convert DataFrame to NumPy array and ensure numeric values
    else:
        num_features = features.shape[1]
        feature_names = [f"Feature {i}" for i in range(num_features)]  # Generate generic feature names for NumPy array
        features = features.astype(float)  # Ensure numeric values

    if target is not None:
        target = np.array(target)
        unique_clusters = np.unique(cluster_assignments)
        cluster_means = {cluster: target[cluster_assignments == cluster].mean() for cluster in unique_clusters}
        sorted_clusters = sorted(cluster_means.keys(), key=lambda x: cluster_means[x])
    else:
        sorted_clusters = sorted(np.unique(cluster_assignments))
    
    sorted_mapping = {cluster: i for i, cluster in enumerate(sorted_clusters)}
    sorted_cluster_assignments = np.array([sorted_mapping[cluster] for cluster in cluster_assignments])
    
    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(8, 6))

        cluster_feature_values = [[] for _ in range(len(sorted_clusters))]
        for cluster, value in zip(sorted_cluster_assignments, features[:, i]):
            cluster_feature_values[cluster].append(value)

        plt.boxplot(cluster_feature_values, positions=range(len(sorted_clusters)), patch_artist=True)
        plt.title(f"Clusters by FAISS (Sorted) vs {feature_name}")
        plt.xlabel("Cluster Index (Sorted by Mean Work-Life Balance Score)")
        plt.ylabel(feature_name)
        plt.xticks(range(len(sorted_clusters)), labels=[f"{i}" for i in range(len(sorted_clusters))])
        filename = os.path.join(output_dir, f"{filename_prefix}_{feature_name.replace(' ', '_')}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()