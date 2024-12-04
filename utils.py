import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering


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
    

def get_clusters(similarity_graph, threshold=0.7, resolution=1.0, method="louvain", n_clusters=None):
    """
    Detect clusters in a graph using specified clustering method.
    
    Parameters:
    - similarity_graph: 2D array representing the similarity graph
    - threshold: Minimum similarity value to consider an edge (applies to louvain)
    - resolution: Resolution parameter for Louvain (applies to louvain)
    - method: Clustering method ("louvain" or "spectral")
    - n_clusters: Number of clusters for spectral clustering (required if method is "spectral")
    
    Returns:
    - clusters: Dictionary where keys are cluster labels and values are lists of node indices
    """
    if method == "louvain":
        return louvain_method(similarity_graph, threshold, resolution)
    elif method == "spectral":
        if n_clusters is None:
            raise ValueError("For spectral clustering, you must specify the number of clusters (n_clusters).")
        return spectral_clustering_method(similarity_graph, n_clusters)
    elif method == "girvan_newman":
        if n_clusters is None:
            raise ValueError("For girvan-newman clustering, you must specify the number of clusters (n_clusters).")
        return girvan_newman_method(similarity_graph, threshold, max_clusters=n_clusters)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

# # Assuming that the similarity graph is a 2D matrix passed
# # Change threshold depending on results
# # Documentation: https://python-louvain.readthedocs.io/en/latest/api.html
# def louvain_method(similarity_graph, threshold=0.2):
#     print('inside louvain_method')
#     assert len(similarity_graph[0]) == len(similarity_graph), 'Similarity graph should be square matrix!'

#     def is_symmetric(graph_sim):
#         sim_graph_np = np.array(graph_sim)
#         return np.array_equal(sim_graph_np, sim_graph_np.T)
    
#     assert is_symmetric(similarity_graph), 'Similarity graph has to be symmetric!'
    
#     graph = nx.Graph()

#     num_nodes = len(similarity_graph)

#     for i in range(num_nodes):
#         for j in range(i+1, num_nodes):
#             # TODO: Check if > or >= is more appropriate
#             if similarity_graph[i][j] >= threshold:
#                 # TODO: Check if labels can be accessed via similarity graph. That way the similarity can 
#                 graph.add_edge(i, j, weight=similarity_graph[i][j])
    
#     best_partition = community_louvain.best_partition(graph, resolution=1.5)

#     clusters = {}
#     for node, community in best_partition.items():
#         # Every node [0 indexed] will be assigned a community
#         # node 0 -> community 0
#         # node 1 -> community 0

#         clusters.setdefault(community, []).append(node)
    
#     return clusters


def louvain_method(similarity_graph, threshold, resolution):
    print('inside louvain_method')
    
    # Check that the similarity graph is a square matrix
    assert similarity_graph.shape[0] == similarity_graph.shape[1], 'Similarity graph should be a square matrix!'
    
    # Ensure the similarity graph is symmetric
    def is_symmetric(graph_sim):
        return np.allclose(graph_sim, graph_sim.T)

    assert is_symmetric(similarity_graph), 'Similarity graph has to be symmetric!'
    
    # Create an undirected weighted graph from the similarity matrix
    graph = nx.from_numpy_array(similarity_graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # Remove edges with weight less than the threshold
    edges_to_remove = [(u, v) for u, v, weight in graph.edges(data="weight") if weight < threshold]
    graph.remove_edges_from(edges_to_remove)

    # Apply the Louvain method for community detection
    best_partition = community_louvain.best_partition(graph, resolution=resolution)

    # Group nodes by their community
    clusters = {}
    for node, community in best_partition.items():
        clusters.setdefault(community, []).append(node)
    
    return clusters

def spectral_clustering_method(similarity_graph, n_clusters, threshold=None):
    """
    Perform spectral clustering on a similarity graph with optional thresholding.
    
    Parameters:
    - similarity_graph: 2D array representing the similarity graph
    - n_clusters: Number of clusters to create
    - threshold: Minimum similarity value to consider an edge (optional)
    
    Returns:
    - clusters: Dictionary where keys are cluster labels and values are lists of node indices
    """
    
    # Apply thresholding to sparsify the similarity graph if a threshold is provided
    if threshold is not None:
        similarity_graph = np.where(similarity_graph >= threshold, similarity_graph, 0)
    
    # Apply Spectral Clustering
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",  # Use the similarity matrix as the affinity matrix
        random_state=42
    )
    
    # Fit and get cluster labels
    labels = spectral.fit_predict(similarity_graph)
    
    # Group nodes by their cluster
    clusters = {}
    for node, cluster in enumerate(labels):
        clusters.setdefault(cluster, []).append(node)
    
    return clusters

def girvan_newman_method(similarity_graph, threshold, max_clusters):
    """
    Perform Girvan-Newman clustering on a similarity graph.

    Parameters:
    - similarity_graph: 2D array representing the similarity graph
    - max_clusters: Maximum number of clusters to generate
    
    Returns:
    - clusters: Dictionary where keys are cluster labels and values are lists of node indices
    """
    # Check that the similarity graph is a square matrix
    assert similarity_graph.shape[0] == similarity_graph.shape[1], 'Similarity graph should be a square matrix!'
    
    # Ensure the similarity graph is symmetric
    def is_symmetric(graph_sim):
        return np.allclose(graph_sim, graph_sim.T)

    assert is_symmetric(similarity_graph), 'Similarity graph has to be symmetric!'
    
    # Create an undirected weighted graph from the similarity matrix
    graph = nx.from_numpy_array(similarity_graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # Remove edges with weight less than the threshold
    edges_to_remove = [(u, v) for u, v, weight in graph.edges(data="weight") if weight < threshold]
    graph.remove_edges_from(edges_to_remove)
    
    # Apply the Girvan-Newman method for community detection
    communities_generator = nx.community.girvan_newman(graph)
    
    # Stop when the desired number of clusters is reached
    for communities in communities_generator:
        if len(communities) >= max_clusters:
            break
    
    # Assign clusters based on the resulting communities
    clusters = {}
    for i, community in enumerate(communities):
        for node in community:
            clusters.setdefault(i, []).append(node)
    
    return clusters



def calculate_cluster_averages(clusters, target_values):
    """
    Calculate the average work-life balance score for each cluster.
    
    Parameters:
    - clusters: dict where keys are cluster labels and values are lists of node indices
    - target_values: Series or list of target values (work-life balance scores) corresponding to each node
    
    Returns:
    - dict with cluster labels as keys and their average work-life balance score as values
    """
    cluster_averages = {}
    for cluster, nodes in clusters.items():
        scores = target_values.iloc[nodes]
        cluster_averages[cluster] = scores.mean()
    
    return cluster_averages

def plot_clusters_with_scores(similarity_graph, clusters, scores, filename="clusters_with_scores.png"):
    """
    Plot clusters with work-life balance scores in a network graph.
    
    Parameters:
    - similarity_graph: 2D array representing the similarity graph
    - clusters: dict where keys are cluster labels and values are lists of node indices
    - scores: Series or list of scores corresponding to each node
    - filename: Name of the file to save the plot
    """
    graph = nx.Graph(similarity_graph)
    # Create a color map for clusters
    unique_clusters = list(clusters.keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    cluster_color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

    # Create network plot
    pos = nx.spring_layout(graph)  # Positions for all nodes

    # Draw nodes with color based on cluster and size based on work-life balance score
    for cluster, nodes in clusters.items():
        node_sizes = [1 for score in scores]  # [scores[node] * 20 for node in nodes]  
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_size=node_sizes,
                               node_color=[cluster_color_map[cluster]] * len(nodes),
                               label=f"Cluster {cluster}")
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, alpha=0.5)

    # Add labels
    nx.draw_networkx_labels(graph, pos, font_size=8)
    
    # Add a legend
    plt.legend(scatterpoints=1, loc="upper right", markerscale=0.5, fontsize=8)
    plt.title("Network Graph of Work-Life Balance Scores by Cluster")

    # Save plot to file
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

def plot_clusters_vs_scores(clusters, scores, filename="clusters_vs_scores.png"):
    """
    Plot each point where the y-axis is the cluster index (sorted by average work-life balance score)
    and the x-axis is the work-life balance score. Each cluster is represented with a different color.
    
    Parameters:
    - clusters: dict where keys are cluster labels and values are lists of node indices
    - scores: Series or list of work-life balance scores corresponding to each node
    - filename: Name of the file to save the plot
    """
    # Calculate average work-life balance score for each cluster
    cluster_averages = {cluster: scores.iloc[nodes].mean() for cluster, nodes in clusters.items()}

    # Sort clusters by average work-life balance score
    sorted_clusters = sorted(cluster_averages.items(), key=lambda x: x[1])
    sorted_cluster_indices = {cluster: idx for idx, (cluster, _) in enumerate(sorted_clusters)}

    # Create a color map for clusters
    unique_clusters = list(sorted_cluster_indices.keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    cluster_color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot each cluster
    for cluster, nodes in clusters.items():
        cluster_scores = scores.iloc[nodes]
        sorted_y = sorted_cluster_indices[cluster]
        # plt.scatter(cluster_scores, [sorted_y] * len(cluster_scores), 
        #             color=cluster_color_map[cluster], label=f"Cluster {cluster}", alpha=0.7)
        plt.scatter([sorted_y] * len(cluster_scores), cluster_scores,
                    color=cluster_color_map[cluster], label=f"Cluster {cluster}", alpha=0.7)

    # Add labels and legend
    # plt.xlabel("Work-Life Balance Score")
    # plt.ylabel("Cluster Index (Sorted by Avg Work-life Balance Score)")
    plt.ylabel("Work-Life Balance Score")
    plt.xlabel("Cluster Index (Sorted by Avg Work-life Balance Score)")
    plt.title("Clusters by Louvain by (Sorted by Mean Work-Life Balance")
    # plt.legend(loc="best", fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save plot to file
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()
