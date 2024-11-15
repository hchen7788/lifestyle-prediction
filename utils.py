import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
from sklearn.metrics.pairwise import cosine_similarity


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
    

def get_clusters(similarity_graph, method="louvain"):
    if method == "louvain":
        return louvain_method(similarity_graph)    


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


#Louvain sean version
def louvain_method(similarity_graph, threshold=0.2):
    print('inside louvain_method')
    
    # Check that the similarity graph is a square matrix
    assert similarity_graph.shape[0] == similarity_graph.shape[1], 'Similarity graph should be a square matrix!'
    
    # Ensure the similarity graph is symmetric
    def is_symmetric(graph_sim):
        return np.allclose(graph_sim, graph_sim.T)

    assert is_symmetric(similarity_graph), 'Similarity graph has to be symmetric!'
    
    # # Create an undirected weighted graph
    # graph = nx.Graph()
    # num_nodes = similarity_graph.shape[0]

    # # Add edges based on thresholded similarity values
    # for i in range(num_nodes):
    #     for j in range(num_nodes):
    #         # if similarity_graph[i, j] >= threshold:
    #         if i != j:
    #             graph.add_edge(i, j, weight=similarity_graph[i, j])

    graph = nx.from_numpy_array(similarity_graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # Apply the Louvain method for community detection
    best_partition = community_louvain.best_partition(graph, resolution=1.25)

    # Group nodes by their community
    clusters = {}
    for node, community in best_partition.items():
        clusters.setdefault(community, []).append(node)
    
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

def plot_clusters_with_scores(similarity_graph, clusters, scores):
    
    graph = nx.Graph(similarity_graph)
    # Create a color map for clusters
    unique_clusters = list(clusters.keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    cluster_color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

    # Create network plot
    pos = nx.spring_layout(graph)  # Positions for all nodes

    # Draw nodes with color based on cluster and size based on work-life balance score
    for cluster, nodes in clusters.items():
        node_sizes = [1 for score in scores] #[scores[node] * 20 for node in nodes]  # Scale size by score
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_size=node_sizes,
                               node_color=[cluster_color_map[cluster]] * len(nodes),
                               label=f"Cluster {cluster}")
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, alpha=0.5)

    # Add labels
    nx.draw_networkx_labels(graph, pos, font_size=8)
    
    # Add a legend
    plt.legend(scatterpoints=1, loc="upper right", markerscale=0.5, fontsize=8)
    plt.title("Network Graph of Well-being Scores by Cluster")
    plt.show()

