import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from sklearn.cluster import KMeans

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
    pass

def get_clusters(similarity_graph):
    pass


# Assuming that the similarity graph is a 2D matrix passed
# Change threshold depending on results
# Documentation: https://python-louvain.readthedocs.io/en/latest/api.html
def louvain_method(similarity_graph, threshold=0.2):
    print('inside louvain_method')
    assert len(similarity_graph[0]) == len(similarity_graph), 'Similarity graph should be square matrix!'

    def is_symmetric(graph_sim):
        sim_graph_np = np.array(graph_sim)
        return np.array_equal(sim_graph_np, sim_graph_np.T)
    
    assert is_symmetric(similarity_graph), 'Similarity graph has to be symmetric!'
    
    graph = nx.Graph()

    num_nodes = len(similarity_graph)

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            # TODO: Check if > or >= is more appropriate
            if similarity_graph[i][j] >= threshold:
                # TODO: Check if labels can be accessed via similarity graph. That way the similarity can 
                graph.add_edge(i, j, weight=similarity_graph[i][j])
    
    best_partition = community_louvain.best_partition(graph, resolution=1.5)

    clusters = {}
    for node, community in best_partition.items():
        # Every node [0 indexed] will be assigned a community
        # node 0 -> community 0
        # node 1 -> community 0

        clusters.setdefault(community, []).append(node)
    
    return clusters

def kmeans(similarity_graph, number_clusters):
    similarity_graph_np = np.array(similarity_graph)
    # We want similar nodes to be closer
    distance_graph = 1 - similarity_graph_np
    kmeans = KMeans(n_clusters=number_clusters)
    labels = kmeans.fit_predict(distance_graph)

    clusters = {}

    for idx, label in enumerate(labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    return clusters


    

