import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import utils

# load data
df = utils.load_data(partition="all")
# df = df[:500]
features = df.drop('WORK_LIFE_BALANCE_SCORE', axis=1)
target = df['WORK_LIFE_BALANCE_SCORE']

# split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# normalize features
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# load model
mlp_embeddings = None
with open('mlp_model_embeddings.pkl', 'rb') as file:
    mlp_embeddings = pickle.load(file)

# load model
mlp = None
with open('mlp_model.pkl', 'rb') as file:
    mlp = pickle.load(file)

# extract embeddings
embeddings = mlp_embeddings.predict(X_test)

predictions = mlp.predict(X_test)

# calculate cosine similarities and similarity graph
similarity_graph = utils.get_similarity_graph(embeddings)

minmax_scaler = MinMaxScaler()

flattened_graph = similarity_graph.flatten()

scaled_flattened_graph = minmax_scaler.fit_transform(flattened_graph.reshape(-1, 1)).flatten()

scaled_similarity_graph = scaled_flattened_graph.reshape(similarity_graph.shape)

# calculate and print mean squared error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error on the test set: {mse}")

# optional: print a few sample predictions and their corresponding actual values
for i in range(len(y_test)):
    print(f"Predicted: {predictions[i]}, Actual: {y_test.iloc[i]}")



# Calculate Louvain clusters and their averages
# louvain_clusters = utils.get_clusters(scaled_similarity_graph, method="louvain", threshold=0.5, resolution=1.0)
# 10 clusters
# louvain_clusters = utils.get_clusters(scaled_similarity_graph, method="louvain", threshold=0.5, resolution=1.0409705)
# 5 clusters
louvain_clusters = utils.get_clusters(scaled_similarity_graph, method="louvain", threshold=0.5, resolution=1.025)
louvain_cluster_averages = utils.calculate_cluster_averages(louvain_clusters, target)

print("Louvain clustering results:")
for cluster, avg in louvain_cluster_averages.items():
    print(f"Cluster {cluster} average work-life balance score: {avg}")

# Plot Louvain clusters
utils.plot_clusters_vs_scores(louvain_clusters, target, filename="louvain_clusters_vs_scores_5cls.png")
utils.plot_clusters_with_scores(scaled_similarity_graph, louvain_clusters, target, filename="louvain_network_graph.png")

# Calculate Spectral clusters and their averages
spectral_clusters = utils.get_clusters(scaled_similarity_graph, method="spectral", n_clusters=5)
spectral_cluster_averages = utils.calculate_cluster_averages(spectral_clusters, target)

print("\nSpectral clustering results:")
for cluster, avg in spectral_cluster_averages.items():
    print(f"Cluster {cluster} average work-life balance score: {avg}")

# Plot Spectral clusters
utils.plot_clusters_vs_scores(spectral_clusters, target, filename="spectral_clusters_vs_scores.png")
utils.plot_clusters_with_scores(scaled_similarity_graph, spectral_clusters, target, filename="spectral_network_graph.png")

# Print each cluster with its nodes for both methods
print("\nLouvain Clustering Nodes:")
for cluster, nodes in louvain_clusters.items():
    print(f"Cluster {cluster} has nodes: {nodes}")

print("\nSpectral Clustering Nodes:")
for cluster, nodes in spectral_clusters.items():
    print(f"Cluster {cluster} has nodes: {nodes}")

# Girvan-Newman Clustering
girvan_newman_clusters = utils.get_clusters(scaled_similarity_graph, method="girvan_newman", n_clusters=5)
girvan_newman_cluster_averages = utils.calculate_cluster_averages(girvan_newman_clusters, target)

print("\nGirvan-Newman clustering results:")
for cluster, avg in girvan_newman_cluster_averages.items():
    print(f"Cluster {cluster} average work-life balance score: {avg}")

# Plot Girvan-Newman clusters
utils.plot_clusters_vs_scores(girvan_newman_clusters, target, filename="girvan_newman_clusters_vs_scores.png")
# utils.plot_clusters_with_scores(scaled_similarity_graph, girvan_newman_clusters, target, filename="girvan_newman_network_graph.png")

# Print each cluster with its nodes for Girvan-Newman
print("\nGirvan-Newman Clustering Nodes:")
for cluster, nodes in girvan_newman_clusters.items():
    print(f"Cluster {cluster} has nodes: {nodes}")