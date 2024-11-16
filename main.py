import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import faiss
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import utils

# load data
df = utils.load_data(partition="all")
# df = df[:500]
features = df.drop('WORK_LIFE_BALANCE_SCORE', axis=1)
target = df['WORK_LIFE_BALANCE_SCORE']

# split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
y_test_values = y_test.values
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
embeddings = np.array(embeddings).astype('float32')

# faiss clustering
k = 3
# Initialize the k-means object
kmeans = faiss.Kmeans(d=embeddings.shape[1], k=k, niter=20, verbose=True)
# Train the clustering model
kmeans.train(embeddings)
# Get the centroids (cluster centers)
centroids = kmeans.centroids
# Assign each data point to a cluster
_, cluster_assignments = kmeans.index.search(embeddings, 1)  # `1` means get 1 nearest centroid
# Output results
print("Centroids shape:", centroids.shape)  # (k, 32)
print("Cluster assignments shape:", cluster_assignments.shape)  # (3195, 1)
cluster_assignments = [cluster_assignments[i][0] for i in range(len(cluster_assignments))]

# plt.figure(figsize=(8, 6))
# plt.scatter(cluster_assignments, y_test_values, alpha=0.7, c=cluster_assignments, cmap='viridis')
# plt.show()

clusters = defaultdict(list)
for i in range(len(cluster_assignments)):
    clusters[cluster_assignments[i]].append(i)
print(clusters)

utils.plot_clusters_vs_scores(clusters, y_test_values)

# predictions = mlp.predict(X_test)

# calculate cosine similarities and similarity graph
# similarity_graph = utils.get_similarity_graph(embeddings)

# minmax_scaler = MinMaxScaler()

# flattened_graph = similarity_graph.flatten()

# scaled_flattened_graph = minmax_scaler.fit_transform(flattened_graph.reshape(-1, 1)).flatten()

# scaled_similarity_graph = scaled_flattened_graph.reshape(similarity_graph.shape)

# optional: print a few sample predictions and their corresponding actual values
# for i in range(len(y_test)):
    # print(f"Predicted: {predictions[i]}, Actual: {y_test.iloc[i]}")


# calculate and print mean squared error
# mse = mean_squared_error(y_test, predictions)
# print(f"Mean Squared Error on the test set: {mse}")
# mae = mean_absolute_error(y_test, predictions)
# print(f"Mean Absolute Error on the test set: {mae}")
# r2 = r2_score(y_test, predictions)
# print(f"R^2 Score on the test set: {r2}")



# clustering (Louvain, etc)
# clusters = utils.get_clusters(scaled_similarity_graph, threshold=0.5, method="louvain")

# Calculate average work-life balance score per cluster
cluster_averages = utils.calculate_cluster_averages(clusters, target)

print("Average work-life balance score per cluster:", cluster_averages)

# Plot with well-being scores and cluster colors, save to file
# utils.plot_clusters_with_scores(scaled_similarity_graph, clusters, target, filename="network_graph.png")

# Plot clusters vs. scores, save to file
utils.plot_clusters_vs_scores(clusters, target, filename="scatter_clusters_vs_scores.png")


# Print each cluster with its nodes
for cluster, nodes in clusters.items():
    print(f"Cluster {cluster} has nodes: {nodes}")