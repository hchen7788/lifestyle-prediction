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



# clustering (Louvain, etc)
clusters = utils.get_clusters(scaled_similarity_graph, threshold=0.5, method="louvain")

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