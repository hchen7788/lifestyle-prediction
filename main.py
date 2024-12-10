import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import faiss
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import utils

df = utils.load_data(partition="all")
features = df.drop('WORK_LIFE_BALANCE_SCORE', axis=1)
target = df['WORK_LIFE_BALANCE_SCORE']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)
y_test_values = y_test.values

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

mlp_embeddings = None
with open('mlp_model_embeddings.pkl', 'rb') as file:
    mlp_embeddings = pickle.load(file)

mlp = None
with open('mlp_model.pkl', 'rb') as file:
    mlp = pickle.load(file)

embeddings = mlp_embeddings.predict(X_test_scaled)

similarity_graph = utils.get_similarity_graph(embeddings)

# louvain_clusters = utils.get_clusters(similarity_graph, threshold=0.5, method="louvain")


num_clusters = 5
faiss_clusters = utils.get_clusters(method="faiss", embeddings=embeddings, k=num_clusters)
# index: data index; elements: corresponding cluster indices
utils.plot_faiss_clusters(cluster_assignments=faiss_clusters, target=y_test_values)

utils.plot_clusters_vs_features(
    cluster_assignments=faiss_clusters, 
    features=X_test,
    target=y_test_values,
    output_dir="plots",
    filename_prefix="sorted_cluster"
)