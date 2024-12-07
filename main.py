import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, normalized_mutual_info_score
import faiss
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import utils

# load data
df = utils.load_data(partition="all")
features = df.drop('WORK_LIFE_BALANCE_SCORE', axis=1)
target = df['WORK_LIFE_BALANCE_SCORE']

# split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
y_test_values = y_test.values

# normalize features
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# load model
mlp_embeddings = None
with open('mlp_model_embeddings.pkl', 'rb') as file:
    mlp_embeddings = pickle.load(file)

# load model
mlp = None
with open('mlp_model.pkl', 'rb') as file:
    mlp = pickle.load(file)

# extract embeddings
embeddings = mlp_embeddings.predict(X_test_scaled)

# calculate cosine similarities and similarity graph
similarity_graph = utils.get_similarity_graph(embeddings)

# louvain_clusters = utils.get_clusters(similarity_graph, threshold=0.5, method="louvain")


num_clusters = 5
# faiss clustering on embeddings
faiss_clusters = utils.get_clusters(method="faiss", embeddings=embeddings, k=num_clusters)
# faiss clustering on naive data
faiss_clusters_naive = utils.get_clusters(method="faiss", embeddings=X_test_scaled, k=num_clusters)
# index: data index; elements: corresponding cluster indices

# evaluation metrics for MLP effectiveness
# A higher Silhouette score indicates better-defined clusters
sil_mlp = silhouette_score(embeddings, faiss_clusters)
sil_naive = silhouette_score(X_test_scaled, faiss_clusters_naive)
print(f"Silhouette Score (MLP): {sil_mlp}")
print(f"Silhouette Score (Naive): {sil_naive}")

# A lower DBI value indicates better clustering
dbi_mlp = davies_bouldin_score(embeddings, faiss_clusters)
dbi_naive = davies_bouldin_score(X_test_scaled, faiss_clusters_naive)
print(f"Davies-Bouldin Index (MLP): {dbi_mlp}")
print(f"Davies-Bouldin Index (Naive): {dbi_naive}")

# NMI ranges from 0 to 1, where 1 indicates perfect alignment
nmi_mlp = normalized_mutual_info_score(y_test_values, faiss_clusters)
nmi_naive = normalized_mutual_info_score(y_test_values, faiss_clusters_naive)
print(f"NMI (MLP): {nmi_mlp}")
print(f"NMI (Naive): {nmi_naive}")


utils.plot_faiss_clusters(cluster_assignments=faiss_clusters, target=y_test_values)

utils.plot_clusters_vs_features(
    cluster_assignments=faiss_clusters, 
    features=X_test,
    target=y_test_values,
    output_dir="plots",
    filename_prefix="sorted_cluster"
)