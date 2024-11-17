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

# calculate cosine similarities and similarity graph
similarity_graph = utils.get_similarity_graph(embeddings)

# clustering (Louvain, etc)
# louvain_clusters = utils.get_clusters(similarity_graph, threshold=0.5, method="louvain")

# faiss clustering
faiss_clusters = utils.get_clusters(method="faiss", embeddings=embeddings, k=10)
# index: data index; elements: corresponding cluster indices
utils.plot_faiss_clusters(cluster_assignments=faiss_clusters, target=y_test_values)
