import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import utils

# load data
df = utils.load_data(partition="all")
features = df.drop('WORK_LIFE_BALANCE_SCORE', axis=1)
target = df['WORK_LIFE_BALANCE_SCORE']

# split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# normalize features
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# load model
mlp = None
with open('mlp_model.pkl', 'rb') as file:
    mlp = pickle.load(file)

# extract embeddings
embeddings = mlp.predict(X_test)

# calculate cosine similarities and similarity graph
similarity_graph = utils.get_similarity_graph(embeddings)

# clustering (Louvain, etc)
clusters = utils.get_clusters(similarity_graph)