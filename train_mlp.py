import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import utils

df = utils.load_data(partition="all")
features = df.drop('WORK_LIFE_BALANCE_SCORE', axis=1)
target = df['WORK_LIFE_BALANCE_SCORE']

# split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# fit MLP model
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=50, warm_start=True, random_state=42)
epochs = 100
for epoch in tqdm(range(epochs), desc="Training Progress"):
    mlp.fit(X_train, y_train)

# save model
with open('mlp_model.pkl', 'wb') as file:
    pickle.dump(mlp, file)