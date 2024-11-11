import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=50, warm_start=True, random_state=42)
# epochs = 100
# for epoch in tqdm(range(epochs), desc="Training Progress"):
#     mlp.fit(X_train, y_train)

# define and train the MLP model
input_layer = Input(shape=(X_train.shape[1],))
x = Dense(64, activation='relu')(input_layer)
x = Dense(32, activation='relu')(x)  # embedding layer
output_layer = Dense(1, activation='linear')(x)  # output layer for regression

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse') # loss: 1.8138
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# create an embedding model to extract the last hidden layer as the embedding layer
embedding_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# save model
with open('mlp_model.pkl', 'wb') as file:
    pickle.dump(embedding_model, file)