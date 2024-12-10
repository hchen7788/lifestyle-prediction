# lifestyle-prediction

## Code Structure
### Train MLP model
Run ```python3 train_mlp.py``` to train the MLP model, the model that predicts the WLB score is saved in file ```mlp_model.pkl```, and model to extract embeddings is saved in file ```mlp_model_embeddings.pkl``` for reuse, no need to train it repeatedly. Can load it directly to access the trained model. If want to retrain the model for other features, simply modify the data loading section and run ```python3 train_mlp.py```.

### Run the main flow
Run ```python3 main.py``` to run the main flow, steps are as commented. This includes generating the results for the naive method, loading the MLP model and creating embeddings, clustering the plain data and embeddings for different clustering methods, generating evaluation of the MLP performance and clustering resutls, generating plots for clustering for other features.

### Adding functionalities
Can add functions in ```utils.py``` and use in other files. The current file contains helper functions for clustering, generating plots, etc.

### Experiment Results
Clustering plots based on WLB score can be found in the root folder with corresponding clustering method names. Clustering plots for different lifestlye factors can be found in the ```naive_plots``` and ```plots``` folder for clustering results from the naive method and our approach.
