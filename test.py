import numpy as np
import utils

# Define a small, intentional array with known relationships for testing
embeddings = np.array([
    [1, 0, 0],   # Vector A
    [0, 1, 0],   # Vector B (orthogonal to A)
    [1, 1, 0],   # Vector C (similar to both A and B)
    [1, 1, 1],   # Vector D (similar to all)
    [0, 0, 1]    # Vector E (orthogonal to A, B, and C, but slightly similar to D)
])

def test_similarity_metrics(embeddings):
    metrics = ["cosine", "euclidean", "manhattan", "pearson", "jaccard", "minkowski"]
    
    for metric in metrics:
        try:
            # Calculate similarity matrix using the specified metric
            similarity_matrix = utils.get_similarity_graph(embeddings, metric=metric)
            print(f"Similarity matrix using {metric} metric:")
            print(similarity_matrix, "\n")
        except Exception as e:
            print(f"Error using {metric} metric: {e}\n")

# Run the test function
test_similarity_metrics(embeddings)
