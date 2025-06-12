# NMF_evaluate.py
import numpy as np
import pandas as pd
import argparse
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.decomposition import NMF
from time import time
from tqdm import tqdm

from Dataset import Dataset
from evaluate import evaluate_model

class NMFWrapper:
    """
    A wrapper class to make the scikit-learn NMF model compatible
    with the existing evaluation pipeline.
    """
    def __init__(self, user_factors, item_factors):
        """
        Initializes the wrapper with the trained factor matrices.
        Args:
            user_factors (W): The user latent factor matrix from NMF.
            item_factors (H): The item latent factor matrix from NMF.
        """
        self.user_factors = user_factors
        self.item_factors = item_factors

    def predict(self, inputs, batch_size=100, verbose=0):
        """
        Predicts scores for given user-item pairs.
        The 'inputs' format is [user_array, item_array] to match
        the Keras model's predict signature used in evaluate.py.
        """
        user_indices = inputs[0]
        item_indices = inputs[1]

        # Get the corresponding user and item vectors
        users = self.user_factors[user_indices]
        items = self.item_factors[:, item_indices]

        # Calculate the dot product for each pair to get the score
        # (user_vector * item_vector.T) -> sum(W[u,k] * H[k,i]) for each u,i pair
        predictions = np.sum(users * items.T, axis=1)
        
        # Reshape to (batch_size, 1) to match Keras output
        return predictions.reshape(-1, 1)

def parse_args():
    parser = argparse.ArgumentParser(description="Run NMF evaluation.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-100k',
                        help='Choose a dataset (e.g., ml-100k or ml-1m).')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # --- 1. Load Data ---
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print(f"Load data done [{time() - t1:.1f} s]. #user={num_users}, #item={num_items}, #train={train.nnz}, #test={len(testRatings)}")

    # --- 2. Create User-Item Matrix for Training ---
    # Convert the DOK training matrix to a CSR matrix for efficient row slicing
    print("Building training matrix...")
    train_matrix = csr_matrix(train)

    # --- 3. Experiment Loop ---
    latent_factors_range = range(1, 31, 5) # 1, 6, 11, 16, 21, 26
    topK = 10
    evaluation_threads = 1 # Set to 1 for simplicity, can be increased
    
    results = []
    best_ndcg = 0
    best_factors = 0

    print("\nStarting NMF evaluation for different numbers of latent factors...")
    for n_factors in tqdm(latent_factors_range, desc="Evaluating NMF"):
        # --- 3a. Train the NMF Model ---
        t_start_train = time()
        # Initialize the NMF model
        # 'l1_ratio=0' makes it a pure Frobenius norm regularization (L2)
        model = NMF(n_components=n_factors, init='random', random_state=0, max_iter=200, solver='cd', l1_ratio=0.0)
        
        # Fit the model to get user (W) and item (H) factors
        W = model.fit_transform(train_matrix)
        H = model.components_
        t_end_train = time()
        
        # --- 3b. Create Wrapper and Evaluate ---
        nmf_wrapper = NMFWrapper(W, H)
        t_start_eval = time()
        (hits, ndcgs) = evaluate_model(nmf_wrapper, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        t_end_eval = time()
        
        results.append({'factors': n_factors, 'HR@10': hr, 'NDCG@10': ndcg})

        print(f"Factors: {n_factors:2d}, HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}, "
              f"Train Time: {t_end_train - t_start_train:.1f}s, Eval Time: {t_end_eval - t_start_eval:.1f}s")
        
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_factors = n_factors
    
    print("\n--- Evaluation Complete ---")
    print("Results Summary:")
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("nmf_results.csv", index=False)

    print(f"\nBest Parameter Setting:")
    print(f"The best performance was achieved with {best_factors} latent factors, resulting in an NDCG@10 of {best_ndcg:.4f}.")