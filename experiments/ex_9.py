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
    latent_factors_range = [1, 5, 10, 15, 20, 25, 30]
    topK = 10
    evaluation_threads = 1 # Set to 1 for simplicity, can be increased
    
    results = []
    best_ndcg = 0
    best_factors = 0

    print("\nStarting NMF evaluation for different numbers of latent factors...")
    for n_factors in tqdm(latent_factors_range, desc="Evaluating NMF"):
        hr_list = []
        ndcg_list = []
        runs_per_factor = 10
        for run in range(runs_per_factor):
            model = NMF(n_components=n_factors, init='random', random_state=run, max_iter=200, solver='cd', l1_ratio=0.0)
            W = model.fit_transform(train_matrix)
            H = model.components_
            nmf_wrapper = NMFWrapper(W, H)
            hits, ndcgs = evaluate_model(nmf_wrapper, testRatings, testNegatives, topK, evaluation_threads)
            hr_list.append(np.mean(hits))
            ndcg_list.append(np.mean(ndcgs))

        hr_mean = np.mean(hr_list)
        hr_std = np.std(hr_list)
        ndcg_mean = np.mean(ndcg_list)
        ndcg_std = np.std(ndcg_list)

        results.append({
            'factors': n_factors,
            'HR@10_mean': hr_mean,
            'HR@10_std': hr_std,
            'NDCG@10_mean': ndcg_mean,
            'NDCG@10_std': ndcg_std
        })

        print(f"Factors: {n_factors:2d} | HR@10: {hr_mean:.4f} ± {hr_std:.4f} | NDCG@10: {ndcg_mean:.4f} ± {ndcg_std:.4f}")


    
    df = pd.DataFrame(results)
    df.to_csv("exercise_9/nmf_results_averaged.csv", index=False)
    print("\n\n")
    print(df)
    print("\nSaved: nmf_results_averaged.csv")
