# calculate_nmf_params.py
import pandas as pd
import argparse
from Dataset import Dataset

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Calculate NMF model parameters.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-100k',
                        help='Choose a dataset (e.g., ml-100k or ml-1m).')
    return parser.parse_args()

def calculate_and_display_params(num_users, num_items, factors_range):
    """
    Calculates and displays the number of parameters for NMF for a given
    range of latent factors.
    """
    results_data = []

    # The formula for NMF parameters is: (num_users * k) + (num_items * k)
    # where k is the number of latent factors.
    
    for k in factors_range:
        user_factor_params = num_users * k
        item_factor_params = num_items * k
        total_params = user_factor_params + item_factor_params
        
        results_data.append({
            'Latent Factors (k)': k,
            'User-Factor Params (M * k)': user_factor_params,
            'Item-Factor Params (N * k)': item_factor_params,
            'Total Parameters': total_params
        })
        
    # Create and print the results table using pandas
    results_df = pd.DataFrame(results_data)
    
    print("\n--- NMF Parameter Calculation ---")
    print(f"Based on M={num_users} users and N={num_items} items.")
    print("\nFormula: Total Parameters = (M + N) * k")
    
    # Use to_string() for clean formatting without the index
    print(results_df.to_string(index=False))

    # --- Discussion part ---
    print("\n--- Discussion ---")
    print("As the results table clearly demonstrates, the relationship between the number of")
    print("latent factors (k) and the total number of trainable parameters in the NMF model")
    print("is perfectly linear.")
    print(f"For each additional latent factor, the model's complexity increases by a fixed")
    print(f"amount equal to the sum of the number of users and items (M + N = {num_users + num_items}).")
    print("This predictable, linear scaling is a characteristic of standard matrix factorization models.")


if __name__ == '__main__':
    args = parse_args()
    
    # --- 1. Load Data to get dimensions ---
    print(f"Loading dataset '{args.dataset}' to determine dimensions...")
    dataset = Dataset(args.path + args.dataset)
    num_users, num_items = dataset.trainMatrix.shape
    print(f"Dataset loaded successfully: {num_users} users, {num_items} items.")
    
    # --- 2. Define the range and run the calculation ---
    latent_factors_range = [1, 5, 10, 15, 20, 25, 30]  # 1, 6, 11, 16, 21, 26
    
    calculate_and_display_params(num_users, num_items, latent_factors_range)