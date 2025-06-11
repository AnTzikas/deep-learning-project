# prepare_ml100k.py
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os

def process_ml100k(file_path, output_path):
    """
    Loads the ml-100k u.data file, converts IDs to be 0-indexed,
    performs a leave-one-out split, and saves the data into the NCF formats.
    """
    dataset_name = "ml-100k"
    print(f"Loading raw data from {file_path}...")
    
    # Define column names as the file has no header
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    # Read the tab-separated file
    df = pd.read_csv(file_path, sep='\t', header=None, names=column_names)

    # --- 1. CRITICAL: Convert 1-based IDs to 0-based IDs ---
    print("Converting 1-based user/item IDs to 0-based...")
    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1
    
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    print(f"Found {num_users} users and {num_items} items.")
    # Check if the counts match the dataset description (943 users, 1682 items)
    assert num_users == 943, "User count does not match expected 943."
    assert num_items == 1682, "Item count does not match expected 1682."

    # --- 2. Perform Leave-One-Out split ---
    print("Performing leave-one-out split based on timestamp...")
    # Sort by user and timestamp to find the last interaction for each user
    df.sort_values(by=['user_id', 'timestamp'], inplace=True)
    
    # The last item for each user is the test set
    test_df = df.groupby('user_id').tail(1)
    
    # All other items are the training set
    train_df = df.drop(test_df.index)

    # Sanity check
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    assert len(train_df) + len(test_df) == len(df), "Split failed!"

    # --- 3. Create the 'Data/' directory if it doesn't exist ---
    data_dir = os.path.join(output_path, 'Data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    # --- 4. Save the training and test rating files ---
    train_file_path = os.path.join(data_dir, f"{dataset_name}.train.rating")
    test_file_path = os.path.join(data_dir, f"{dataset_name}.test.rating")

    # The NCF code expects a placeholder "rating" column for the train file.
    # The format is user_id, item_id, rating, timestamp
    train_df_to_save = train_df[['user_id', 'item_id', 'rating', 'timestamp']]

    print(f"Saving training data to {train_file_path}...")
    train_df_to_save.to_csv(train_file_path, sep='\t', header=False, index=False)

    print(f"Saving test data to {test_file_path}...")
    test_df[['user_id', 'item_id']].to_csv(test_file_path, sep='\t', header=False, index=False)

    # --- 5. Generate and save the negative samples file ---
    negative_file_path = os.path.join(data_dir, f"{dataset_name}.test.negative")
    print(f"Generating negative samples and saving to {negative_file_path}...")

    # Get a set of all unique item IDs for fast sampling
    all_item_ids = set(df['item_id'].unique())
    
    # Create a map of all items each user has interacted with
    user_history = df.groupby('user_id')['item_id'].apply(set)

    num_neg_samples = 99
    with open(negative_file_path, 'w') as f:
        # Use tqdm for a progress bar
        for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Generating negatives"):
            user_id = row['user_id']
            pos_item_id = row['item_id']
            
            interacted_items = user_history[user_id]
            negative_samples = []
            
            while len(negative_samples) < num_neg_samples:
                # Randomly sample an item from the entire item set
                candidate_item = np.random.choice(list(all_item_ids))
                # If the user has not interacted with it, it's a valid negative sample
                if candidate_item not in interacted_items:
                    negative_samples.append(candidate_item)
            
            # Format the line: (user_id,pos_item_id) \t neg1 \t neg2 ...
            positive_pair = f"({user_id},{pos_item_id})"
            negative_list = "\t".join(map(str, negative_samples))
            f.write(f"{positive_pair}\t{negative_list}\n")

    print(f"\nDataset '{dataset_name}' preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the MovieLens 100K dataset for NCF.")
    parser.add_argument('--file', type=str, required=True,
                        help='Path to the raw u.data file.')
    parser.add_argument('--out', type=str, default='.',
                        help='Output path for the processed data. Defaults to current directory.')
    args = parser.parse_args()

    process_ml100k(args.file, args.out)