# calculate_neumf_params.py
import argparse
from NeuMF import get_model # We import the model-building function from our NeuMF.py
from Dataset import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate NeuMF model parameters.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-100k',
                        help='Choose a dataset.')
    parser.add_argument('--num_factors', type=int, default=64,
                        help='Embedding size of GMF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load dataset to get dimensions
    dataset = Dataset(args.path + args.dataset)
    num_users, num_items = dataset.trainMatrix.shape

    # Build the model with the best parameters to count its weights
    model = get_model(num_users, num_items, args.num_factors, eval(args.layers), eval(args.reg_layers), args.reg_mf)
    
    total_params = model.count_params()
    
    print("\n--- NeuMF Parameter Calculation ---")
    print(f"For the best parameter setting:")
    print(f"  - Predictive Factors (GMF): {args.num_factors}")
    print(f"  - MLP Layers: {args.layers}")
    print(f"Based on {num_users} users and {num_items} items.")
    print(f"\nTotal number of trainable parameters: {total_params:,}")