# run_experiment.py (Modified to calculate Mean and Standard Deviation)
import subprocess
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
NUM_RUNS = 1
DATASET = 'ml-100k'
NUM_FACTORS = '16'
EPOCHS = '2'

# --- CONFIGURATION ---
CONFIGS = [
    {'name': '1 Layer', 'layers_arg': '[16]', 'reg_layers_arg': '[0]'},
    {'name': '2 Layers', 'layers_arg': '[32,16]', 'reg_layers_arg': '[0,0]'},
    {'name': '3 Layers', 'layers_arg': '[64,32,16]', 'reg_layers_arg': '[0,0,0]'}
]
# --------------------

def run_command(command):
    """Runs a command and returns its stdout. Prints command for clarity."""
    print(f"\n> Running: {' '.join(command)}")
    try:
        # Python 3.6 compatible version: no 'text' or 'capture_output'
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE, # Capture stdout
            stderr=subprocess.PIPE, # Capture stderr
            check=True              # Raise an error if the command fails
        )
        # Manually decode the byte string to a regular string
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        # Also decode the error output for clear printing
        print(f"---! ERROR running command: {' '.join(command)} !---")
        print(f"---! STDOUT: {e.stdout.decode('utf-8')} !---")
        print(f"---! STDERR: {e.stderr.decode('utf-8')} !---")
        return None

def parse_hr(output_string):
    """Parses the final HR value from the script's output."""
    if output_string is None: return None
    match = re.search(r"End\. Best Iteration \d+:.*?HR = (\d+\.\d+)", output_string)
    if match: return float(match.group(1))
    print("---! WARNING: Could not parse HR value from output. !---")
    return None

def parse_saved_filename(output_string):
    """Parses the saved model filename from the script's output."""
    if output_string is None: return None
    match = re.search(r"The best.*?model is saved to (Pretrain/.*\.npy)", output_string)
    if match: return match.group(1)
    print("---! WARNING: Could not parse saved filename from output. !---")
    return None
def parse_metrics(output_string):
    """Parses HR and NDCG from the final output line."""
    if output_string is None:
        return None, None
    match = re.search(r"End\. Best Iteration \d+:.*?HR\s*=\s*(\d+\.\d+),\s*NDCG\s*=\s*(\d+\.\d+)", output_string)
    if match:
        return float(match.group(1)), float(match.group(2))
    print("---! WARNING: Could not parse metrics from output. !---")
    return None, None


def main():
    """Main function to run the entire experiment."""
    results = {f"{cfg['name']} (No Pre-train)": [] for cfg in CONFIGS}
    results.update({f"{cfg['name']} (With Pre-train)": [] for cfg in CONFIGS})

    for i in range(1, NUM_RUNS + 1):
        print(f"\n{'='*20} STARTING MASTER RUN {i}/{NUM_RUNS} {'='*20}")

        # --- Phase 1: NeuMF WITHOUT Pre-training ---
        print(f"\n--- Phase 1: Running models WITHOUT pre-training (Run {i}) ---")
        for config in tqdm(CONFIGS, desc="Without Pre-train"):
            cmd = [
                'python', 'NeuMF.py', '--dataset', DATASET, '--num_factors', NUM_FACTORS,
                '--layers', config['layers_arg'], '--reg_layers', config['reg_layers_arg'],
                '--epochs', EPOCHS, '--out', '0'
            ]
            output = run_command(cmd)
            #hr = parse_hr(output)
            # if hr:
            #     results[f"{config['name']} (No Pre-train)"].append(hr)
            hr, ndcg= parse_metrics(output)
            if hr is not None:
                results[f"{config['name']} (No Pre-train)"].append({
                    "Run": i,
                    "HR@10": hr,
                    "NDCG@10": ndcg,
                })



        # # --- Phase 2: NeuMF WITH Pre-training ---
        print(f"\n--- Phase 2: Running models WITH pre-training (Run {i}) ---")
        gmf_cmd = [
            'python', 'GMF.py', '--dataset', DATASET, '--num_factors', NUM_FACTORS,
            '--epochs', EPOCHS, '--out', '1'
        ]
        gmf_output = run_command(gmf_cmd)
        gmf_model_file = parse_saved_filename(gmf_output)
        if not gmf_model_file:
            print("---! FATAL: Could not get GMF pre-trained model. Aborting pre-train phase. !---")
            continue

        for config in tqdm(CONFIGS, desc="With Pre-train   "):
            mlp_cmd = [
                'python', 'MLP.py', '--dataset', DATASET, '--layers', config['layers_arg'],
                '--reg_layers', config['reg_layers_arg'], '--epochs', EPOCHS, '--out', '1'
            ]
            mlp_output = run_command(mlp_cmd)
            mlp_model_file = parse_saved_filename(mlp_output)
            if not mlp_model_file:
                print(f"---! FATAL: Could not get MLP pre-trained model for {config['name']}. Skipping. !---")
                continue

            neumf_cmd = [
                'python', 'NeuMF.py', '--dataset', DATASET, '--num_factors', NUM_FACTORS,
                '--layers', config['layers_arg'], '--reg_layers', config['reg_layers_arg'],
                '--epochs', EPOCHS, '--out', '0',
                '--mf_pretrain', gmf_model_file, '--mlp_pretrain', mlp_model_file
            ]
            neumf_output = run_command(neumf_cmd)
            #hr = parse_hr(neumf_output)
            # if hr:
            #     results[f"{config['name']} (With Pre-train)"].append(hr)
            hr, ndcg = parse_metrics(neumf_output)
            if hr is not None:
                results[f"{config['name']} (No Pre-train)"].append({
                    "Run": i,
                    "HR@10": hr,
                    "NDCG@10": ndcg,
                })


    # ================== MODIFIED SUMMARY BLOCK STARTS HERE ==================
    print(f"\n{'='*20} EXPERIMENT COMPLETE - FINAL MEAN & STD VALUES {'='*20}")
    
    summary_data = []
    for config in CONFIGS:
        no_pretrain_key = f"{config['name']} (No Pre-train)"
        with_pretrain_key = f"{config['name']} (With Pre-train)"

        hr_list_no_pretrain = results.get(no_pretrain_key, [])
        if hr_list_no_pretrain:
            hr_values_no_pretrain = [entry["HR@10"] for entry in hr_list_no_pretrain]
            mean_no_pretrain = np.mean(hr_values_no_pretrain)
            std_no_pretrain = np.std(hr_values_no_pretrain)
            no_pretrain_str = f"{mean_no_pretrain:.4f} ± {std_no_pretrain:.4f}"
        else:
            no_pretrain_str = "N/A"

        hr_list_with_pretrain = results.get(with_pretrain_key, [])
        if hr_list_with_pretrain:
            hr_values_with_pretrain = [entry["HR@10"] for entry in hr_list_with_pretrain]
            mean_with_pretrain = np.mean(hr_values_with_pretrain)
            std_with_pretrain = np.std(hr_values_with_pretrain)
            with_pretrain_str = f"{mean_with_pretrain:.4f} ± {std_with_pretrain:.4f}"
        else:
            with_pretrain_str = "N/A"

        summary_data.append({
            "Number of MLP Layers": config['name'],
            "HR@10 (Without Pre-training)": no_pretrain_str,
            "HR@10 (With Pre-training)": with_pretrain_str
        })


    df_summary = pd.DataFrame(summary_data)
    print("\nFinal Results (Mean ± Standard Deviation over 10 runs):\n")
    print(df_summary.to_string(index=False))
    df_all = pd.DataFrame(results)
    df_all.to_csv("raw_results.csv", index=False)
    print("\n✅ All raw results saved to 'raw_results.csv'")

    # =================== MODIFIED SUMMARY BLOCK ENDS HERE ===================


if __name__ == '__main__':
    main()