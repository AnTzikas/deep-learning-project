import subprocess
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

NUM_RUNS = 10
DATASET = 'ml-100k'
NUM_FACTORS = '32'
EPOCHS = '10'

CONFIGS = [
    # {'name': '1 Layer', 'layers_arg': '[16]', 'reg_layers_arg': '[0]'},
    # {'name': '2 Layers', 'layers_arg': '[32,16]', 'reg_layers_arg': '[0,0]'},
    # {'name': '3 Layers', 'layers_arg': '[64,32,16]', 'reg_layers_arg': '[0,0,0]'}
    {'name': '4 Layers', 'layers_arg': '[64,32,16,8]', 'reg_layers_arg': '[0,0,0,0]'}
]

def run_command(cmd):
    print(f"\n> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def parse_metrics(output):
    match = re.search(r"End\. Best Iteration \d+:.*?HR\s*=\s*(\d+\.\d+),\s*NDCG\s*=\s*(\d+\.\d+)", output)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def parse_saved_model(output):
    match = re.search(r"The best.*?model is saved to (Pretrain/.*\.npy)", output)
    if match:
        return match.group(1)
    return None

def main():
    results = []

    for run in range(1, NUM_RUNS + 1):
        print(f"\n========== RUN {run}/{NUM_RUNS} ==========\n")

        # ---- Phase 1: Without Pretraining ----
        for cfg in tqdm(CONFIGS, desc="No Pretraining"):
            if run == 1:
                cmd = [
                    'python', 'NeuMF.py', '--dataset', DATASET, '--num_factors', NUM_FACTORS,
                    '--layers', cfg['layers_arg'], '--reg_layers', cfg['reg_layers_arg'],
                    '--epochs', EPOCHS, '--out', '1'
                ]
            else:
                cmd = [
                    'python', 'NeuMF.py', '--dataset', DATASET, '--num_factors', NUM_FACTORS,
                    '--layers', cfg['layers_arg'], '--reg_layers', cfg['reg_layers_arg'],
                    '--epochs', EPOCHS, '--out', '0'
                ]
            output = run_command(cmd)
            hr, ndcg = parse_metrics(output)
            if hr is not None:
                results.append({
                    "Run": run, "Config": cfg['name'], "Pretrain": 0,
                    "HR@10": hr, "NDCG@10": ndcg
                })

        # ---- Train GMF ----
        # gmf_cmd = [
        #     'python', 'GMF.py', '--dataset', DATASET, '--num_factors', NUM_FACTORS,
        #     '--epochs', EPOCHS, '--out', '1'
        # ]
        # gmf_out = run_command(gmf_cmd)
        # gmf_model = parse_saved_model(gmf_out)

        # # ---- For each config: train MLP + then NeuMF with pretraining ----
        # for cfg in tqdm(CONFIGS, desc="With Pretraining"):
        #     mlp_cmd = [
        #         'python', 'MLP.py', '--dataset', DATASET, '--layers', cfg['layers_arg'],
        #         '--reg_layers', cfg['reg_layers_arg'], '--epochs', EPOCHS, '--out', '1'
        #     ]
        #     mlp_out = run_command(mlp_cmd)
        #     mlp_model = parse_saved_model(mlp_out)

        #     neumf_cmd = [
        #         'python', 'NeuMF.py', '--dataset', DATASET, '--num_factors', NUM_FACTORS,
        #         '--layers', cfg['layers_arg'], '--reg_layers', cfg['reg_layers_arg'],
        #         '--epochs', EPOCHS, '--out', '0',
        #         '--mf_pretrain', gmf_model, '--mlp_pretrain', mlp_model
        #     ]
        #     output = run_command(neumf_cmd)
        #     hr, ndcg = parse_metrics(output)
        #     if hr is not None:
        #         results.append({
        #             "Run": run, "Config": cfg['name'], "Pretrain": 1,
        #             "HR@10": hr, "NDCG@10": ndcg
        #         })

    # ---- Save Raw Data ----
    df = pd.DataFrame(results)
    df.to_csv("results_hr_ndcg.csv", index=False)
    print("Saved all results to results_hr_ndcg.csv")

    # ---- Summary (Mean ± Std) ----
    summary = []
    for cfg in CONFIGS:
        for pretrain_flag in [0, 1]:
            subset = df[(df["Config"] == cfg['name']) & (df["Pretrain"] == pretrain_flag)]
            label = f"{cfg['name']} | {'With' if pretrain_flag else 'No'} Pretrain"
            if not subset.empty:
                hr_mean = subset["HR@10"].mean()
                hr_std = subset["HR@10"].std()
                ndcg_mean = subset["NDCG@10"].mean()
                ndcg_std = subset["NDCG@10"].std()
                summary.append({
                    "Configuration": label,
                    "HR@10": f"{hr_mean:.4f} ± {hr_std:.4f}",
                    "NDCG@10": f"{ndcg_mean:.4f} ± {ndcg_std:.4f}"
                })

    df_summary = pd.DataFrame(summary)
    print("\nSummary Results (Mean ± Std over 10 runs):\n")
    print(df_summary.to_string(index=False))
    df_summary.to_csv("summary_results.csv", index=False)

if __name__ == '__main__':
    main()
