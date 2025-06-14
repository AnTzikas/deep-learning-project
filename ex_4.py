import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import argparse


def parse_metric_block(output):
    epoch_metrics = []
    lines = output.strip().split("\n")
    for line in lines:
        # Init line (only HR and NDCG, no loss)
        match_init = re.search(r"Init:\s*HR\s*=\s*(\d+\.\d+),\s*NDCG\s*=\s*(\d+\.\d+)", line)
        if match_init:
            epoch_metrics.append({
                "epoch": 0,
                "HR@10": float(match_init.group(1)),
                "NDCG@10": float(match_init.group(2)),
                "loss": None  # or 0.0, up to you
            })

        # Iteration lines (HR, NDCG, and loss)
        match_iter = re.search(
            r"Iteration\s+(\d+).*?HR\s*=\s*(\d+\.\d+),\s*NDCG\s*=\s*(\d+\.\d+),\s*loss\s*=\s*(\d+\.\d+)", line
        )
        if match_iter:
            epoch_metrics.append({
                "epoch": int(match_iter.group(1)) + 1,  # +1 so epoch starts from 1
                "HR@10": float(match_iter.group(2)),
                "NDCG@10": float(match_iter.group(3)),
                "loss": float(match_iter.group(4))
            })
    return pd.DataFrame(epoch_metrics)

def run_command(cmd):
    print(f"\n> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def main():
    NUM_RUNS = 10
    EPOCHS = 50
    DATASET = "ml-100k"
    LAYERS = "[32,16]"
    REG_LAYERS = "[0,0]"
    # LAYERS = "[16]"
    # REG_LAYERS = "[0]"
    NUM_FACTORS = "16"
    all_dfs = []
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['GMF', 'MLP', 'NeuMF'], required=True)
    args = parser.parse_args()

    if args.model == 'GMF':
        MODEL = "GMF.py"
        cmd = [
            "python", MODEL,
            "--dataset", DATASET,
            "--num_factors", NUM_FACTORS,
            "--epochs", str(EPOCHS),
            "--out", "0",
        ]
    elif args.model == 'MLP':
        MODEL = "MLP.py"
        cmd = [
            "python", MODEL,
            "--dataset", DATASET,
            "--layers", LAYERS,
            "--reg_layers", REG_LAYERS,
            "--epochs", str(EPOCHS),
            "--out", "0",
        ]
    elif args.model == 'NeuMF':
        MODEL = "NeuMF.py"
        cmd = [
            "python", MODEL,
            "--dataset", DATASET,
            "--num_factors", NUM_FACTORS,
            "--layers", LAYERS,
            "--reg_layers", REG_LAYERS,
            "--epochs", str(EPOCHS),
            "--out", "0",
        ]

    # === Execute training runs ===
    for run in range(1, NUM_RUNS + 1):
        print(f"\n  Running {MODEL} - Run {run}/{NUM_RUNS}")

        output = run_command(cmd)

        df_run = parse_metric_block(output)
        print(df_run)
        df_run["run"] = run
        all_dfs.append(df_run)

    # === Aggregate results ===
    filename=f"exercise_4/{args.model}_all_metrics"
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(f"{filename}.csv", index=False)

    summary = df_all.groupby("epoch").agg({
        "HR@10": ['mean', 'std'],
        "NDCG@10": ['mean', 'std'],
        "loss": ['mean', 'std']
    }).reset_index()

    summary.columns = ["epoch", "HR_mean", "HR_std", "NDCG_mean", "NDCG_std", "Loss_mean", "Loss_std"]
    filename=f"exercise_4/{args.model}_summary"
    summary.to_csv(f"{filename}.csv", index=False)


if __name__ == '__main__':
    main()
