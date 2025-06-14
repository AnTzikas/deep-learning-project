import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_metrics(output):
    match = re.search(r"End\. Best (?:Epoch|Iteration) \d+:.*?HR\s*=\s*(\d+\.\d+),\s*NDCG\s*=\s*(\d+\.\d+)", output)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def run_command(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def main():
    NUM_RUNS = 10
    EPOCHS = "11"
    DATASET = "ml-100k"
    LAYERS = "[32,16]"
    REG_LAYERS = "[0,0]"
    NUM_FACTORS = "16"

    results = []

    for k in range(1, 11):
        print(f"\n--- Evaluating for topK = {k} ---")
        for run in range(NUM_RUNS):
            print(f"Run {run + 1}/{NUM_RUNS}")
            cmd = [
                "python", "NeuMF.py",
                "--dataset", DATASET,
                "--num_factors", NUM_FACTORS,
                "--layers", LAYERS,
                "--reg_layers", REG_LAYERS,
                "--epochs", EPOCHS,
                "--topK", str(k),
                "--out", "0"
            ]
            output = run_command(cmd)
            hr, ndcg = parse_metrics(output)
            if hr is not None:
                results.append({"K": k, "HR@K": hr, "NDCG@K": ndcg})
            else:
                print(f"Failed to parse metrics for K={k}, run={run+1}")

    df = pd.DataFrame(results)
    df.to_csv("exercise_5/topK_all_runs.csv", index=False)

    # Aggregate
    summary = df.groupby("K").agg({
        "HR@K": ["mean", "std"],
        "NDCG@K": ["mean", "std"]
    }).reset_index()
    summary.columns = ["K", "HR_mean", "HR_std", "NDCG_mean", "NDCG_std"]
    summary.to_csv("exercise_5/topK_summary.csv", index=False)


if __name__ == '__main__':
    main()
