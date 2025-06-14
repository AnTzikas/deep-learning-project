import pandas as pd
import matplotlib.pyplot as plt

# Load your results (make sure the path is correct)
df = pd.read_csv("nmf_results_averaged.csv")

# Plot
plt.figure(figsize=(10, 6))

# HR@10 plot
plt.errorbar(df['factors'], df['HR@10_mean'], yerr=df['HR@10_std'], fmt='-o', capsize=4, label="HR@10")

# NDCG@10 plot
plt.errorbar(df['factors'], df['NDCG@10_mean'], yerr=df['NDCG@10_std'], fmt='-s', capsize=4, label="NDCG@10")

plt.xlabel("Number of Latent Factors")
plt.ylabel("Metric Value")
plt.title("NMF Performance vs Latent Factors")
plt.legend()
plt.grid(True)
plt.xticks(df['factors'])
plt.tight_layout()
plt.savefig("nmf_hr_ndcg_vs_factors.png")
plt.show()
