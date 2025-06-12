import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv("NeuMF_summary.csv")  # replace with your actual filename
df = df[df['epoch'] <= 10]  # Keep K = 1 to 10 (epoch 1-10)

# Create figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Performance of NeuMF with Respect to K (Top-K Recommendation)", fontsize=14)

# Plot HR@K
axs[0].errorbar(df['epoch'], df['HR_mean'], yerr=df['HR_std'], fmt='-o', color='crimson', label='HR@K')
axs[0].set_title('(a) HR@K')
axs[0].set_xlabel('K')
axs[0].set_ylabel('HR@K')
axs[0].set_xticks(df['epoch'])
axs[0].grid(True)
axs[0].legend()

# Plot NDCG@K
axs[1].errorbar(df['epoch'], df['NDCG_mean'], yerr=df['NDCG_std'], fmt='-o', color='navy', label='NDCG@K')
axs[1].set_title('(b) NDCG@K')
axs[1].set_xlabel('K')
axs[1].set_ylabel('NDCG@K')
axs[1].set_xticks(df['epoch'])
axs[1].grid(True)
axs[1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figure_5_topk_metrics.png")
plt.show()
