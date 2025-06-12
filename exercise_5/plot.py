import matplotlib.pyplot as plt
import pandas as pd

summary = pd.read_csv("exercise_4/NeuMF_summary.csv")

plt.style.use("seaborn-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

# === Training Loss ===
axes[0].plot(summary["epoch"], summary["Loss_mean"], color='purple', label='NeuMF', linewidth=2)
axes[0].fill_between(summary["epoch"], summary["Loss_mean"] - summary["Loss_std"], summary["Loss_mean"] + summary["Loss_std"], alpha=0.2, color='purple')
axes[0].set_title("(a) Training Loss", fontsize=12)
axes[0].set_ylabel("Training Loss", fontsize=11)
axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylim(bottom=0)
axes[0].legend()
axes[0].tick_params(labelsize=10)

# === HR@10 ===
axes[1].plot(summary["epoch"], summary["HR_mean"], color='darkred', label='NeuMF', linewidth=2)
axes[1].fill_between(summary["epoch"], summary["HR_mean"] - summary["HR_std"], summary["HR_mean"] + summary["HR_std"], alpha=0.2, color='salmon')
axes[1].set_title("(b) HR@10", fontsize=12)
axes[1].set_ylabel("HR@10", fontsize=11)
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylim(bottom=0)
axes[1].legend()
axes[1].tick_params(labelsize=10)

# === NDCG@10 ===
axes[2].plot(summary["epoch"], summary["NDCG_mean"], color='darkblue', label='NeuMF', linewidth=2)
axes[2].fill_between(summary["epoch"], summary["NDCG_mean"] - summary["NDCG_std"], summary["NDCG_mean"] + summary["NDCG_std"], alpha=0.2, color='lightblue')
axes[2].set_title("(c) NDCG@10", fontsize=12)
axes[2].set_ylabel("NDCG@10", fontsize=11)
axes[2].set_xlabel("Epoch", fontsize=11)
axes[2].set_ylim(bottom=0)
axes[2].legend()
axes[2].tick_params(labelsize=10)

plt.suptitle("Training Loss and Recommendation Performance of NeuMF\nw.r.t. the Number of Epochs", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("metrics_plot_cleaned.png", dpi=300)
plt.show()
