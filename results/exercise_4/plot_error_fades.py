import pandas as pd
import matplotlib.pyplot as plt

# Load the summary data
gmf = pd.read_csv("GMF_summary.csv")
mlp = pd.read_csv("MLP_summary.csv")
neumf = pd.read_csv("NeuMF_summary.csv")

def plot_with_shaded_error(x, y, std, label, color, linestyle):
    plt.plot(x, y, linestyle=linestyle, color=color, label=label)
    plt.fill_between(x, y - std, y + std, color=color, alpha=0.2)

# HR@10 Plot
plt.figure(figsize=(4, 3))
plot_with_shaded_error(gmf["epoch"], gmf["HR_mean"], gmf["HR_std"], 'GMF', 'blue', '--')
plot_with_shaded_error(mlp["epoch"], mlp["HR_mean"], mlp["HR_std"], 'MLP', 'purple', '-')
plot_with_shaded_error(neumf["epoch"], neumf["HR_mean"], neumf["HR_std"], 'NeuMF', 'red', '-')

plt.xlabel("Iteration", fontsize=10, weight='bold')
plt.ylabel("HR@10", fontsize=10, weight='bold')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.title("MovieLens-100K", fontsize=11, weight='bold')
plt.legend(loc='lower right', frameon=True, fontsize=8, edgecolor='black')
plt.tight_layout()
plt.savefig("hr_vs_epoch_with_band.png", dpi=300)
plt.show()

# NDCG@10 Plot
plt.figure(figsize=(4, 3))
plot_with_shaded_error(gmf["epoch"], gmf["NDCG_mean"], gmf["NDCG_std"], 'GMF', 'blue', '--')
plot_with_shaded_error(mlp["epoch"], mlp["NDCG_mean"], mlp["NDCG_std"], 'MLP', 'purple', '-')
plot_with_shaded_error(neumf["epoch"], neumf["NDCG_mean"], neumf["NDCG_std"], 'NeuMF', 'red', '-')

plt.xlabel("Iteration", fontsize=10, weight='bold')
plt.ylabel("NDCG@10", fontsize=10, weight='bold')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.title("MovieLens-100K", fontsize=11, weight='bold')
plt.legend(loc='lower right', frameon=True, fontsize=8, edgecolor='black')
plt.tight_layout()
plt.savefig("ndcg_vs_epoch_with_band.png", dpi=300)
plt.show()

# Loss Plot
plt.figure(figsize=(4, 3))
plot_with_shaded_error(gmf["epoch"], gmf["Loss_mean"], gmf["Loss_std"], 'GMF', 'blue', '--')
plot_with_shaded_error(mlp["epoch"], mlp["Loss_mean"], mlp["Loss_std"], 'MLP', 'purple', '-')
plot_with_shaded_error(neumf["epoch"], neumf["Loss_mean"], neumf["Loss_std"], 'NeuMF', 'red', '-')

plt.xlabel("Iteration", fontsize=10, weight='bold')
plt.ylabel("Loss", fontsize=10, weight='bold')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.title("MovieLens-100K", fontsize=11, weight='bold')
plt.legend(loc='upper right', frameon=True, fontsize=8, edgecolor='black')
plt.tight_layout()
plt.savefig("loss_vs_epoch_with_band.png", dpi=300)
plt.show()
