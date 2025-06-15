import pandas as pd
import matplotlib.pyplot as plt

# Load the summary data
gmf = pd.read_csv("GMF_summary.csv")
mlp = pd.read_csv("MLP_summary.csv")
neumf = pd.read_csv("NeuMF_summary.csv")

# Plot
plt.figure(figsize=(4, 3))
plt.errorbar(gmf["epoch"], gmf["HR_mean"], yerr=gmf["HR_std"], linestyle='--', color='blue', label='GMF', capsize=2)
plt.errorbar(mlp["epoch"], mlp["HR_mean"], yerr=mlp["HR_std"], linestyle='-', color='purple', label='MLP', capsize=2)
plt.errorbar(neumf["epoch"], neumf["HR_mean"], yerr=neumf["HR_std"], linestyle='-', color='red', label='NeuMF', capsize=2)

plt.xlabel("Iteration", fontsize=10, weight='bold')
plt.ylabel("HR@10", fontsize=10, weight='bold')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.title("MovieLens-100K", fontsize=11, weight='bold')
plt.legend(loc='lower right', frameon=True, fontsize=8, edgecolor='black')

plt.tight_layout()
plt.savefig("hr_vs_epoch_with_error.png", dpi=300)
plt.show()

plt.figure(figsize=(4, 3))
plt.errorbar(gmf["epoch"], gmf["NDCG_mean"], yerr=gmf["NDCG_std"], linestyle='--', color='blue', label='GMF', capsize=2)
plt.errorbar(mlp["epoch"], mlp["NDCG_mean"], yerr=mlp["NDCG_std"], linestyle='-', color='purple', label='MLP', capsize=2)
plt.errorbar(neumf["epoch"], neumf["NDCG_mean"], yerr=neumf["NDCG_std"], linestyle='-', color='red', label='NeuMF', capsize=2)

plt.xlabel("Iteration", fontsize=10, weight='bold')
plt.ylabel("NDCG@10", fontsize=10, weight='bold')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.title("MovieLens-100K", fontsize=11, weight='bold')
plt.legend(loc='lower right', frameon=True, fontsize=8, edgecolor='black')

plt.tight_layout()
plt.savefig("ndcg_vs_epoch_with_error.png", dpi=300)
plt.show()

plt.figure(figsize=(4, 3))
plt.errorbar(gmf["epoch"], gmf["Loss_mean"], yerr=gmf["Loss_std"], linestyle='--', color='blue', label='GMF', capsize=2)
plt.errorbar(mlp["epoch"], mlp["Loss_mean"], yerr=mlp["Loss_std"], linestyle='-', color='purple', label='MLP', capsize=2)
plt.errorbar(neumf["epoch"], neumf["Loss_mean"], yerr=neumf["Loss_std"], linestyle='-', color='red', label='NeuMF', capsize=2)

plt.xlabel("Iteration", fontsize=10, weight='bold')
plt.ylabel("Loss", fontsize=10, weight='bold')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.title("MovieLens-100K", fontsize=11, weight='bold')
plt.legend(loc='upper right', frameon=True, fontsize=8, edgecolor='black')

plt.tight_layout()
plt.savefig("loss_vs_epoch_with_error.png", dpi=300)
plt.show()


