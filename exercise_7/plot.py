import pandas as pd
import matplotlib.pyplot as plt

# Load the summary data
neumf = pd.read_csv("num_of_negatives_summary.csv")

# -------- HR@K Plot --------
plt.figure(figsize=(4.2, 3.2))
plt.errorbar(
    neumf["Num_of_negatives"],
    neumf["HR_mean"],
    yerr=neumf["HR_std"],
    fmt='o--',
    color='blue',
    capsize=3,
    label='NeuMF'
)

plt.xlabel("# negatives", fontsize=10, weight='bold')
plt.ylabel("HR@K", fontsize=10, weight='bold')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.ylim(0.60, 0.69)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.title("MovieLens-100K", fontsize=11, weight='bold')
plt.legend(loc='lower right', frameon=True, fontsize=8, edgecolor='black')
plt.tight_layout()
plt.savefig("Num_of_negatives_HR.png", dpi=300)
plt.show()

# -------- NDCG@K Plot --------
plt.figure(figsize=(4.2, 3.2))
plt.errorbar(
    neumf["Num_of_negatives"],
    neumf["NDCG_mean"],
    yerr=neumf["NDCG_std"],
    fmt='o--',
    color='red',
    capsize=3,
    label='NeuMF'
)

plt.xlabel("# negatives", fontsize=10, weight='bold')
plt.ylabel("NDCG@K", fontsize=10, weight='bold')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.ylim(0.35, 0.42) 
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.title("MovieLens-100K", fontsize=11, weight='bold')
plt.legend(loc='lower right', frameon=True, fontsize=8, edgecolor='black')
plt.tight_layout()
plt.savefig("Num_of_negatives_NDCG.png", dpi=300)
plt.show()
