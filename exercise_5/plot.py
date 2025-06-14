import pandas as pd
import matplotlib.pyplot as plt

# Load the summary data
neumf = pd.read_csv("topK_summary.csv")


# Plot
plt.figure(figsize=(4, 3))
plt.errorbar(neumf["K"], neumf["HR_mean"], yerr=neumf["HR_std"], linestyle='--', color='blue', label='NeuMF', capsize=2)

plt.xlabel("TopK", fontsize=10, weight='bold')
plt.ylabel("HR@K", fontsize=10, weight='bold')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.title("MovieLens-100K", fontsize=11, weight='bold')
plt.legend(loc='lower right', frameon=True, fontsize=8, edgecolor='black')

plt.tight_layout()
plt.savefig("TopK_HR.png", dpi=300)
plt.show()



plt.figure(figsize=(4, 3))
plt.errorbar(neumf["K"], neumf["NDCG_mean"], yerr=neumf["NDCG_std"], linestyle='--', color='red', label='NeuMF', capsize=2)

plt.xlabel("TopK", fontsize=10, weight='bold')
plt.ylabel("NDCG@K", fontsize=10, weight='bold')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.title("MovieLens-100K", fontsize=11, weight='bold')
plt.legend(loc='lower right', frameon=True, fontsize=8, edgecolor='black')

plt.tight_layout()
plt.savefig("TopK_NDCG.png", dpi=300)
plt.show()
