import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    "Teacher\n[32,16] f6",
    "Student\n[16] f16 Response",
    "Student\n[16] f16 Relation",
    "Student\n[16] f8 Respsponse",
    "Student\n[16] f8 Relation",
    "Student\n[32,16] f8 Response",
    "Student\n[32,16] f8 Featture",
    "Student\n[32,16] f8 Relation",
]

hr_means = [
    0.6668,
    0.6817,
    0.66607,
    0.6713,
    0.6583,
    0.67689,
    0.65991,
    0.66013,
]

hr_stds = [
    0.0098,
    0.0043,
    0.0056,
    0.0050,
    0.0068,
    0.0054,
    0.0057,
    0.0059,
]

ndcg_means = [
    0.3862,
    0.3944,
    0.38476,
    0.3892,
    0.3780,
    0.38976,
    0.37715,
    0.37989,
]

ndcg_stds = [
    0.0057,
    0.0025,
    0.0045,
    0.0038,
    0.0060,
    0.0059,
    0.0051,
    0.0041,
]

x = np.arange(len(models))
width = 0.6

# Plot HR@10
plt.figure(figsize=(12, 6))
plt.bar(x, hr_means, yerr=hr_stds, capsize=5)
plt.axhline(y=0.6668, color='black', linestyle='--', linewidth=1, label='Teacher Baseline') 
plt.xticks(x, models, rotation=45, ha='right')
plt.ylabel("HR@10")
plt.title("Hit Ratio @10 Comparison")
plt.tight_layout()
plt.ylim(0, 0.8)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# Plot NDCG@10
plt.figure(figsize=(12, 6))
plt.bar(x, ndcg_means, yerr=ndcg_stds, capsize=5, color='orange')
plt.xticks(x, models, rotation=45, ha='right')
plt.ylabel("NDCG@10")
plt.title("NDCG @10 Comparison")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
