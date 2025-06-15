import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    "Teacher\n[32,16] f6",
    "Student\n[16] f16 Resp",
    "Student\n[16] f16 Rel",
    "Student\n[16] f8 Resp",
    "Student\n[16] f8 Rel",
    "Student\n[32,16] f8 Resp",
    "Student\n[32,16] f8 Feat",
    "Student\n[32,16] f8 Rel",
]

hr_means = [0.6668, 0.6817, 0.66607, 0.6713, 0.6683, 0.67689, 0.66991, 0.66013]
hr_stds  = [0.0098,  0.0043,  0.0056,   0.0050,  0.0068,  0.0054,   0.0057,   0.0059]

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
colors = plt.get_cmap("tab10").colors
markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']

plt.figure(figsize=(12, 6))
for i in range(len(models)):
    plt.errorbar(x[i], hr_means[i], yerr=hr_stds[i],
                 fmt=markers[i], color=colors[i % len(colors)],
                 capsize=4, elinewidth=1.2, markersize=9, label=models[i])

# Draw baseline for teacher
plt.axhline(y=hr_means[0], linestyle='--', color='black', linewidth=1.2, label='Teacher Baseline')

# Labels and layout
plt.xticks(x, models, rotation=45, ha='right')
plt.ylabel("HR@10", fontsize=12)
plt.title("HR@10 Comparison (Mean ± Std)", fontsize=13, weight='bold')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.ylim(0.64, 0.70)
plt.tight_layout()
plt.legend(fontsize=9)
plt.show()

x = np.arange(len(models))
colors = plt.get_cmap("tab10").colors
markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']

plt.figure(figsize=(12, 6))
for i in range(len(models)):
    plt.errorbar(x[i], ndcg_means[i], yerr=ndcg_stds[i],
                 fmt=markers[i], color=colors[i % len(colors)],
                 capsize=4, elinewidth=1.2, markersize=9, label=models[i])

# Draw baseline for teacher
plt.axhline(y=ndcg_means[0], linestyle='--', color='black', linewidth=1.2, label='Teacher Baseline')

# Labels and layout
plt.xticks(x, models, rotation=45, ha='right')
plt.ylabel("NDCG@10", fontsize=12)
plt.title("NDCG@10 Comparison (Mean ± Std)", fontsize=13, weight='bold')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.ylim(0.35, 0.42)
plt.tight_layout()
plt.legend(fontsize=9)
plt.show()