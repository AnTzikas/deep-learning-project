import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    "Teacher\n[32,16] f6",
    "Student\n[16] f16 Response",
    "Student\n[16] f16 Relation",
    "Student\n[16] f8 Response",
    "Student\n[16] f8 Relation",
    "Student\n[32,16] f8 Response",
    "Student\n[32,16] f8 Feature",
    "Student\n[32,16] f8 Relation",
]


hr_means = [
    0.6668,
    0.6817,
    0.66607,
    0.6713,
    0.6683,#0.6583
    0.67689,
    0.66991, #0.65991
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

# x = np.arange(len(models))
plt.figure(figsize=(12, 6))
plt.errorbar(x, hr_means, yerr=hr_stds, fmt='o', capsize=5, elinewidth=1.5, marker='s', markersize=8, color='blue')
plt.axhline(y=hr_means[0], linestyle='--', color='black', label='Teacher Baseline')
plt.xticks(x, models, rotation=45, ha='right')
plt.ylabel("HR@10")
plt.title("HR@10 Comparison (Mean ± Std)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.legend()
plt.savefig("Hr_comparison_zoomed.png", dpi=300)
plt.show()



# Plot HR@10 with text std
plt.figure(figsize=(12, 6))
plt.axhline(y=hr_means[0], color='black', linestyle='--', linewidth=1, label='Teacher Baseline') 
bars = plt.bar(x, hr_means, color='cornflowerblue', width=width)
plt.xticks(x, models, rotation=45, ha='right')
plt.ylabel("HR@10")
plt.title("Hit Ratio @10 Comparison")
plt.tight_layout()
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
# for i, (mean, std) in enumerate(zip(hr_means, hr_stds)):
#     plt.hlines([mean - std, mean + std], i - 0.2, i + 0.2, colors='black', linestyles='dotted', linewidth=1)
# for i, (mean, std) in enumerate(zip(hr_means, hr_stds)):
#     plt.fill_between(
#         [i - 0.25, i + 0.25],  # X range of the bar
#         [mean - std, mean - std],
#         [mean + std, mean + std],
#         color='black',
#         alpha=0.2,
#         linestyle='dotted'
#     )

# Annotate std values
for i, bar in enumerate(bars):
    height = bar.get_height()
    std_text = f"±{hr_stds[i]:.3f}"
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.001, std_text, ha='center', va='bottom', fontsize=9)
plt.savefig("Hr_comparison_simple.png", dpi=300)
plt.show()

# Plot NDCG@10 with text std
plt.figure(figsize=(12, 6))
plt.axhline(y=ndcg_means[0], color='black', linestyle='--', linewidth=1, label='Teacher Baseline')
bars = plt.bar(x, ndcg_means, color='red', width=width)
plt.xticks(x, models, rotation=45, ha='right')
plt.ylabel("NDCG@10")
plt.title("NDCG @10 Comparison")
plt.tight_layout()
plt.ylim(0,1)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Annotate std values
for i, bar in enumerate(bars):
    height = bar.get_height()
    std_text = f"±{ndcg_stds[i]:.3f}"
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.001, std_text, ha='center', va='bottom', fontsize=9)
plt.savefig("Ndcg_comparison.png", dpi=300)
plt.show()

