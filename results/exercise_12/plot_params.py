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

parameters = [
    84561,
    63033,
    63033,
    42025,
    42025,
    63533,
    63533,
    63533
]


x = np.arange(len(models))
width = 0.6

plt.figure(figsize=(12, 6))
bars = plt.bar(x, parameters, color='orange', width=width)
plt.axhline(y=parameters[0], color='black', linestyle='--', linewidth=1, label='Teacher Baseline') 

plt.xticks(x, models, rotation=45, ha='right')
plt.ylabel("# of Parameters")
plt.title("Parameters Comparison")
plt.tight_layout()
# plt.ylim(0.37, 0.40)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Params_comparison.png", dpi=300)
plt.show()