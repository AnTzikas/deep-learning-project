import pandas as pd
import matplotlib.pyplot as plt

# Raw data as a CSV string
data = """
Configuration,HR@10,NDCG@10
1 Layer | No Pretrain,0.6735 ± 0.0080,0.3925 ± 0.0079
1 Layer | With Pretrain,0.6720 ± 0.0104,0.3923 ± 0.0048
2 Layers | No Pretrain,0.6743 ± 0.0084,0.3857 ± 0.0052
2 Layers | With Pretrain,0.6717 ± 0.0085,0.3898 ± 0.0034
3 Layers | No Pretrain,0.6657 ± 0.0076,0.3859 ± 0.0051
3 Layers | With Pretrain,0.6654 ± 0.0064,0.3835 ± 0.0059
"""

# Load data into a DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data))

# Extract layer count and pretrain status
df[['Layers', 'Pretrain']] = df['Configuration'].str.extract(r'(\d+) Layer[s]* \| (.*)')
df['Layers'] = df['Layers'].astype(int)

# Helper to split mean ± std
def parse_metric(metric):
    value, std = metric.split('±')
    return float(value.strip()), float(std.strip())

# Parse HR@10 metrics and errors
df[['HR@10_mean', 'HR@10_std']] = df['HR@10'].apply(parse_metric).tolist()

# Create side-by-side plots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for pretrain_status, ax in zip(['No Pretrain', 'With Pretrain'], axs):
    subset = df[df['Pretrain'] == pretrain_status]
    ax.errorbar(subset['Layers'], subset['HR@10_mean'], yerr=subset['HR@10_std'],
                label='HR@10', marker='o', linestyle='-', capsize=5)
    ax.set_title(pretrain_status)
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('HR@10')
    ax.set_ylim(0.60, 0.70)
    ax.set_xticks(subset['Layers'])
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
