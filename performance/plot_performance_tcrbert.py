import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data_bl = pd.read_csv('m1_auc01.tsv', sep='\t')
data_tb = pd.read_csv('m3_auc01.tsv', sep='\t')

datasets = [data_bl, data_tb]
#labels = ['No weight', 'Weighted', 'Weighted p=2', 'ESM-2', 'ESM-2 short seq']
labels = ['Baseline (BLOSUM)', 'BLOSUM (CDR12) + TCR-BERT (CDR3)']

merged_data = pd.concat([data_bl[['Peptide', 'Positive_Count']],
                         *[data[['AUC_0.1']] for data in datasets]], axis=1)
merged_data.columns = ['Peptide', 'Positive_Count', *labels]
merged_data['Positive_Count'] = merged_data['Positive_Count'].astype('Int64')

x = np.arange(len(merged_data))  
width = 0.15  



plt.figure(figsize=(14, 8))

for i, label in enumerate(labels):
    plt.bar(x + i * width, merged_data[label], width, label=label)

xtick_labels = [f"{p} {c}" for p, c in zip(merged_data['Peptide'], merged_data['Positive_Count'])]
plt.xticks(x + width * (len(labels) - 1) / 2, xtick_labels, rotation=45, ha='right')

plt.xlabel('Peptide (Positive Count)')
plt.ylabel('AUC 0.1')
#plt.title('AUC 0.1 for Different Embeddings')
plt.ylim(0.45, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例
plt.legend()

# 保存并显示图表
plt.tight_layout()
plt.savefig('../plots/auc01_tcrbert_bar_chart.png')
plt.show()

