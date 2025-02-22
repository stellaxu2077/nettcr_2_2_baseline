import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
data_bl = pd.read_csv('m1_auc01.tsv', sep='\t')
data_esm2 = pd.read_csv('m2_auc01.tsv', sep='\t')
data_tcrbert = pd.read_csv('m3_auc01.tsv', sep='\t')
data_esm2_tcrbert = pd.read_csv('m4_auc01.tsv', sep='\t')

# 数据集合和对应标签
datasets = [data_bl, data_esm2, 
            data_tcrbert, 
            data_esm2_tcrbert]
labels = ['Baseline (BLOSUM)', 'ESM-2 (CDR1/2/3)', 
          'BLOSUM (CDR12) + TCR-BERT (CDR3)', 
          'ESM-2 (CDR1/2) + TCR-BERT (CDR3)']

# 按Peptide合并数据
merged_data = pd.concat([data_bl[['Peptide', 'Positive_Count']],
                         *[data[['AUC_0.1']] for data in datasets]], axis=1)
merged_data.columns = ['Peptide', 'Positive_Count', *labels]
merged_data['Positive_Count'] = merged_data['Positive_Count'].astype('Int64')

# 准备作图数据
x = np.arange(len(merged_data))  # 每个Peptide的X位置
width = 0.15  # 每个柱的宽度

# 创建画布
plt.figure(figsize=(14, 8))

# 绘制每组柱状图
for i, label in enumerate(labels):
    plt.bar(x + i * width, merged_data[label], width, label=label)

# 设置X轴标签（包含Peptide名称和Positive_Count）
xtick_labels = [f"{p} {c}" for p, c in zip(merged_data['Peptide'], merged_data['Positive_Count'])]
plt.xticks(x + width * (len(labels) - 1) / 2, xtick_labels, rotation=45, ha='right')

# 添加标题和标签
plt.xlabel('Peptide (Positive Count)')
plt.ylabel('AUC 0.1')
#plt.title('AUC 0.1 for Different Embeddings')
plt.ylim(0.45, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例
plt.legend()

# 保存并显示图表
plt.tight_layout()
plt.savefig('../plots/auc01_esm2_tcrbert_bar_chart.png')
plt.show()

