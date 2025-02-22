#!/usr/bin/env python
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel, AutoModel
import datasets
import os

# ✅ 直接定义 YAML 变量，替代外部配置文件
#embedder_index_cdr3 = "01"
embedder_name_cdr3 = "wukevin/tcr-bert-mlm-only"
embedder_source_cdr3 = "hugging_face"
embedder_batch_size_cdr3 = 6353
embedder_backend_cdr3 = "pt"
data_filename = "nettcr_train_swapped_peptide_ls_3_26_peptides_full_tcr_final.csv"  

# ✅ 读取数据
features = datasets.Features({
    'A3': datasets.Value('string'),
    'B3': datasets.Value('string'),
    'binder': datasets.Value('bool'),
    'original_index': datasets.Value('int32')
    #'new_index': datasets.Value('int32')
})


data = datasets.load_dataset(path='csv',
                             data_files=os.path.join('../data/raw', data_filename),
                             features=features)

# ✅ 定义 CDR3 长度计算
def get_cdr3_length(example):
    return {'a3_length': len(example['A3']), 'b3_length': len(example['B3'])}

# ✅ 分割氨基酸序列
def split_amino_acids(example):
    example['A3'] = ' '.join(example['A3'])
    example['B3'] = ' '.join(example['B3'])
    return example

# ✅ 加载 `transformers` 预训练模型
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=embedder_name_cdr3)

if embedder_backend_cdr3 == "tf":
    model = TFAutoModel.from_pretrained(pretrained_model_name_or_path=embedder_name_cdr3)
elif embedder_backend_cdr3 == "pt":
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=embedder_name_cdr3)

# ✅ 获取嵌入（embedding）
def add_embeddings(batch):
    embeddings_dict = {}
    batch_size = len(batch['A3'])
    print(f"Processing batch of {batch_size} sequences...")

    cdr3_name_tuple = ('A3', 'B3')
    cdr3_encoded_name_tuple = ('a3_encoded', 'b3_encoded')

    for i in range(len(cdr3_name_tuple)):
        cdr3_name = cdr3_name_tuple[i]
        cdr3_encoded_name = cdr3_encoded_name_tuple[i]
        
        print(f"Computing embeddings for {cdr3_name}...")
        outputs = model(**tokenizer(batch[cdr3_name], return_tensors=embedder_backend_cdr3, padding='longest'))
        embeddings_dict[cdr3_encoded_name] = outputs['last_hidden_state']
        print(f"Completed {cdr3_name} embeddings")

    return embeddings_dict

# Add progress tracking to add_embeddings function
def add_embeddings(batch):
    embeddings_dict = {}
    batch_size = len(batch['A3'])
    print(f"Processing batch of {batch_size} sequences...")

    cdr3_name_tuple = ('A3', 'B3')
    cdr3_encoded_name_tuple = ('a3_encoded', 'b3_encoded')

    for i in range(len(cdr3_name_tuple)):
        cdr3_name = cdr3_name_tuple[i]
        cdr3_encoded_name = cdr3_encoded_name_tuple[i]
        
        print(f"Computing embeddings for {cdr3_name}...")
        outputs = model(**tokenizer(batch[cdr3_name], return_tensors=embedder_backend_cdr3, padding='longest'))
        embeddings_dict[cdr3_encoded_name] = outputs['last_hidden_state']
        print(f"Completed {cdr3_name} embeddings")

    return embeddings_dict

# Add counter for filtering and mapping operations
counter = {'processed': 0}

def count_progress(example):
    counter['processed'] += 1
    if counter['processed'] % 100 == 0:
        print(f"Processed {counter['processed']} sequences...")
    return example

# ✅ 仅处理 `binder == True` 的数据
print("Filtering and processing data...")
data = (data
        .filter(lambda x: x['binder'] is True)
        .map(get_cdr3_length)
        .map(split_amino_acids)
        .map(count_progress)  # Add progress tracking
        .map(add_embeddings, batched=True, batch_size=embedder_batch_size_cdr3))

print("Converting to DataFrame...")
# ✅ 转换为 Pandas DataFrame
data.set_format(type='pandas', 
                columns=[
                'original_index', 
                'a3_encoded', 
                'b3_encoded', 
                'a3_length', 
                'b3_length'])

data_df = data['train'][:].set_index('original_index')

print("Processing embeddings...")
# Add progress tracking for embedding processing
total_rows = len(data_df)
processed = 0

# ✅ 确保 embedding 是 NumPy 数组
for idx in data_df.index:
    data_df.at[idx, 'a3_encoded'] = np.vstack(data_df.at[idx, 'a3_encoded'])
    data_df.at[idx, 'b3_encoded'] = np.vstack(data_df.at[idx, 'b3_encoded'])
    processed += 1
    if processed % 100 == 0:
        print(f"Processed embeddings for {processed}/{total_rows} sequences...")

print("Removing padding...")
# ✅ 去除 padding
data_df['a3_encoded'] = data_df.apply(lambda x: x['a3_encoded'][1:x['a3_length'] + 1], axis=1)
data_df['b3_encoded'] = data_df.apply(lambda x: x['b3_encoded'][1:x['b3_length'] + 1], axis=1)

print("Filtering columns...")
# ✅ 仅保留重要列
data_df = data_df[['a3_encoded', 'b3_encoded']]

print("Saving results...")
# ✅ 保存结果
experiment = "tcrbert_cdr3"

data_df.to_pickle(path=f"../data/{experiment}_embedding_orgid.pkl")
print("Process completed successfully!")





import pandas as pd

# 读取保存的pkl文件
pkl_path = f"../data/{experiment}_embedding_orgid.pkl"
data_loaded = pd.read_pickle(pkl_path)

# 打印前几行数据
print("First few rows of the loaded data:")
print(data_loaded.head())

# 检查列名，确保 new_index 存在
print("\nColumns in the loaded dataframe:")
print(data_loaded.columns)

# 检查索引是否为 new_index
print("\nIndex of the dataframe:")
print(data_loaded.index)
