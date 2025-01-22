# preprocess_embeddings.py

import os
import numpy as np
import pandas as pd
import torch
import esm
import argparse
import h5py

# 导入 keras_utils.py 中的函数和字典
import keras_utils

def get_tensor_name(row):
    """根据 DataFrame 行生成唯一的 tensor 文件名。"""
    #return str(row['original_index'])
    return str(row['new_index'])

def get_name_from_cdrs(row):
    """根据 DataFrame 中的 CDR 信息生成唯一的 TCR 名称。"""
    return keras_utils.get_name_from_cdrs(row)

def encode_and_save_hdf5(df, tensor_dir, hdf5_path, blosum, model_name="esm2_t33_650M_UR50D"):
    """
    对 DataFrame 中的每一行进行编码，并保存为 .npy 文件。
    
    - 对 peptide 使用 blosum 编码，
    - 对 TRA 和 TRB 使用 ESM2 编码，
    - 提取 CDR 区域的 embeddings，
    - 填充到 max_seq_lens，
    - 保存为 .npy 文件。
    """
    # 初始化 ESM2 模型
    print("Loading ESM2 model...")
    model, alphabet = getattr(esm.pretrained, f'{model_name}')()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    print("ESM2 model loaded.")

    # 定义 CDR 名称和对应的最大长度
    cdr_names = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']
    max_seq_lens = {'A1':7, 'A2':8, 'A3':22, 'B1':6, 'B2':7, 'B3':23}
    peptide_max_len = 12
    blosum_dim = len(next(iter(blosum.values())))  # e.g., 20

    total_samples = len(df)

    with h5py.File(hdf5_path, 'w') as hdf5_file:
        # 创建数据集
        for cdr in cdr_names:
            hdf5_file.create_dataset(cdr, shape=(total_samples, max_seq_lens[cdr], 1280), dtype='float32')
        hdf5_file.create_dataset('peptide', shape=(total_samples, peptide_max_len, blosum_dim), dtype='float32')
        
        # 创建标签和权重数据集
        hdf5_file.create_dataset('binder', data=df['binder'].values, dtype='int64')
        #hdf5_file.create_dataset('sample_weight', data=df['sample_weight'].values, dtype='float32')
        hdf5_file.create_dataset('partition', data=df['partition'].values, dtype='int64')

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            tensor_name = get_tensor_name(row)
            tcr_name = get_name_from_cdrs(row)
            
            # 编码 peptide 使用 BLOSUM
            peptide_seq = row_dict['peptide']
            peptide_encoding = keras_utils.enc_list_bl_max_len(
                [peptide_seq],
                blosum,
                max_seq_len=peptide_max_len,
                padding='right'
            )[0]  # Shape: (12, 20)
            
            # 编码 TRA_aa 和 TRB_aa 使用 ESM2
            sequences = [
                (tensor_name, row_dict['TRA_aa']),
                (tensor_name, row_dict['TRB_aa'])
            ]
            batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
            
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[model.num_layers])
            token_representations = results["representations"][model.num_layers].cpu().numpy()  # Shape: (2, seq_len, 1280)
            
            # 获取 TRA 和 TRB 的完整 embeddings
            tra_embedding_full = token_representations[0]  # (seq_len, 1280)
            trb_embedding_full = token_representations[1]  # (seq_len, 1280)
            
            # 提取并填充 CDR 区域
            embeddings = {}
            for cdr in cdr_names[:3]:  # TRA CDRs: A1, A2, A3
                start = int(row_dict.get(f'{cdr}_start', 0))
                end = int(row_dict.get(f'{cdr}_end', 0))
                max_len = max_seq_lens[cdr]
                if 0 <= start < end <= tra_embedding_full.shape[0]:
                    cdr_emb = tra_embedding_full[start:end, :]  # (current_len, 1280)
                    # Pad or truncate to max_len
                    if cdr_emb.shape[0] < max_len:
                        pad_len = max_len - cdr_emb.shape[0]
                        pad_block = np.full((pad_len, cdr_emb.shape[1]), -5, dtype=np.float32)
                        cdr_emb_padded = np.concatenate([cdr_emb, pad_block], axis=0)
                    else:
                        cdr_emb_padded = cdr_emb[:max_len, :]
                    embeddings[cdr] = cdr_emb_padded  # (max_len, 1280)
                else:
                    # Invalid indices, pad with -5
                    embeddings[cdr] = np.full((max_len, tra_embedding_full.shape[1]), -5, dtype=np.float32)
            
            for cdr in cdr_names[3:]:  # TRB CDRs: B1, B2, B3
                start = int(row_dict.get(f'{cdr}_start', 0))
                end = int(row_dict.get(f'{cdr}_end', 0))
                max_len = max_seq_lens[cdr]
                if 0 <= start < end <= trb_embedding_full.shape[0]:
                    cdr_emb = trb_embedding_full[start:end, :]  # (current_len, 1280)
                    # Pad or truncate to max_len
                    if cdr_emb.shape[0] < max_len:
                        pad_len = max_len - cdr_emb.shape[0]
                        pad_block = np.full((pad_len, cdr_emb.shape[1]), -5, dtype=np.float32)
                        cdr_emb_padded = np.concatenate([cdr_emb, pad_block], axis=0)
                    else:
                        cdr_emb_padded = cdr_emb[:max_len, :]
                    embeddings[cdr] = cdr_emb_padded  # (max_len, 1280)
                else:
                    # Invalid indices, pad with -5
                    embeddings[cdr] = np.full((max_len, trb_embedding_full.shape[1]), -5, dtype=np.float32)
            
            # 将 peptide 的 BLOSUM 编码添加到 embeddings
            embeddings['peptide'] = peptide_encoding  # (12, 20)
            
            # 保存为 .npy 文件
            '''
            esm_embedding_dir = os.path.join(tensor_dir, "esm_embeddings_train")
            os.makedirs(esm_embedding_dir, exist_ok=True)
            np.save(os.path.join(esm_embedding_dir, f"{tensor_name}.npy"), embeddings)
            '''
            for cdr in cdr_names:
                hdf5_file[cdr][idx] = embeddings[cdr]
            hdf5_file['peptide'][idx] = embeddings['peptide']

            # 打印进度
            if (idx + 1) % 100 == 0 or (idx + 1) == total_samples:
                print(f"Processed {idx + 1} / {total_samples} samples.")

def main():
    parser = argparse.ArgumentParser(description='Precompute and save ESM2 embeddings for TRA and TRB sequences with CDR extractions, and BLOSUM encoding for peptides.')
    parser.add_argument('-trf', '--train_file', required=True, type=str, help='Path to the training CSV file')
    parser.add_argument('-out', '--out_dir', required=True, type=str, help='Output directory to save HDF5 file')
    parser.add_argument('-tensordir', '--tensor_dir', required=True, type=str, help='Directory holding ESM-2 tensors')
    parser.add_argument('-h5', '--hdf5_path', required=True, type=str, help='Path to save the HDF5 file')
    args = parser.parse_args()
    
    # 读取数据
    print("Reading training data...")
    data = pd.read_csv(args.train_file)
    print(f"Total samples: {len(data)}")
    
    # 确保 `original_index` 列存在
    '''
    if 'original_index' not in data.columns:
        print("`original_index` column not found. Resetting index and renaming to `original_index`.")
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'original_index'}, inplace=True)
    '''
    if 'new_index' not in data.columns:
        print("`new_index` column not found. Resetting index and renaming to `new_index`.")
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'new_index'}, inplace=True)


    # 定义 BLOSUM 字典
    blosum = keras_utils.blosum62_20aa  # 从 keras_utils.py 中获取
    
    # 编码并保存
    print("Starting encoding and saving...")
    encode_and_save_hdf5(data, args.tensor_dir, args.hdf5_path, blosum, model_name="esm2_t33_650M_UR50D")
    print("Encoding and saving completed.")

if __name__ == "__main__":
    main()

