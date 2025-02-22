#!/usr/bin/env python
import numpy as np
import pandas as pd
import bio_embeddings
import os

print("Starting the embedding process...")

# ---- set config param ----
embedder_name_tcr = "esm2"
#embedder_index_tcr = 1
#data_filename = "../data/raw/nettcr_train_swapped_peptide_ls_3_26_peptides_full_tcr_final_index.csv"  # 你的数据文件
data_filename = "train400_index.csv"
print(f"Configuration set: embedder={embedder_name_tcr}, file={data_filename}") #      index={embedder_index_tcr}, 

# ---- read data ----
print("Reading data from CSV...")
data = pd.read_csv(filepath_or_buffer=os.path.join('./', data_filename),
                   #index_col='original_index',
                   index_col='new_index',
                   dtype={'A1_start': np.ushort, 'A1_end': np.ushort,
                          'A2_start': np.ushort, 'A2_end': np.ushort,
                          'A3_start': np.ushort, 'A3_end': np.ushort,
                          'B1_start': np.ushort, 'B1_end': np.ushort,
                          'B2_start': np.ushort, 'B2_end': np.ushort,
                          'B3_start': np.ushort, 'B3_end': np.ushort},
                   usecols=['TRA_aa', 'TRB_aa',
                            'A1_start', 'A1_end', 'A2_start', 'A2_end',
                            'A3_start', 'A3_end', 'B1_start', 'B1_end',
                            'B2_start', 'B2_end', 'B3_start', 'B3_end',
                            'binder', 
                            #'original_index'
                            'new_index'
                            ])
print(f"Data loaded. Shape: {data.shape}")

# ---- get ESM Embedder ----
print("Initializing ESM embedder...")
embedder = bio_embeddings.get_bio_embedder(name=embedder_name_tcr)

# ----  binder = 1 data embedding ----
print("Filtering binder=1 data and performing embedding...")
count = 0
def embed_with_progress(seq):
    global count
    count += 1
    if count % 100 == 0:
        print(f"Embedded {count} sequences...")
    return embedder.embed(seq)

data = (data
        .query('binder == 1')
        #.assign(tra_aa_encoded=lambda x: x.TRA_aa.map(embedder.embed),
        #        trb_aa_encoded=lambda x: x.TRB_aa.map(embedder.embed)))
        .assign(tra_aa_encoded=lambda x: x.TRA_aa.map(embed_with_progress),
                trb_aa_encoded=lambda x: x.TRB_aa.map(embed_with_progress)))
print(f"Embedding completed. Remaining samples: {len(data)}")

# ---- Extract CDR seqs ----
print("Extracting CDR sequences...")
cdr_name_tuple = ('a1_encoded', 'a2_encoded', 'a3_encoded',
                  'b1_encoded', 'b2_encoded', 'b3_encoded')

tcr_name_tuple = ('tra_aa_encoded', 'tra_aa_encoded', 'tra_aa_encoded',
                  'trb_aa_encoded', 'trb_aa_encoded', 'trb_aa_encoded')

cdr_start_name_tuple = ('A1_start', 'A2_start', 'A3_start',
                        'B1_start', 'B2_start', 'B3_start')

cdr_end_name_tuple = ('A1_end', 'A2_end', 'A3_end',
                      'B1_end', 'B2_end', 'B3_end')

print("Processing CDR regions...")
for i in range(len(cdr_name_tuple)):
    cdr_name = cdr_name_tuple[i]
    tcr_name = tcr_name_tuple[i]
    cdr_start_name = cdr_start_name_tuple[i]
    cdr_end_name = cdr_end_name_tuple[i]

    print(f"Processing {cdr_name}...")
    data[cdr_name] = data.apply(lambda x: x[tcr_name][x[cdr_start_name]:x[cdr_end_name]], axis=1)

# ---- Only reserve CDR embedding and save ----
print("Filtering and saving results...")
#data = data.filter(items=cdr_name_tuple)
data = data.filter(items=list(cdr_name_tuple) + ['new_index'])

#data.to_pickle(path=f'./s01_et{embedder_index_tcr}_embedding.pkl')
data.to_pickle(path=f'./esm2_cdr123_embedding.pkl')

#print(f"Embedding completed, results saved as './s01_et{embedder_index_tcr}_embedding.pkl'")
print(f"Embedding completed, results saved as './esm2_cdr123_embedding.pkl'")
print("Process completed successfully!")





