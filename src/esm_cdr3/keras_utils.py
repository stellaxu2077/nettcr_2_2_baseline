####!/usr/bin/env python

"""
Functions for data IO for neural network training.
"""

from __future__ import print_function
import argparse
import sys
import os
import time
import matplotlib.pyplot as plt
from operator import add
import math
import numpy as np
import pandas as pd
import torch

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParam import ProtParamData
from sklearn.metrics.pairwise import cosine_similarity

def mkdir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

def enc_list_bl_max_len(aa_seqs, blosum, max_seq_len, padding = "right"):
    '''
    blosum encoding of a list of amino acid sequences with padding
    to a max length

    parameters:
        - aa_seqs : list with AA sequences
        - blosum : dictionnary: key= AA, value= blosum encoding
        - max_seq_len: common length for padding
    returns:
        - enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    '''

    # encode sequences:
    sequences=[]
    for seq in aa_seqs:
        e_seq= np.zeros((len(seq),len(blosum["A"])))
        count=0
        for aa in seq:
            if aa in blosum:
                e_seq[count]=blosum[aa]
                count+=1
            else:
                sys.stderr.write("Unknown amino acid in peptides: "+ aa +", encoding aborted!\n")
                sys.exit(2)

        sequences.append(e_seq)

    # pad sequences:
    #max_seq_len = max([len(x) for x in aa_seqs])
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = -5*np.ones((n_seqs, max_seq_len, n_features))
    if padding == "right":
        for i in range(0,n_seqs):
            enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]

    elif padding == "left":
        for i in range(0,n_seqs):
            enc_aa_seq[i, max_seq_len-sequences[i].shape[0]:max_seq_len, :n_features] = sequences[i]

    else:
        print("Error: No valid padding has been chosen.\nValid options: 'right', 'left'")


    return enc_aa_seq


def get_name_from_cdrs(df: pd.DataFrame) -> list:
    return (df["A1"]
        + "_"
        + df["A2"]
        + "_"
        + df["A3"]
        + "_"
        + df["B1"]
        + "_"
        + df["B2"]
        + "_"
        + df["B3"]
    )


def pad_sequence(tensor, max_len, padding="right"):
    """
    Pad a tensor to the given maximum length.

    Parameters:
        - tensor: Torch tensor to pad.
        - max_len: Maximum length to pad to.
        - padding: Padding direction, "right" or "left".

    Returns:
        - padded_tensor: Padded tensor.
    """
    seq_len, embedding_dim = tensor.shape
    if seq_len > max_len:
        print("Error: sequence is longer than provided max_len")
        sys.exit(1)

    if seq_len == max_len: # sequence is already at max length, no need to pad
        return tensor

    # Create padded tensor
    pad_size = max_len - seq_len
    if padding == "right":
        padded_tensor = torch.cat([tensor, torch.zeros((pad_size, embedding_dim))], dim=0)
    elif padding == "left":
        padded_tensor = torch.cat([torch.zeros((pad_size, embedding_dim)), tensor], dim=0)
    else:
        raise ValueError("Padding must be 'right' or 'left'")

    return padded_tensor

def enc_list_esm2(df, tensor_dir, seq_names=['peptide', 'A1','A2','A3','B1','B2','B3'], max_seq_lens=[12,7,8,22,6,7,23], padding = "right"):
    '''
    esm2 encoding of a list of amino acid sequences 

    parameters:
        - df : dataframe holding the sequence data
        - seq_names: list of column names to encode (peptide and/or CDR names)
        - tensor_dir: full path to directory containing the tensors

    returns:
        - enc_aa_seq : list of lists, each list has np.ndarrays with encoded amino acid sequences
    '''


    sequences = [[] for _ in range(len(seq_names))]

    # encode sequences:

    for idx, row in df.iterrows():

        row = row.to_dict()
        tcr_name = get_name_from_cdrs(row) 
        
        # Tensors are pre-stored, padded
        #tensors = np.load(tensor_dir + "/esm_embeddings_train/{}.npy".format(tcr_name), allow_pickle=True).item()
        #tensors['peptide'] = np.load(tensor_dir + "/esm_embeddings_train/{}.npy".format(row['peptide']), allow_pickle=True)
        tensor_path = os.path.join(tensor_dir, "esm_embeddings_train", f"{row['original_index']}.npy")
        tensors = np.load(tensor_path, allow_pickle=True).item() 
        
        for i, name in enumerate(seq_names):
            sequences[i].append(tensors[name])
        
    return [np.array(l) for l in sequences]

def enc_list_esm2_opt(df, tensor_dir, seq_names=['peptide', 'A1','A2','A3','B1','B2','B3'], max_seq_lens=[12,7,8,22,6,7,23], padding="right"):
    '''
    esm2 encoding of a list of amino acid sequences with optimized batch loading of tensors (direct processing).

    parameters:
        - df : dataframe holding the sequence data
        - seq_names: list of column names to encode (peptide and/or CDR names)
        - tensor_dir: full path to directory containing the tensors

    returns:
        - enc_aa_seq : list of lists, each list has np.ndarrays with encoded amino acid sequences
    '''

    # Initialize sequences for all seq_names
    sequences = [[] for _ in range(len(seq_names))]

    # Keep track of already loaded tensors to avoid reloading
    loaded_tensors = {}

    # Process each row in the dataframe
    for idx, row in df.iterrows():
        row = row.to_dict()
        tcr_name = get_name_from_cdrs(row)
        peptide_name = row['peptide']

        # Load TCR tensor if not already loaded
        if tcr_name not in loaded_tensors:
            tcr_path = f"{tensor_dir}/esm_embeddings_train/{tcr_name}.npy"
            loaded_tensors[tcr_name] = np.load(tcr_path, allow_pickle=True).item()

        # Load peptide tensor if not already loaded
        if peptide_name not in loaded_tensors:
            peptide_path = f"{tensor_dir}/esm_embeddings_train/{peptide_name}.npy"
            loaded_tensors[peptide_name] = np.load(peptide_path, allow_pickle=True)

        # Combine TCR and peptide tensors for this row
        tensors = loaded_tensors[tcr_name]
        tensors['peptide'] = loaded_tensors[peptide_name]

        # Extract and append tensors for each sequence name
        for i, name in enumerate(seq_names):
            sequences[i].append(tensors[name])

    # Convert lists to numpy arrays for batch processing
    return [np.array(l) for l in sequences]



def construct_one_hot(aa_seqs):
    '''
    makeshift function for constructing the right format for one-hot encoding
    '''

    # encode sequences:
    sequences=[]
    for seq in aa_seqs:
        if seq == "paired":
            e_seq = np.array([1,1])
        elif seq == "alpha":
            e_seq = np.array([1,0])
        elif seq == "beta":
            e_seq = np.array([0,1])

        sequences.append(e_seq)

    # pad sequences:
    #max_seq_len = max([len(x) for x in aa_seqs])
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[0]

    enc_aa_seq = -5*np.ones((n_seqs, n_features))

    for i in range(0,n_seqs):
        enc_aa_seq[i, :n_features] = sequences[i]



    return enc_aa_seq

##############################
# Different encoding schemes #
##############################


mhc_one_hot = {
    'HLA-A*01:01': np.array((1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-A*02:01': np.array((0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-A*03:01': np.array((0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-A*11:01': np.array((0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-A*24:02': np.array((0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-A*30:02': np.array((0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-A*68:01': np.array((0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*07:02': np.array((0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*08:01': np.array((0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*15:01': np.array((0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*18:01': np.array((0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*27:01': np.array((0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*27:05': np.array((0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*35:01': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*35:08': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*35:42': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*37:01': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*40:01': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*42:01': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*44:02': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*44:03': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*51:01': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*51:193': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0)).reshape(1,29),
    'HLA-B*53:01': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0)).reshape(1,29),
    'HLA-B*57:01': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0)).reshape(1,29),
    'HLA-B*58:01': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0)).reshape(1,29),
    'HLA-B*81:01': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0)).reshape(1,29),
    'HLA-C*03:04': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0)).reshape(1,29),
    'HLA-C*07:02': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1)).reshape(1,29),
    'missing': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).reshape(1,29)
}
"""
mhc_one_hot = {
    'HLA-A*01:01': [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-A*02:01': [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-A*03:01': [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-A*11:01': [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-A*24:02': [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-A*30:02': [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-A*68:01': [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*07:02': [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*08:01': [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*15:01': [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*18:01': [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*27:01': [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*27:05': [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*35:01': [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*35:08': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*35:42': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*37:01': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*40:01': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*42:01': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    'HLA-B*44:02': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    'HLA-B*44:03': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    'HLA-B*51:01': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    'HLA-B*51:193': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    'HLA-B*53:01': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    'HLA-B*57:01': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    'HLA-B*58:01': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    'HLA-B*81:01': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    'HLA-C*03:04': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    'HLA-C*07:02': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    'missing': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
}
"""
blosum50_20aa = {
        'A': np.array((5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0)),
        'R': np.array((-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3)),
        'N': np.array((-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3)),
        'D': np.array((-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4)),
        'C': np.array((-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1)),
        'Q': np.array((-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3)),
        'E': np.array((-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3)),
        'G': np.array((0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4)),
        'H': np.array((-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4)),
        'I': np.array((-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4)),
        'L': np.array((-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1)),
        'K': np.array((-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3)),
        'M': np.array((-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1)),
        'F': np.array((-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1)),
        'P': np.array((-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3)),
        'S': np.array((1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2)),
        'T': np.array((0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0)),
        'W': np.array((-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3)),
        'Y': np.array((-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1)),
        'V': np.array((0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5))
    }

blosum50_20aa_masking = {
        'A': np.array((5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0)),
        'R': np.array((-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3)),
        'N': np.array((-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3)),
        'D': np.array((-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4)),
        'C': np.array((-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1)),
        'Q': np.array((-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3)),
        'E': np.array((-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3)),
        'G': np.array((0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4)),
        'H': np.array((-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4)),
        'I': np.array((-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4)),
        'L': np.array((-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1)),
        'K': np.array((-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3)),
        'M': np.array((-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1)),
        'F': np.array((-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1)),
        'P': np.array((-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3)),
        'S': np.array((1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2)),
        'T': np.array((0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0)),
        'W': np.array((-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3)),
        'Y': np.array((-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1)),
        'V': np.array((0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5)),
        'X': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))
    }

blosum50_20aa_negative_masking = {
        'A': np.array((5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0)),
        'R': np.array((-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3)),
        'N': np.array((-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3)),
        'D': np.array((-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4)),
        'C': np.array((-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1)),
        'Q': np.array((-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3)),
        'E': np.array((-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3)),
        'G': np.array((0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4)),
        'H': np.array((-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4)),
        'I': np.array((-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4)),
        'L': np.array((-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1)),
        'K': np.array((-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3)),
        'M': np.array((-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1)),
        'F': np.array((-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1)),
        'P': np.array((-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3)),
        'S': np.array((1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2)),
        'T': np.array((0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0)),
        'W': np.array((-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3)),
        'Y': np.array((-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1)),
        'V': np.array((0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5)),
        'X': np.array((-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5))
    }

blosum50 = {
    'A': np.array(( 5,-2,-1,-2,-1,-1,-1, 0,-2,-1,-2,-1,-1,-3,-1, 1, 0,-3,-2, 0,-2,-1,-1,-5)),
    'R': np.array((-2, 7,-1,-2,-4, 1, 0,-3, 0,-4,-3, 3,-2,-3,-3,-1,-1,-3,-1,-3,-1, 0,-1,-5)),
    'N': np.array((-1,-1, 7, 2,-2, 0, 0, 0, 1,-3,-4, 0,-2,-4,-2, 1, 0,-4,-2,-3, 4, 0,-1,-5)),
    'D': np.array((-2,-2, 2, 8,-4, 0, 2,-1,-1,-4,-4,-1,-4,-5,-1, 0,-1,-5,-3,-4, 5, 1,-1,-5)),
    'C': np.array((-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1,-3,-3,-2,-5)),
    'Q': np.array((-1, 1, 0, 0,-3, 7, 2,-2, 1,-3,-2, 2, 0,-4,-1, 0,-1,-1,-1,-3, 0, 4,-1,-5)),
    'E': np.array((-1, 0, 0, 2,-3, 2, 6,-3, 0,-4,-3, 1,-2,-3,-1,-1,-1,-3,-2,-3, 1, 5,-1,-5)),
    'G': np.array(( 0,-3, 0,-1,-3,-2,-3, 8,-2,-4,-4,-2,-3,-4,-2, 0,-2,-3,-3,-4,-1,-2,-2,-5)),
    'H': np.array((-2, 0, 1,-1,-3, 1, 0,-2,10,-4,-3, 0,-1,-1,-2,-1,-2,-3, 2,-4, 0, 0,-1,-5)),
    'I': np.array((-1,-4,-3,-4,-2,-3,-4,-4,-4, 5, 2,-3, 2, 0,-3,-3,-1,-3,-1, 4,-4,-3,-1,-5)),
    'L': np.array((-2,-3,-4,-4,-2,-2,-3,-4,-3, 2, 5,-3, 3, 1,-4,-3,-1,-2,-1, 1,-4,-3,-1,-5)),
    'K': np.array((-1, 3, 0,-1,-3, 2, 1,-2, 0,-3,-3, 6,-2,-4,-1, 0,-1,-3,-2,-3, 0, 1,-1,-5)),
    'M': np.array((-1,-2,-2,-4,-2, 0,-2,-3,-1, 2, 3,-2, 7, 0,-3,-2,-1,-1, 0, 1,-3,-1,-1,-5)),
    'F': np.array((-3,-3,-4,-5,-2,-4,-3,-4,-1, 0, 1,-4, 0, 8,-4,-3,-2, 1, 4,-1,-4,-4,-2,-5)),
    'P': np.array((-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3,-2,-1,-2,-5)),
    'S': np.array(( 1,-1, 1, 0,-1, 0,-1, 0,-1,-3,-3, 0,-2,-3,-1, 5, 2,-4,-2,-2, 0, 0,-1,-5)),
    'T': np.array(( 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 2, 5,-3,-2, 0, 0,-1, 0,-5)),
    'W': np.array((-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1, 1,-4,-4,-3,15, 2,-3,-5,-2,-3,-5)),
    'Y': np.array((-2,-1,-2,-3,-3,-1,-2,-3, 2,-1,-1,-2, 0, 4,-3,-2,-2, 2, 8,-1,-3,-2,-1,-5)),
    'V': np.array(( 0,-3,-3,-4,-1,-3,-3,-4,-4, 4, 1,-3, 1,-1,-3,-2, 0,-3,-1, 5,-4,-3,-1,-5)),
    'B': np.array((-2,-1, 4, 5,-3, 0, 1,-1, 0,-4,-4, 0,-3,-4,-2, 0, 0,-5,-3,-4, 5, 2,-1,-5)),
    'Z': np.array((-1, 0, 0, 1,-3, 4, 5,-2, 0,-3,-3, 1,-1,-4,-1, 0,-1,-2,-2,-3, 2, 5,-1,-5)),
    'X': np.array((-1,-1,-1,-1,-2,-1,-1,-2,-1,-1,-1,-1,-1,-2,-2,-1, 0,-3,-1,-1,-1,-1,-1,-5)),
    '*': np.array((-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5, 1))
}

one_hot = {
        'A': np.array((1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'R': np.array((0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'N': np.array((0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'D': np.array((0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'C': np.array((0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'Q': np.array((0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'E': np.array((0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'G': np.array((0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'H': np.array((0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0)),
        'I': np.array((0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0)),
        'L': np.array((0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0)),
        'K': np.array((0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0)),
        'M': np.array((0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0)),
        'F': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0)),
        'P': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0)),
        'S': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0)),
        'T': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0)),
        'W': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0)),
        'Y': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0)),
        'V': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0)),
        'X': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1))
    }

one_hot_20aa = {
        'A': np.array((1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'R': np.array((0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'N': np.array((0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'D': np.array((0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'C': np.array((0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'Q': np.array((0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'E': np.array((0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        'G': np.array((0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0)),
        'H': np.array((0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0)),
        'I': np.array((0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0)),
        'L': np.array((0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0)),
        'K': np.array((0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0)),
        'M': np.array((0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0)),
        'F': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0)),
        'P': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0)),
        'S': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0)),
        'T': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0)),
        'W': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0)),
        'Y': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0)),
        'V': np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1))
    }


amino_to_idx={
        'A': np.array((1,)),
        'R': np.array((2,)),
        'N': np.array((3,)),
        'D': np.array((4,)),
        'C': np.array((5,)),
        'Q': np.array((6,)),
        'E': np.array((7,)),
        'G': np.array((8,)),
        'H': np.array((9,)),
        'I': np.array((10,)),
        'L': np.array((11,)),
        'K': np.array((12,)),
        'M': np.array((13,)),
        'F': np.array((14,)),
        'P': np.array((15,)),
        'S': np.array((16,)),
        'T': np.array((17,)),
        'W': np.array((18,)),
        'Y': np.array((19,)),
        'V': np.array((20,)),
        'X': np.array((21,))
    }

phys_chem = {
        'A': np.array((1, -6.7, 0, 0, 0)),
        'R': np.array((4, -11.7, 0, 0, 0)),
        'N': np.array((6.13, 51.5, 4, 0, 1)),
        'D': np.array((4.77, 36.8, 2, 0, 1)),
        'C': np.array((2.95, 20.1, 2, 2, 0)),
        'Q': np.array((4.43, -14.4, 0, 0, 0)),
        'E': np.array((2.78, 38.5, 1, 4, -1)),
        'G': np.array((5.89, -15.5, 0, 0, 0)),
        'H': np.array((2.43, -8.4, 0, 0, 0)),
        'I': np.array((2.72, 0.8, 0, 0, 0)),
        'L': np.array((3.95, 17.2, 2, 2, 0)),
        'K': np.array((1.6, -2.5, 1,  2, 0)),
        'M': np.array((3.78, 34.3, 1, 4, -1)),
        'F': np.array((2.6, -5, 1, 2, 0)),
        'P': np.array((0, -4.2, 0, 0,0)),
        'S': np.array((8.08, -7.9, 1, 0,0)),
        'T': np.array((4.66, 12.6, 1, 1, 0)),
        'W': np.array((6.47, 2.9, 1, 1, 0)),
        'Y': np.array((4, -13, 0, 0,0)),
        'V': np.array((3, -10.9, 0, 0,0))
    }

blosum62 = {
        'A': np.array((4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2,  0,  0)),
        'R': np.array((-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1)),
        'N': np.array((-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  -1)),
        'D': np.array((-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  -1)),
        'C': np.array((0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -2)),
        'Q': np.array((-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  -1)),
        'E': np.array((-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  -1)),
        'G': np.array((0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1)),
        'H': np.array((-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  -1)),
        'I': np.array((-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -1)),
        'L': np.array((-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -1)),
        'K': np.array((-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -1)),
        'M': np.array((-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -1)),
        'F': np.array((-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -1)),
        'P': np.array((-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2)),
        'S': np.array((1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,   0)),
        'T': np.array((0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0,  0)),
        'W': np.array((-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -2)),
        'Y': np.array((-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -1)),
        'V': np.array((0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -1)),
        'X': np.array((0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1))
        }

blosum62_20aa = {
        'A': np.array((4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2,  0)),
        'R': np.array((-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3)),
        'N': np.array((-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3)),
        'D': np.array((-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3)),
        'C': np.array((0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1)),
        'Q': np.array((-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2)),
        'E': np.array((-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2)),
        'G': np.array((0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3)),
        'H': np.array((-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3)),
        'I': np.array((-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3)),
        'L': np.array((-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1)),
        'K': np.array((-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1)),
        'M': np.array((-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1)),
        'F': np.array((-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1)),
        'P': np.array((-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2)),
        'S': np.array((1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2)),
        'T': np.array((0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0)),
        'W': np.array((-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3)),
        'Y': np.array((-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1)),
        'V': np.array((0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4)),
        }


def adjust_batch_size(obs, batch_size, threshold = 0.5):
    if obs/batch_size < threshold:
        pass

    else:
        if (obs/batch_size % 1) >= threshold:
            pass
        else:
            while (obs/batch_size % 1) < threshold and (obs/batch_size % 1) != 0:
                batch_size += 1
    return batch_size


# Function to calculate the peptide properties
def pep_properties(seq):

    # Individual lists for each property
    masses = []
    ips = []
    hphos = []
    hphis = []

    for aa in seq:
        X = ProteinAnalysis(aa)
        masses.append(X.molecular_weight())
        ips.append(X.isoelectric_point())
        hphos.append(ProtParamData.kd[aa])   # Kyte-Doolittle hydrophobicity index
        hphis.append(ProtParamData.hw[aa])   # Hopp-Woods hydrophilicity index

    return np.array(masses), np.array(ips), np.array(hphos), np.array(hphis)

# Function to calculate pairwise absolute differences
def pairwise_diff(CDR, pep, CDR_len, pep_len = 12):

    # Vector of max-lengths shape to store the pairwise absolute differences
    differences = np.zeros((CDR_len, pep_len))
    for i in range(len(CDR)):
        for j in range(len(pep)):
            differences[i, j] = np.abs(CDR[i] - pep[j])

    # Scale differences matrix between 0 and 1 (min-max normalization)
    scaled_differences = (differences - np.min(differences)) / (np.max(differences) - np.min(differences))

    return scaled_differences

# Function to create the interaction maps from two amino acid sequences
def interaction_maps(data, encoding, seq1_name, seq1_len, seq2_name='peptide', seq2_len=12, properties_to_calculate=None):

    # Valid properties
    valid_properties = ['mass', 'ip', 'hpho', 'hphi', 'blosum']

    # Check if properties_to_calculate is provided
    if properties_to_calculate is not None:
        # Check if all specified properties are valid
        for prop in properties_to_calculate:
            assert prop in valid_properties, f"Invalid property '{prop}'. Valid properties are: {', '.join(valid_properties)}."

    # Calculate the interaction map final tensor according to the properties specified
    if properties_to_calculate is not None:

        # Initialize an empty tensor to store the pairwise absolute differences
        pairwise_diff_tensor = np.zeros((len(data), seq1_len, seq2_len, len(properties_to_calculate)))

        # Iterate over each row of the DataFrame
        new_index = 0
        for index, row in data.iterrows():

            # Store values from columns 'seq1'/CDR3 and 'seq2'/pep
            seq1 = row[seq1_name]
            seq2 = row[seq2_name]

            # Get all of the properties
            seq1_mass, seq1_ip, seq1_hpho, seq1_hphi = pep_properties(seq1)
            seq2_mass, seq2_ip, seq2_hpho, seq2_hphi = pep_properties(seq2)

            # Loop to calculate the interaction maps for each specified property
            it = 0
            for prop in properties_to_calculate:

                if prop == 'mass':
                    mass_diff = pairwise_diff(seq1_mass, seq2_mass, seq1_len, seq2_len)
                    pairwise_diff_tensor[new_index, :, :, it] = mass_diff

                elif prop == 'ip':
                    ip_diff = pairwise_diff(seq1_ip, seq2_ip, seq1_len, seq2_len)
                    pairwise_diff_tensor[new_index, :, :, it] = ip_diff

                elif prop == 'hpho':
                    hpho_diff = pairwise_diff(seq1_hpho, seq2_hpho, seq1_len, seq2_len)
                    pairwise_diff_tensor[new_index, :, :, it] = hpho_diff

                elif prop == 'hphi':
                    hphi_diff = pairwise_diff(seq1_hphi, seq2_hphi, seq1_len, seq2_len)
                    pairwise_diff_tensor[new_index, :, :, it] = hphi_diff

                elif prop == 'blosum':
                    seq1_encoded = enc_list_bl_max_len([seq1], encoding, seq1_len)/5
                    seq2_encoded = enc_list_bl_max_len([seq2], encoding, seq2_len)/5
                    cos_sim_result = cosine_similarity(seq1_encoded[0, :, :], seq2_encoded[0, :, :])
                    pairwise_diff_tensor[new_index, :, :, it] = cos_sim_result

                it += 1

            # Update index
            new_index += 1

    else:

        print("\nNo specific properties were introduced. Therefore, the interaction maps have been calculated for all possible properties.")

        # Initialize an empty tensor to store the pairwise absolute differences
        pairwise_diff_tensor = np.zeros((len(data), seq1_len, seq2_len, len(valid_properties)))

        # Iterate over each row of the DataFrame
        new_index = 0
        for index, row in data.iterrows():

            # Store values from columns 'A3', 'B3', and 'peptide'
            seq1 = row[seq1_name]
            seq2 = row[seq2_name]

            # Get the mass, isoelectric point, hydrophobicity and hydrophilicity
            seq1_mass, seq1_ip, seq1_hpho, seq1_hphi = pep_properties(seq1)
            seq2_mass, seq2_ip, seq2_hpho, seq2_hphi = pep_properties(seq2)

            # Get the absolute difference matrixes for CDR3a-pep and CDR3b-pep properties
            mass_diff = pairwise_diff(seq1_mass, seq2_mass, seq1_len, seq2_len)
            ip_diff = pairwise_diff(seq1_ip, seq2_ip, seq1_len, seq2_len)
            hpho_diff = pairwise_diff(seq1_hpho, seq2_hpho, seq1_len, seq2_len)
            hphi_diff = pairwise_diff(seq1_hphi, seq2_hphi, seq1_len, seq2_len)

            # Get the blosum interaction map
            seq1_encoded = enc_list_bl_max_len([seq1], encoding, seq1_len)/5
            seq2_encoded = enc_list_bl_max_len([seq2], encoding, seq2_len)/5
            cos_sim_result = cosine_similarity(seq1_encoded[0, :, :], seq2_encoded[0, :, :])

            # Assign the pairwise differences to the tensor for A3
            pairwise_diff_tensor[new_index] = np.array([
                mass_diff, ip_diff, hpho_diff, hphi_diff, cos_sim_result
            ])

            # Update index
            new_index += 1

    return pairwise_diff_tensor


