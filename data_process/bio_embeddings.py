#!/usr/bin/env python
import bio_embeddings.embed
import sys
import yaml

import esm2_embedder

# Function to load yaml configuration file
def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config

def get_bio_embedder(name = 'esm1b'):
    if name == 'bepler':
        embedder = bio_embeddings.embed.BeplerEmbedder()
    elif name == 'cpcprot':
        embedder = bio_embeddings.embed.CPCProtEmbedder()
    elif name == 'esm':
        embedder = bio_embeddings.embed.ESMEmbedder()
    elif name == 'esm1b':
        embedder = bio_embeddings.embed.ESM1bEmbedder()
    elif name == 'esm1v':
        embedder = bio_embeddings.embed.ESM1vEmbedder()
    
    elif name == 'esm2':
        embedder = esm2_embedder.ESM2Embedder()
        
    elif name == 'plus_rnn':
        embedder = bio_embeddings.embed.PLUSRNNEmbedder()
    elif name == 'prottrans_albert_bfd':
        embedder = bio_embeddings.embed.ProtTransAlbertBFDEmbedder()
    elif name == 'prottrans_bert_bfd':
        embedder = bio_embeddings.embed.ProtTransBertBFDEmbedder()
    elif name == 'prottrans_t5_bfd':
        embedder = bio_embeddings.embed.ProtTransT5BFDEmbedder()
    elif name == 'prottrans_t5_uniref50':
        embedder = bio_embeddings.embed.ProtTransT5UniRef50Embedder()
    elif name == 'prottrans_t5_xl_u50':
        embedder = bio_embeddings.embed.ProtTransT5XLU50Embedder()
    elif name == 'prottrans_xlnet_uniref100':
        embedder = bio_embeddings.embed.ProtTransXLNetUniRef100Embedder()
    elif name == 'seqvec':
        embedder = bio_embeddings.embed.SeqVecEmbedder()
    elif name == 'unirep':
        embedder = bio_embeddings.embed.UniRepEmbedder()
    elif name == 'fasttext':
        embedder = bio_embeddings.embed.FastTextEmbedder()
    elif name == 'glove':
        embedder = bio_embeddings.embed.GloveEmbedder()
    elif name == 'one_hot_encoding':
        embedder = bio_embeddings.embed.OneHotEncodingEmbedder()
    elif name == 'word2vec':
        embedder = bio_embeddings.embed.Word2VecEmbedder()
    else:
        try:
            raise ValueError('Unknown embedding name')
        except ValueError as error:
            print(error)
            sys.exit(1)

    return embedder
