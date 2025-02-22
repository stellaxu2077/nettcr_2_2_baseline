#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
import os
import sys
import numpy as np
import pandas as pd
import utils
import argparse
import yaml

#Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
args = parser.parse_args()
config_filename_model = args.config

# Load config
config_main = utils.load_config('main_config.yaml')
config_model = utils.load_config(config_filename_model)

embedder_index_tcr = config_model['default']['embedder_index_tcr']
embedder_index_peptide = config_model['default']['embedder_index_peptide']
embedder_index_cdr3 = config_model['default']['embedder_index_cdr3']

config_filename_tcr = 't{}_config.yaml'.format(embedder_index_tcr)
config_filename_peptide = 'p{}_config.yaml'.format(embedder_index_peptide)

config_tcr = utils.load_config(config_filename_tcr)
config_peptide = utils.load_config(config_filename_peptide)

# Set parameters from config
data_filename = config_main['default']['data_filename']
seed = config_main['default']['seed']

model_index = config_model['default']['model_index']
padding_value_peptide = config_model['default']['padding_value_peptide']
padding_side_peptide = config_model['default']['padding_side_peptide']
truncating_side_peptide = config_model['default']['truncating_side_peptide']
padding_value_tcr = config_model['default']['padding_value_tcr']
padding_side_tcr = config_model['default']['padding_side_tcr']
truncating_side_tcr = config_model['default']['truncating_side_tcr']
peptide_normalization_divisor = config_model['default']['peptide_normalization_divisor']
tcr_normalization_divisor = config_model['default']['tcr_normalization_divisor']
cdr3_normalization_divisor = config_model['default']['cdr3_normalization_divisor']
model_architecture_name = config_model['default']['model_architecture_name']

embedder_name_tcr = config_tcr['default']['embedder_name_tcr']
embedder_source_tcr = config_tcr['default']['embedder_source_tcr']

embedder_name_peptide = config_peptide['default']['embedder_name_peptide']
embedder_source_peptide = config_peptide['default']['embedder_source_peptide']

# Set random seed
keras.utils.set_random_seed(seed)

### Input/Output ###
# Read in data
data = pd.read_csv(filepath_or_buffer = os.path.join('../data/raw',
                                                     data_filename),
                   usecols = ['A1',
                              'A2',
                              'A3',
                              'B1',
                              'B2',
                              'B3',
                              'peptide',
                              'binder',
                              'partition',
                              'original_index'])

# Get dataframe with unique encoded peptides
if embedder_source_peptide == 'in-house':

    df_peptides = (utils
                   .encode_unique_peptides(df = data,
                                           encoding_name = embedder_name_peptide))

else:
    df_peptides = pd.read_pickle(filepath_or_buffer = ('../data/p{}_embedding.pkl'
                                                       .format(embedder_index_peptide)))

# Get dataframe with unique encoded CDRs
if embedder_source_tcr == 'in-house':

    df_tcrs = (utils
               .encode_unique_tcrs(df = data,
                                   encoding_name = embedder_name_tcr))
else:
    df_tcrs = pd.read_pickle(filepath_or_buffer = ('../data/t{}_embedding.pkl'
                                                   .format(embedder_index_tcr)))

if embedder_index_cdr3:
# Replace CDR3 embeddings
    df_cdr3 = pd.read_pickle(filepath_or_buffer = ('../data/c{}_embedding.pkl'
                                                   .format(embedder_index_cdr3)))

    df_tcrs = (df_tcrs
               .drop(labels = ['a3_encoded',
                               'b3_encoded'],
                     axis = 'columns')
               .merge(right = df_cdr3,
                      how = 'left',
                      on = 'original_index'))

    del df_cdr3

# Pad unique peptides and CDRs
df_peptides = (utils
               .pad_unique_peptides(df = df_peptides,
                                    padding_value = padding_value_peptide,
                                    padding_side = padding_side_peptide,
                                    truncating_side = truncating_side_peptide))
df_tcrs = (utils
           .pad_unique_tcrs(df = df_tcrs,
                            padding_value = padding_value_tcr,
                            padding_side = padding_side_tcr,
                            truncating_side = truncating_side_tcr))

df_peptides = (df_peptides
               .drop(labels = 'count',
                     axis = 1))


# Normalise embeddings
df_tcrs[['a1_encoded',
         'a2_encoded',
         'b1_encoded',
         'b2_encoded']] /= tcr_normalization_divisor

df_tcrs[['a3_encoded',
         'b3_encoded']] /= cdr3_normalization_divisor

df_peptides['peptide_encoded'] /= peptide_normalization_divisor

df_tcrs = df_tcrs[['a3_encoded', 'b3_encoded']]

if model_architecture_name == 'ff_CDR3':
# Calculate embeddings per CDR and flatten peptide embeddings
    df_peptides['peptide_encoded'] = (df_peptides['peptide_encoded']
                                      .map(arg = lambda x: x.flatten()))
    df_tcrs = (df_tcrs
               .applymap(func = lambda x: np.mean(a = x,
                                                  axis = 0)))

# Do the prediction
partitions_count = 5
for t in range(partitions_count):

    # Get test data
    test_partition_mask = data['partition'] == t
    df_test = (data[test_partition_mask]
               .filter(items = ['peptide',
                                'original_index']))

    # Join unique encoded sequences onto the data set
    df_test = (df_test
               .merge(right = df_peptides,
                      how = 'left',
                      on = 'peptide')
               .merge(right = df_tcrs,
                      how = 'left',
                      on = 'original_index')
               .drop(labels = ['peptide',
                               'original_index'],
                     axis = 'columns'))

    # Make test data into a dict of numpy arrays
    df_test = dict(df_test)

    for key, value in df_test.items():
        df_test[key] = np.stack(arrays = value)

    # Rename keys
    key_map = (('peptide_encoded', 'pep'),
              # ('a1_encoded', 'a1'),
              # ('a2_encoded', 'a2'),
               ('a3_encoded', 'a3'),
              # ('b1_encoded', 'b1'),
              # ('b2_encoded', 'b2'),
               ('b3_encoded', 'b3'))

    for old_key, new_key in key_map:
        df_test[new_key] = df_test.pop(old_key)

    # Initialize
    avg_prediction = 0

    for v in range(partitions_count):
        if v!=t:

            # Load the model
            model = keras.models.load_model(filepath = '../checkpoint/m{}/m{}_t{}v{}'.format(model_index, model_index, t, v),
                                            custom_objects = {'auc01': utils.auc01})

            # Do prediction by one model
            #avg_prediction += model.predict(x = df_test)


            # Do prediction by one model
            preds = model.predict(x=df_test)
            print(f"Predictions for model m{model_index}_t{t}v{v}: {preds[:5]} ...")  # Print first 5 predictions

            # Get the corresponding true labels from the original data
            true_labels = data.loc[test_partition_mask, 'binder'].values

            # Calculate AUC0.1 for the current model
            auc_01_score = roc_auc_score(true_labels, preds, max_fpr=0.1)
            print(f"AUC0.1 for model m{model_index}_t{t}v{v}: {auc_01_score:.4f}")

            # Accumulate the predictions for averaging
            avg_prediction += preds

            # Clear the session for the next model
            tf.keras.backend.clear_session()

    # Average the predictions between all models in the inner loop
    avg_prediction = avg_prediction / 4
    data.loc[test_partition_mask, 'prediction'] = avg_prediction

# Save predictions
data.to_csv(path_or_buf = '../data/m{}_predictions.tsv'.format(model_index),
            sep = '\t',
            index = False)
