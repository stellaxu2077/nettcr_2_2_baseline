# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
@NetTCR2.2_author: Mathias
@modifications_authors: Carlos, Jonas
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score

import os
import sys
import numpy as np
import pandas as pd
import h5py

# Imports the util module and network architectures for NetTCR
# If changes are wanted, change this directory to your own directory
# Directory with the "keras_utils.py" script and model architecture
sys.path.append("/home/projects2/jonnil/nettcr_2024/nettcr_2d/NetTCR-2Dmaps/keras_src")

import keras_utils
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse

# Importing the model
from CNN_keras_carlos_architecture import CNN_1D_global_max

# Makes the plots look better
sns.set()

def args_parser():
    parser = argparse.ArgumentParser(description='NetTCR training script')
    """
    Data processing args
    """
    parser.add_argument('-trf', '--train_file', dest='train_file', required=True, type=str,
                        default='/home/projects/vaccine/people/cadsal/MasterThesis/data/nettcr_2_2_full_dataset.csv',
                        help='Filename of the data input file')
    parser.add_argument('-out', '--out_dir', dest='out_directory', required=True, type=str,
                        default='/home/projects/vaccine/people/cadsal/MasterThesis/baseline/outdir',
                        help='Output directory')
    '''
    parser.add_argument('-tensordir', '--tensor_dir', dest='tensor_directory', required=True, type=str,
                        default='/home/projects2/jonnil/nettcr_2024/data/',
                        help='Directory holding ESM-2 tensors')
    '''
    parser.add_argument('-h5', '--hdf5_path', dest='hdf5_path', required=True, type=str,
                        default='/home/projects2/jonnil/nettcr_2024/data/data.h5',
                        help='Path to the HDF5 file containing ESM-2 tensors')
    parser.add_argument("-tp", "--test_partition", dest='test_fold', required=False, type=int,
                        default=0, help='Test partition for the nested-CV')
    parser.add_argument("-vp", "--valid_partition", dest='valid_fold', required=False, type=int,
                        default=1, help='Test partition for the nested-CV')
    parser.add_argument("-s", "--seed", dest="seed", required=False, type=int,
                        default=1, help='Seed for fixing random initializations')
    parser.add_argument("-epochs", "--epochs", dest="epochs", required=False, type=int,
                        default=200, help='Number of training epochs')
    parser.add_argument("-bs", "--batch_size", dest="batch_size", required=False, type=int,
                        default=64, help='Batch size')
    parser.add_argument("-patience", "--patience", dest="patience", required=False, type=int,
                        default=100, help='Patience for early stopping')
    parser.add_argument("-do", "--dropout", dest="dropout", required=False, type=float,
                        default=0.6, help='Dropout value for CNN layers after max.pooling and concatenating them')
    parser.add_argument("-lr", "--learning_rate", dest="learning_rate", required=False, type=float,
                        default=0.0005, help='Learning rate')
    parser.add_argument("-w", "--sample_weight", dest="sample_weight", action="store_true",
                        default=False, help='Use sample weighting')
    return parser.parse_args()


# Parse the command-line arguments and store them in the 'args' variable
args = args_parser()
args_dict = vars(args)

# Define the test and validation partitions
t = int(args_dict['test_fold'])
v = int(args_dict['valid_fold'])

# Set random seed
seed = int(args.seed)
np.random.seed(seed)      # Numpy module
random.seed(seed)         # Python random module
tf.random.set_seed(seed)  # Tensorflow random seed

# Plots function
def plot_loss_auc(train_losses, valid_losses, train_aucs, valid_aucs,
                   filename, dpi=300):
    f, a = plt.subplots(2, 1, figsize=(12, 10))
    a = a.ravel()
    a[0].plot(train_losses, label='train_losses')
    a[0].plot(valid_losses, label='valid_losses')
    a[0].legend()
    a[0].set_title('Baseline CDR123 Loss (TestFold: ' + str(t) + ', ValidFold: ' + str(v) + ')')
    a[1].plot(train_aucs, label='train_aucs')
    a[1].plot(valid_aucs, label='valid_aucs')
    a[1].legend()
    a[1].set_title('Baseline CDR123 AUC (TestFold: ' + str(t) + ', ValidFold: ' + str(v) + ')')
    a[1].set_xlabel('Epoch')
    f.savefig(f'{filename}', dpi=dpi, bbox_inches='tight')

def plot_loss_auc01(train_losses, valid_losses, train_aucs, valid_aucs,
                    filename, dpi=300):
    f, a = plt.subplots(2, 1, figsize=(12, 10))
    a = a.ravel()
    a[0].plot(train_losses, label='train_losses')
    a[0].plot(valid_losses, label='valid_losses')
    a[0].legend()
    a[0].set_title('Baseline CDR123 Loss (TestFold: ' + str(t) + ', ValidFold: ' + str(v) + ')')
    a[1].plot(train_aucs, label='train_aucs')
    a[1].plot(valid_aucs, label='valid_aucs')
    a[1].legend()
    a[1].set_title('Baseline CDR123 AUC0.1 (TestFold: ' + str(t) + ', ValidFold: ' + str(v) + ')')
    a[1].set_xlabel('Epoch')
    f.savefig(f'{filename}', dpi=dpi, bbox_inches='tight')

### Input/Output ###

# Read in data
data = pd.read_csv(args_dict['train_file'])
# Directories
outdir = args_dict['out_directory']

### Weigthed loss definition ###

if args_dict['sample_weight']:

    # Sample weights
    weight_dict = np.log2(data.shape[0]/(data.peptide.value_counts()))
    # Normalize, so that loss is comparable
    weight_dict = weight_dict*(data.shape[0]/np.sum(weight_dict*data.peptide.value_counts()))
    data["sample_weight"] = data["peptide"].map(weight_dict)
    # # Adjust according to if the observation include both paired-chain sequence data for the CDRs
    # weight_multiplier_dict = {"alpha": 1, "beta": 1, "paired": 2}
    # data["weight_multiplier"] = data.input_type.map(weight_multiplier_dict)
    # data["sample_weight"] = data["sample_weight"]*data["weight_multiplier"]/weight_multiplier_dict["paired"]

else:
    # If sample_weight is False, set sample_weight to 1 for all rows
    data["sample_weight"] = 1

# Define the list of binding peptides in the data (descending order according to the number of occurrences)
pep_list = list(data[data.binder==1].peptide.value_counts(ascending=False).index)

### Model training parameters ###

train_parts = {0, 1, 2, 3, 4}                    # Partitions
patience = args.patience                                    # Patience for Early Stopping
dropout_rate = args_dict['dropout']              # Dropout Rate
EPOCHS = args.epochs                                     # Number of epochs in the training
batch_size = args.batch_size                                  # Number of elements in each batch

# Padding to max. length according to the observations in the dataset
a1_max = 7
a2_max = 8
a3_max = 22
b1_max = 6
b2_max = 7
b3_max = 23
pep_max = 12

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, 
                 #tensor_directory, 
                 hdf5_path,
                 max_seq_lens, seq_names, inp_names, target_column='binder', weight_column='sample_weight', shuffle=True):
        self.df = df
        self.batch_size = batch_size
        #self.tensor_directory = tensor_directory
        self.hdf5_path = hdf5_path
        self.max_seq_lens = max_seq_lens
        self.seq_names = seq_names
        self.inp_names = inp_names
        self.target_column = target_column
        self.weight_column = weight_column
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))  # Store indices for shuffling
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        # Number of batches
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        # Select a batch of data based on shuffled indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        sorted_batch_indices = np.sort(batch_indices)
        batch_df = self.df.iloc[sorted_batch_indices]

        '''
        # Encode the inputs for this batch
        encoded_inputs = keras_utils.enc_list_esm2(batch_df, 
                                                 self.tensor_directory, 
                                                 seq_names=self.seq_names, 
                                                 max_seq_lens=self.max_seq_lens)
        # Split the encoded sequences
        #encoded_inputs = encoded_seqs[:len(self.seq_names)]
        targets = batch_df[self.target_column].values
        sample_weights = batch_df[self.weight_column].values

        #return {name: inp/5 for name, inp in zip(self.inp_names, encoded_inputs)}, targets, sample_weights
        input_dict = {name: inp for name, inp in zip(self.inp_names, encoded_inputs)}
        input_dict['pep'] = input_dict['pep'] / 5
        '''
        hdf5_indices = batch_df['new_index'].values  # from original_index get true index
        sorted_hdf5_indices = np.sort(hdf5_indices)  # hdf5 require increasing index

        '''
        input_dict = {name: np.zeros((len(batch_indices), self.max_seq_lens[name],
                      20 if name == 'peptide' else 1280), dtype=np.float32)
                      for name in self.inp_names}
        '''
        input_dict = {}

        if not os.path.exists(self.hdf5_path):
            print(f"HDF5 file does not exist: {self.hdf5_path}")
        if not os.access(self.hdf5_path, os.R_OK):
            print(f"HDF5 file is not readable: {self.hdf5_path}")

        print(f"Opening HDF5 file at: {self.hdf5_path}") 
        '''
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            for name in self.inp_names:
                #input_dict[name] = hdf5_file[name][batch_indices]
                input_dict[name] = hdf5_file[name][sorted_hdf5_indices]
                if name == 'pep' or name == 'peptide':  # 根据你的输入名称调整
                    input_dict[name] = input_dict[name] / 5.0  # 归一化处理

            # 读取标签和样本权重
            targets = hdf5_file[self.target_column][sorted_hdf5_indices]
            #sample_weights = hdf5_file[self.weight_column][batch_indices]
            original_order = np.argsort(hdf5_indices)  # 根据原始顺序还原
            '''
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:

            for h5_key, model_input_key in zip(['peptide', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3'], self.inp_names):
                max_valid_index = hdf5_file[h5_key].shape[0]
                sorted_hdf5_indices = sorted_hdf5_indices[sorted_hdf5_indices < max_valid_index]
                if len(sorted_hdf5_indices) == 0:
                    raise ValueError(f"All indices are out of bounds for dataset {h5_key}.")

                max_index = hdf5_file[h5_key].shape[0]
                valid_indices = sorted_hdf5_indices[sorted_hdf5_indices < max_index]
                if len(valid_indices) == 0:
                    raise ValueError(f"No valid indices for dataset {h5_key}. All indices are out of bounds.")
                
                input_dict[model_input_key] = hdf5_file[h5_key][sorted_hdf5_indices]
                if model_input_key == 'pep':  # 对 peptide 进行归一化
                    input_dict[model_input_key] = input_dict[model_input_key] / 5.0
                
            targets = hdf5_file[self.target_column][sorted_hdf5_indices]

            original_order = np.argsort(hdf5_indices)
            targets = targets[original_order]
            print(f"Original order: {original_order}, shape: {original_order.shape}")
            print(f"Targets shape: {targets.shape}")
 
        for key in input_dict.keys():
            input_dict[key] = input_dict[key][original_order]
        
        '''
        for name in self.inp_names:
            input_dict[name] = input_dict[name][original_order]
        '''
        sample_weights = batch_df["sample_weight"].values
        
        return input_dict, targets, sample_weights

    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)


# AUC custom function
def my_numpy_function(y_true, y_pred):
    try:
        auc = roc_auc_score(y_true, y_pred, max_fpr = 0.1)
    except ValueError:
        # Exception for when a positive observation is not present in a batch
        auc = np.array([float(0)])
    return auc

# Custom metric for AUC 0.1
def auc_01(y_true, y_pred):
    "Allows Tensorflow to use the function during training"
    auc_01 = tf.numpy_function(my_numpy_function, [y_true, y_pred], tf.float64)
    return auc_01

# Creates the directory to save the model in
if not os.path.exists(outdir):
    os.makedirs(outdir)

dependencies = {
    'auc_01': auc_01
}

outfile = open(outdir + "/" + "s.{}.t.{}.v.{}.fold_validation.tsv".format(seed,t,v), mode = "w")
print("fold", "valid_loss_auc01", "best_auc_0.1", "best_epoch_auc01", "best_auc", "best_epoch_auc", sep = "\t", file = outfile)

# Prepare plotting
fig, ax = plt.subplots(figsize=(15, 10))

### Data and model initialization ###


# Training data (not including validation and test sets)
x_train_df = data[(data.partition!=t)&(data.partition!=v)].reset_index()

# Define sequence names and max lengths
seq_names = ['peptide', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3']
#inp_names = ['peptide','A1','A2','A3','B1','B2','B3']
inp_names = ['pep', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3']
#max_seq_lens = [pep_max, a1_max, a2_max, a3_max, b1_max, b2_max, b3_max]
max_seq_lens = {
    'peptide': 12,
    'A1': 7,
    'A2': 8,
    'A3': 22,
    'B1': 6,
    'B2': 7,
    'B3': 23
}


# Create training and validation generators
train_generator = DataGenerator(df=x_train_df, 
                                 batch_size=batch_size, 
                                 #tensor_directory=args.tensor_directory,
                                 hdf5_path=args.hdf5_path,
                                 max_seq_lens=max_seq_lens, 
                                 seq_names=seq_names,
                                 inp_names=inp_names,
                                 shuffle=True)  # Enable shuffling
# Validation data - Used for early stopping
# x_valid_df = data[(data.partition==v) & (data.input_type == "paired")]
x_valid_df = data[(data.partition==v)]


valid_generator = DataGenerator(df=x_valid_df, 
                                 batch_size=batch_size, 
                                 #tensor_directory=args.tensor_directory,
                                 hdf5_path=args.hdf5_path, 
                                 max_seq_lens=max_seq_lens, 
                                 seq_names=seq_names,
                                 inp_names=inp_names,
                                 shuffle=False)


# Debugging: check one batch
sample_batch = train_generator[0]
print("Sample batch inputs:", sample_batch[0].keys())
for name, arr in sample_batch[0].items():
    print(f"{name} shape: {arr.shape}")
print("Targets shape:", sample_batch[1].shape)
print("Sample weights shape:", sample_batch[2].shape)


# Selection of the model to train
model = CNN_1D_global_max(dropout_rate, seed, hidden_sizes=[64,32], embed_dims={'peptide': 20, 'cdr': 1280}, nfilters = 16, input_dict={'pep':{'ks':[1,3,5,7,9],'dim':20},
                                                                        'a1':{'ks':[1,3,5,7,9],'dim':7},
                                                                        'a2':{'ks':[1,3,5,7,9],'dim':8},
                                                                        'a3':{'ks':[1,3,5,7,9],'dim':22},
                                                                        'b1':{'ks':[1,3,5,7,9],'dim':6},
                                                                        'b2':{'ks':[1,3,5,7,9],'dim':7},
                                                                        'b3':{'ks':[1,3,5,7,9],'dim':23}})

#model =  CNN_CDR123_1D_baseline(dropout_rate, seed, nr_of_filters_1 = 32)

# Saves the model at the best epoch (based on validation loss or other metric)
ModelCheckpoint = keras.callbacks.ModelCheckpoint(
        filepath = outdir + '/checkpoint/' + 's.' + str(seed) + '.t.' + str(t) + '.v.' + str(v) + ".keras",
        monitor = "val_auc", # We use auc instead of auc 0.1 for esm models
        mode = "max",
        save_best_only = True)

# EarlyStopping function used for stopping model training when the model does not improve
EarlyStopping = keras.callbacks.EarlyStopping(
    monitor = "val_auc", # We use auc instead of auc 0.1 for esm models
    mode = "max",
    patience = patience)

# Callbacks to include for the model training
callbacks_list = [EarlyStopping,
                  ModelCheckpoint
    ]

# Optimizers, loss functions, and additional metrics to track
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = args_dict['learning_rate']),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = [auc_01, "AUC"],
              weighted_metrics = [])


### Announce Training ###

print("Training model with test_partition = {} & validation_partition = {}".format(t,v), end = "\n")

# Train the model using generators
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    verbose=2,
    callbacks=callbacks_list
)

# Loss and metrics for each epoch during training
valid_loss = history.history["val_loss"]
train_loss = history.history["loss"]
valid_auc = history.history["val_auc"]
valid_auc01 = history.history["val_auc_01"]
train_auc = history.history["auc"]
train_auc01 = history.history["auc_01"]

plotdir = outdir + "/plots"

if not os.path.exists(plotdir):
    os.makedirs(plotdir)

# Plotting the losses
ax.plot(train_loss, label='train')
ax.plot(valid_loss, label='validation')
ax.set_ylabel("Loss")
ax.set_xlabel("Epoch")
ax.legend()
ax.set_title('Baseline CDR123 Loss (TestFold: ' + str(t) + ', ValidFold: ' + str(v) + ')')
# Save training/validation loss plot
plt.tight_layout()
plt.show()
fig.savefig(plotdir + '/s.{}.t.{}.v.{}.learning_curves.png'.format(seed,t,v), dpi=200)

# Plotting the losses with the AUC value
plot_loss_auc(train_loss, valid_loss, train_auc, valid_auc,
              plotdir + '/s.{}.t.{}.v.{}.lossVSAUC.png'.format(seed,t,v), dpi=300)
plot_loss_auc01(train_loss, valid_loss, train_auc01, valid_auc01,
                plotdir + '/s.{}.t.{}.v.{}.lossVSAUC01.png'.format(seed,t,v), dpi=300)


# Record metrics at checkpoint
fold = outdir + '/checkpoint/' + "s." + str(seed) + '.t.' + str(t) + '.v.' + str(v) + ".keras"
valid_best = valid_loss[np.argmax(valid_auc01)]
best_epoch_auc01 = np.argmax(valid_auc01)
best_auc01 = np.max(valid_auc01)
best_epoch_auc = np.argmax(valid_auc)
best_auc = np.max(valid_auc)

# Load the best model
model = keras.models.load_model(outdir + '/checkpoint/' + 's.' + str(seed) +  '.t.' + str(t) + '.v.' + str(v) + ".keras", custom_objects=dependencies)

# Converting the model to a TFlite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the model
with open(outdir + '/checkpoint/' + 's.' + str(seed) +  '.t.' + str(t) + '.v.' + str(v) + ".tflite", 'wb') as f:
  f.write(tflite_model)

# Records loss and metrics at saved epoch
print(fold, valid_best, best_auc01, best_epoch_auc01, best_auc, best_epoch_auc, sep = "\t", file = outfile)

# Clears the session for the next model
tf.keras.backend.clear_session()

# Close log file
outfile.close()
