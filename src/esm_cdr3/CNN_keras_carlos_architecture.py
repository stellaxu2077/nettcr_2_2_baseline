# -*- coding: utf-8 -*-
"""
@authors: Mathias and Carlos
"""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2
import numpy as np

#These networks are based on NetTCR 2.1 by Alessandro Montemurro


def CNN_1D_global_max_1(dropout_rate, seed, hidden_sizes=[64,32], 
                                            embed_dim = 20, 
                                            nfilters=16, 
                                            input_dict={'pep':{'ks':[1,3,5,7,9],'dim':12},
                                                       # 'a1':{'ks':[1,3,5,7,9],'dim':7},
                                                       # 'a2':{'ks':[1,3,5,7,9],'dim':8},
                                                        'a3':{'ks':[1,3,5,7,9],'dim':22},
                                                       # 'b1':{'ks':[1,3,5,7,9],'dim':6},
                                                       # 'b2':{'ks':[1,3,5,7,9],'dim':7},
                                                        'b3':{'ks':[1,3,5,7,9],'dim':23}}):

    #"""A generic 1D CNN model initializer function, used for basic NetTCR-like architectures"""

    assert len(input_dict) > 0, "input_names cannot be empty"
    assert len(hidden_sizes) > 0, "Must provide at least one hidden layer size"

    # Second dimension of the embedding
    embed_dim = embed_dim

    conv_activation = "relu"
    dense_activation = "sigmoid"

    input_names = [name for name in input_dict]

    # Inputs
    inputs = [keras.Input(shape = (input_dict[name]['dim'], embed_dim), name=name) for name in input_names]

    cnn_layers = []
    # Define CNN layers inputs
    for name, inp in zip(input_names, inputs):
        for k in input_dict[name]['ks']:  # kernel sizes
            conv = layers.Conv1D(filters=nfilters, kernel_size=k, padding="same", name=f"{name}_conv_{k}")(inp)
            bn = layers.BatchNormalization()(conv)
            activated = layers.Activation(conv_activation)(bn)  # Apply activation after BatchNorm
            cnn_layers.append(activated)

    pool_layers = [layers.GlobalMaxPooling1D()(p) for p in cnn_layers]

    # Concatenate all max pooling layers to a single layer
    cat = layers.Concatenate()(pool_layers)

    # Dropout - Required to prevent overfitting
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    # Dense layer
    dense = [layers.Dense(hidden_sizes[0], activation = dense_activation)(cat_dropout)]

    if len(hidden_sizes) > 1:
        for i in range(1, len(hidden_sizes)):
            dense.append(layers.Dense(hidden_sizes[i], activation = dense_activation)(dense[i-1]))

    # Output layer
    out = layers.Dense(1,activation = "sigmoid")(dense[-1])

    # Prepare model object
    model = keras.Model(inputs = inputs, outputs = out)

    return model



def CNN_1D_global_max(dropout_rate, seed, hidden_sizes=[64, 32],
                      #embed_dim=20,
                      embed_dims={'peptide': 20, 'cdr': 1280},
                      nfilters=16,
                      input_dict={'pep': {'ks': [1, 3, 5, 7, 9], 'dim': 12},
                                 # 'a1': {'ks': [1, 3, 5, 7, 9], 'dim': 7},
                                 # 'a2': {'ks': [1, 3, 5, 7, 9], 'dim': 8},
                                  'a3': {'ks': [1, 3, 5, 7, 9], 'dim': 22},
                                 # 'b1': {'ks': [1, 3, 5, 7, 9], 'dim': 6},
                                 # 'b2': {'ks': [1, 3, 5, 7, 9], 'dim': 7},
                                  'b3': {'ks': [1, 3, 5, 7, 9], 'dim': 23}}):
    """A 1D CNN model mimicking a fixed structure but with dynamic input support."""

    assert len(input_dict) > 0, "input_dict cannot be empty"
    assert len(hidden_sizes) > 0, "Must provide at least one hidden layer size"

    conv_activation = "relu"
    dense_activation = "sigmoid"

    # Define inputs dynamically based on input_dict
    '''
    #inputs = {name: keras.Input(shape=(info['dim'], embed_dim), name=name) for name, info in input_dict.items()}
    '''
    inputs = {}
    for name, info in input_dict.items():
        if name == 'pep':
            embed_dim = embed_dims['peptide']
        else:
            embed_dim = embed_dims['cdr']
        inputs[name] = keras.Input(shape=(info['dim'], embed_dim), name=name)

    # Define CNN layers with fixed kernel sizes for each input
    cnn_outputs = []
    for name, inp in inputs.items():
        conv_outputs = []
        for k in input_dict[name]['ks']:  # Fixed kernel sizes
            conv = layers.Conv1D(filters=nfilters, kernel_size=k, padding="same",
                                 activation=conv_activation, name=f"{name}_conv_{k}")(inp)
            pool = layers.GlobalMaxPooling1D(name=f"{name}_pool_{k}")(conv)  # MaxPooling after Conv
            conv_outputs.append(pool)
        # Concatenate outputs for all kernel sizes for the current input
        concatenated = layers.Concatenate(name=f"{name}_concat")(conv_outputs)
        cnn_outputs.append(concatenated)

    # Concatenate all inputs
    cat = layers.Concatenate(name="global_concat")(cnn_outputs)

    # Apply dropout
    cat_dropout = layers.Dropout(dropout_rate, seed=seed)(cat)

    # Fully connected layers
    dense = [layers.Dense(hidden_sizes[0], activation=dense_activation, name="dense_0")(cat_dropout)]
    for i in range(1, len(hidden_sizes)):
        dense.append(layers.Dense(hidden_sizes[i], activation=dense_activation, name=f"dense_{i}")(dense[-1]))

    # Output layer
    output = layers.Dense(1, activation="sigmoid", name="output_layer")(dense[-1])

    # Create model
    model = keras.Model(inputs=list(inputs.values()), outputs=output)

    return model



# Main baseline architecture for all paired-chain CDR sequences
def CNN_CDR123_1D_baseline(dropout_rate, seed, conv_activation = "relu", dense_activation = "sigmoid",
                           embed_dim = 20, nr_of_filters_1 = 16, max_lengths = None):

    # Max.length of the sequences from the dataset
    if max_lengths:
        a1_max = max_lengths[0]
        a2_max = max_lengths[1]
        a3_max = max_lengths[2]
        b1_max = max_lengths[3]
        b2_max = max_lengths[4]
        b3_max = max_lengths[5]
        pep_max = max_lengths[6]
    else:
        a1_max = 7
        a2_max = 8
        a3_max = 22
        b1_max = 6
        b2_max = 7
        b3_max = 23
        pep_max = 12
    
    # Input dimensions
    pep = keras.Input(shape = (pep_max, embed_dim), name ="pep")
    a1 = keras.Input(shape = (a1_max, embed_dim), name ="a1")
    a2 = keras.Input(shape = (a2_max, embed_dim), name ="a2")
    a3 = keras.Input(shape = (a3_max, embed_dim), name ="a3")
    b1 = keras.Input(shape = (b1_max, embed_dim), name ="b1")
    b2 = keras.Input(shape = (b2_max, embed_dim), name ="b2")
    b3 = keras.Input(shape = (b3_max, embed_dim), name ="b3")
    
    # Trained in a pan-specific setup (padding="same" for keeping original input sizes)

    # CNN layers for each feature and the different 5 kernel-sizes
    pep_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_pep_1_conv")(pep)
    pep_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_pep_3_conv")(pep)
    pep_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_pep_5_conv")(pep)
    pep_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_pep_7_conv")(pep)
    pep_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_pep_9_conv")(pep)
    
    a1_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a1_1_conv")(a1)
    a1_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a1_3_conv")(a1)
    a1_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a1_5_conv")(a1)
    a1_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a1_7_conv")(a1)
    a1_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a1_9_conv")(a1)
    
    a2_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a2_1_conv")(a2)
    a2_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a2_3_conv")(a2)
    a2_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a2_5_conv")(a2)
    a2_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a2_7_conv")(a2)
    a2_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a2_9_conv")(a2)
    
    a3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    a3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a3_7_conv")(a3)
    a3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a3_9_conv")(a3)
    
    b1_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b1_1_conv")(b1)
    b1_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b1_3_conv")(b1)
    b1_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b1_5_conv")(b1)
    b1_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b1_7_conv")(b1)
    b1_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b1_9_conv")(b1)
    
    b2_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b2_1_conv")(b2)
    b2_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b2_3_conv")(b2)
    b2_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b2_5_conv")(b2)
    b2_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b2_7_conv")(b2)
    b2_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b2_9_conv")(b2)
    
    b3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3)
    b3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b3_7_conv")(b3)
    b3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b3_9_conv")(b3) 
    
    # GlobalMaxPooling: takes the maximum value along the entire sequence for each feature map (16 filters)
    pep_1_pool = layers.GlobalMaxPooling1D(name = "first_pep_1_pool")(pep_1_CNN)
    pep_3_pool = layers.GlobalMaxPooling1D(name = "first_pep_3_pool")(pep_3_CNN)
    pep_5_pool = layers.GlobalMaxPooling1D(name = "first_pep_5_pool")(pep_5_CNN)
    pep_7_pool = layers.GlobalMaxPooling1D(name = "first_pep_7_pool")(pep_7_CNN)
    pep_9_pool = layers.GlobalMaxPooling1D(name = "first_pep_9_pool")(pep_9_CNN)
    
    a1_1_pool = layers.GlobalMaxPooling1D(name = "first_a1_1_pool")(a1_1_CNN)
    a1_3_pool = layers.GlobalMaxPooling1D(name = "first_a1_3_pool")(a1_3_CNN)
    a1_5_pool = layers.GlobalMaxPooling1D(name = "first_a1_5_pool")(a1_5_CNN)
    a1_7_pool = layers.GlobalMaxPooling1D(name = "first_a1_7_pool")(a1_7_CNN)
    a1_9_pool = layers.GlobalMaxPooling1D(name = "first_a1_9_pool")(a1_9_CNN)
    
    a2_1_pool = layers.GlobalMaxPooling1D(name = "first_a2_1_pool")(a2_1_CNN)
    a2_3_pool = layers.GlobalMaxPooling1D(name = "first_a2_3_pool")(a2_3_CNN)
    a2_5_pool = layers.GlobalMaxPooling1D(name = "first_a2_5_pool")(a2_5_CNN)
    a2_7_pool = layers.GlobalMaxPooling1D(name = "first_a2_7_pool")(a2_7_CNN)
    a2_9_pool = layers.GlobalMaxPooling1D(name = "first_a2_9_pool")(a2_9_CNN)
    
    a3_1_pool = layers.GlobalMaxPooling1D(name = "first_a3_1_pool")(a3_1_CNN)
    a3_3_pool = layers.GlobalMaxPooling1D(name = "first_a3_3_pool")(a3_3_CNN)
    a3_5_pool = layers.GlobalMaxPooling1D(name = "first_a3_5_pool")(a3_5_CNN)
    a3_7_pool = layers.GlobalMaxPooling1D(name = "first_a3_7_pool")(a3_7_CNN)
    a3_9_pool = layers.GlobalMaxPooling1D(name = "first_a3_9_pool")(a3_9_CNN)
    
    b1_1_pool = layers.GlobalMaxPooling1D(name = "first_b1_1_pool")(b1_1_CNN)
    b1_3_pool = layers.GlobalMaxPooling1D(name = "first_b1_3_pool")(b1_3_CNN)
    b1_5_pool = layers.GlobalMaxPooling1D(name = "first_b1_5_pool")(b1_5_CNN)
    b1_7_pool = layers.GlobalMaxPooling1D(name = "first_b1_7_pool")(b1_7_CNN)
    b1_9_pool = layers.GlobalMaxPooling1D(name = "first_b1_9_pool")(b1_9_CNN)
    
    b2_1_pool = layers.GlobalMaxPooling1D(name = "first_b2_1_pool")(b2_1_CNN)
    b2_3_pool = layers.GlobalMaxPooling1D(name = "first_b2_3_pool")(b2_3_CNN)
    b2_5_pool = layers.GlobalMaxPooling1D(name = "first_b2_5_pool")(b2_5_CNN)
    b2_7_pool = layers.GlobalMaxPooling1D(name = "first_b2_7_pool")(b2_7_CNN)
    b2_9_pool = layers.GlobalMaxPooling1D(name = "first_b2_9_pool")(b2_9_CNN)
    
    b3_1_pool = layers.GlobalMaxPooling1D(name = "first_b3_1_pool")(b3_1_CNN)
    b3_3_pool = layers.GlobalMaxPooling1D(name = "first_b3_3_pool")(b3_3_CNN)
    b3_5_pool = layers.GlobalMaxPooling1D(name = "first_b3_5_pool")(b3_5_CNN)
    b3_7_pool = layers.GlobalMaxPooling1D(name = "first_b3_7_pool")(b3_7_CNN)
    b3_9_pool = layers.GlobalMaxPooling1D(name = "first_b3_9_pool")(b3_9_CNN)

    # Concatenation of all MaxPool outputs from all features and kernel-sizes
    cat = layers.Concatenate(name = "first_cat")([pep_1_pool, pep_3_pool, pep_5_pool, pep_7_pool, pep_9_pool,
                                a1_1_pool, a1_3_pool, a1_5_pool, a1_7_pool, a1_9_pool,
                                a2_1_pool, a2_3_pool, a2_5_pool, a2_7_pool, a2_9_pool,
                                a3_1_pool, a3_3_pool, a3_5_pool, a3_7_pool, a3_9_pool,
                                b1_1_pool, b1_3_pool, b1_5_pool, b1_7_pool, b1_9_pool,
                                b2_1_pool, b2_3_pool, b2_5_pool, b2_7_pool, b2_9_pool,
                                b3_1_pool, b3_3_pool, b3_5_pool, b3_7_pool, b3_9_pool])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)
    
    # Dense layers after concatenation+dropout
    dense = layers.Dense(64, activation = dense_activation, name = "first_dense")(cat_dropout)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)
    
    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)
    
    # Model definition
    model = keras.Model(inputs = [pep, a1, a2, a3, b1, b2, b3],
                        outputs = out)
    
    return model

# Main baseline architecture for CDR3 paired-chain sequences
def CNN_CDR3_1D_baseline(dropout_rate, seed, conv_activation = "relu", dense_activation = "sigmoid",
                         embed_dim = 20, nr_of_filters_1 = 16):
    
    # Max.length of the sequences from the dataset
    a3_max = 22
    b3_max = 23
    pep_max = 12
    
    # Input dimensions
    pep = keras.Input(shape = (pep_max, embed_dim), name ="pep")
    a3 = keras.Input(shape = (a3_max, embed_dim), name ="a3")
    b3 = keras.Input(shape = (b3_max, embed_dim), name ="b3")
    
    # Trained in a pan-specific setup (padding="same" for keeping original input sizes)

    # CNN layers for each feature and the different 5 kernel-sizes
    pep_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_pep_1_conv")(pep)
    pep_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_pep_3_conv")(pep)
    pep_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_pep_5_conv")(pep)
    pep_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_pep_7_conv")(pep)
    pep_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_pep_9_conv")(pep)
    
    a3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    a3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a3_7_conv")(a3)
    a3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a3_9_conv")(a3)
    
    b3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3)
    b3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b3_7_conv")(b3)
    b3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b3_9_conv")(b3) 
    
    # GlobalMaxPooling: takes the maximum value along the entire sequence for each feature map (16 filters)
    pep_1_pool = layers.GlobalMaxPooling1D(name = "first_pep_1_pool")(pep_1_CNN)
    pep_3_pool = layers.GlobalMaxPooling1D(name = "first_pep_3_pool")(pep_3_CNN)
    pep_5_pool = layers.GlobalMaxPooling1D(name = "first_pep_5_pool")(pep_5_CNN)
    pep_7_pool = layers.GlobalMaxPooling1D(name = "first_pep_7_pool")(pep_7_CNN)
    pep_9_pool = layers.GlobalMaxPooling1D(name = "first_pep_9_pool")(pep_9_CNN)
    
    a3_1_pool = layers.GlobalMaxPooling1D(name = "first_a3_1_pool")(a3_1_CNN)
    a3_3_pool = layers.GlobalMaxPooling1D(name = "first_a3_3_pool")(a3_3_CNN)
    a3_5_pool = layers.GlobalMaxPooling1D(name = "first_a3_5_pool")(a3_5_CNN)
    a3_7_pool = layers.GlobalMaxPooling1D(name = "first_a3_7_pool")(a3_7_CNN)
    a3_9_pool = layers.GlobalMaxPooling1D(name = "first_a3_9_pool")(a3_9_CNN)
      
    b3_1_pool = layers.GlobalMaxPooling1D(name = "first_b3_1_pool")(b3_1_CNN)
    b3_3_pool = layers.GlobalMaxPooling1D(name = "first_b3_3_pool")(b3_3_CNN)
    b3_5_pool = layers.GlobalMaxPooling1D(name = "first_b3_5_pool")(b3_5_CNN)
    b3_7_pool = layers.GlobalMaxPooling1D(name = "first_b3_7_pool")(b3_7_CNN)
    b3_9_pool = layers.GlobalMaxPooling1D(name = "first_b3_9_pool")(b3_9_CNN)

    # Concatenation of all MaxPool outputs from all features and kernel-sizes
    cat = layers.Concatenate(name = "first_cat")([pep_1_pool, pep_3_pool, pep_5_pool, pep_7_pool, pep_9_pool,
                                                  a3_1_pool, a3_3_pool, a3_5_pool, a3_7_pool, a3_9_pool,
                                                  b3_1_pool, b3_3_pool, b3_5_pool, b3_7_pool, b3_9_pool])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)
    
    # Dense layers after concatenation+dropout
    dense = layers.Dense(64, activation = dense_activation, name = "first_dense")(cat_dropout)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)
    
    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)
    
    # Model definition
    model = keras.Model(inputs = [pep, a3, b3],
                        outputs = out)
    
    return model

# 'Old' redefined 2D baseline (interaction maps) architecture for CDR3 paired-chain sequences
def CNN_CDR3_2D_old_redefined(dropout_rate, n_maps, seed, conv_activation = "relu", dense_activation = "relu", nr_of_filters_1 = 16):
    
    # Max.length of the sequences from the dataset
    a3_max = 22
    b3_max = 23
    pep_max = 12

    # 2D-Input dimensions
    a3 = keras.Input(shape = (a3_max, pep_max, n_maps), name ="a3")
    b3 = keras.Input(shape = (b3_max, pep_max, n_maps), name ="b3")

    # Trained in a pan-specific setup (padding="same" for keeping original input sizes)

    # CNN layers for each feature and the different 5 kernel-sizes
    a3_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_2_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a3_2_conv")(a3)
    a3_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_4_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_a3_4_conv")(a3)
    a3_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    
    # CNN layers for each feature and the different 5 kernel-sizes
    b3_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_2_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b3_2_conv")(b3)
    b3_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_4_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_b3_4_conv")(b3)
    b3_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3) 
    
    # GlobalMaxPooling: downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (2, 2) 
    
    # a3_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_1_pool")(a3_1_CNN)
    # a3_2_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_2_pool")(a3_2_CNN)
    # a3_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_3_pool")(a3_3_CNN)
    # a3_4_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_4_pool")(a3_4_CNN)
    # a3_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_5_pool")(a3_5_CNN)
      
    # b3_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_1_pool")(b3_1_CNN)
    # b3_2_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_2_pool")(b3_2_CNN)
    # b3_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_3_pool")(b3_3_CNN)
    # b3_4_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_4_pool")(b3_4_CNN)
    # b3_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_5_pool")(b3_5_CNN)

    a3_1_pool = layers.GlobalMaxPooling2D(name = "first_a3_1_pool", keepdims=True)(a3_1_CNN)
    a3_2_pool = layers.GlobalMaxPooling2D(name = "first_a3_2_pool", keepdims=True)(a3_2_CNN)
    a3_3_pool = layers.GlobalMaxPooling2D(name = "first_a3_3_pool", keepdims=True)(a3_3_CNN)
    a3_4_pool = layers.GlobalMaxPooling2D(name = "first_a3_4_pool", keepdims=True)(a3_4_CNN)
    a3_5_pool = layers.GlobalMaxPooling2D(name = "first_a3_5_pool", keepdims=True)(a3_5_CNN)
      
    b3_1_pool = layers.GlobalMaxPooling2D(name = "first_b3_1_pool", keepdims=True)(b3_1_CNN)
    b3_2_pool = layers.GlobalMaxPooling2D(name = "first_b3_2_pool", keepdims=True)(b3_2_CNN)
    b3_3_pool = layers.GlobalMaxPooling2D(name = "first_b3_3_pool", keepdims=True)(b3_3_CNN)
    b3_4_pool = layers.GlobalMaxPooling2D(name = "first_b3_4_pool", keepdims=True)(b3_4_CNN)
    b3_5_pool = layers.GlobalMaxPooling2D(name = "first_b3_5_pool", keepdims=True)(b3_5_CNN)

    # Flatten the layers
    a3_1_flatten = layers.Flatten(name="flatten_a3_1")(a3_1_pool)
    a3_2_flatten = layers.Flatten(name="flatten_a3_2")(a3_2_pool)
    a3_3_flatten = layers.Flatten(name="flatten_a3_3")(a3_3_pool)
    a3_4_flatten = layers.Flatten(name="flatten_a3_4")(a3_4_pool)
    a3_5_flatten = layers.Flatten(name="flatten_a3_5")(a3_5_pool)

    b3_1_flatten = layers.Flatten(name="flatten_b3_1")(b3_1_pool)
    b3_2_flatten = layers.Flatten(name="flatten_b3_2")(b3_2_pool)
    b3_3_flatten = layers.Flatten(name="flatten_b3_3")(b3_3_pool)
    b3_4_flatten = layers.Flatten(name="flatten_b3_4")(b3_4_pool)
    b3_5_flatten = layers.Flatten(name="flatten_b3_5")(b3_5_pool)

    # Concatenation of all MaxPool outputs from all features and kernel-sizes
    cat = layers.Concatenate(name = "first_cat")([a3_1_flatten, a3_2_flatten, a3_3_flatten, a3_4_flatten, a3_5_flatten,
                                                  b3_1_flatten, b3_2_flatten, b3_3_flatten, b3_4_flatten, b3_5_flatten])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)
    
    # Dense layers after concatenation+dropout
    # prev_dense = layers.Dense(256, activation = dense_activation, name = "previous_dense")(cat_dropout)
    dense = layers.Dense(64, activation = dense_activation, name = "first_dense")(cat_dropout)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)
    
    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)
    
    # Model definition
    model = keras.Model(inputs = [a3, b3],
                        outputs = out)
    
    return model

# Corrected-alternative 2D baseline (interaction maps) architecture for CDR3 paired-chain sequences
def CNN_CDR3_2D_new_corrected(dropout_rate, n_maps, seed, conv_activation = "relu", dense_activation = "relu",
                              nr_of_filters_1 = 8, nr_of_filters_2 = 32, l2_reg = 0.001):

    # Max.length of the sequences from the dataset
    a3_max = 22
    b3_max = 23
    pep_max = 12

    # 2D-Input dimensions
    a3 = keras.Input(shape = (a3_max, pep_max, n_maps), name ="a3")
    b3 = keras.Input(shape = (b3_max, pep_max, n_maps), name ="b3")

    # Trained in a pan-specific setup (padding="same" for keeping original input sizes)

    # CNN layers for each feature and the different 5 kernel-sizes
    a3_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    a3_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_a3_7_conv")(a3)
    a3_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_a3_9_conv")(a3)

    # CNN layers for each feature and the different 5 kernel-sizes
    b3_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3)
    b3_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_b3_7_conv")(b3)
    b3_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_b3_9_conv")(b3)

    # Batch normalization after convolution
    a3_1_CNN = layers.BatchNormalization()(a3_1_CNN)
    a3_3_CNN = layers.BatchNormalization()(a3_3_CNN)
    a3_5_CNN = layers.BatchNormalization()(a3_5_CNN)
    a3_7_CNN = layers.BatchNormalization()(a3_7_CNN)
    a3_9_CNN = layers.BatchNormalization()(a3_9_CNN)

    b3_1_CNN = layers.BatchNormalization()(b3_1_CNN)
    b3_3_CNN = layers.BatchNormalization()(b3_3_CNN)
    b3_5_CNN = layers.BatchNormalization()(b3_5_CNN)
    b3_7_CNN = layers.BatchNormalization()(b3_7_CNN)
    b3_9_CNN = layers.BatchNormalization()(b3_9_CNN)

    # MaxPooling: downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (2, 2)
    a3_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_1_maxpool")(a3_1_CNN)
    a3_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_3_maxpool")(a3_3_CNN)
    a3_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_5_maxpool")(a3_5_CNN)
    a3_7_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_7_maxpool")(a3_7_CNN)
    a3_9_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_9_maxpool")(a3_9_CNN)

    b3_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_1_pool")(b3_1_CNN)
    b3_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_3_pool")(b3_3_CNN)
    b3_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_5_pool")(b3_5_CNN)
    b3_7_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_7_pool")(b3_7_CNN)
    b3_9_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_9_pool")(b3_9_CNN)

    # Second CNN layers for each feature and the different 5 kernel-sizes
    a3_1_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_a3_1_conv")(a3_1_pool)
    a3_3_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_a3_3_conv")(a3_3_pool)
    a3_5_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_a3_5_conv")(a3_5_pool)
    a3_7_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "second_a3_7_conv")(a3_7_pool)
    a3_9_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "second_a3_9_conv")(a3_9_pool)

    b3_1_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(1, 1), padding="same", activation=conv_activation, name="second_b3_1_conv")(b3_1_pool)
    b3_3_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(3, 3), padding="same", activation=conv_activation, name="second_b3_3_conv")(b3_3_pool)
    b3_5_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(5, 5), padding="same", activation=conv_activation, name="second_b3_5_conv")(b3_5_pool)
    b3_7_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(7, 7), padding="same", activation=conv_activation, name="second_b3_7_conv")(b3_7_pool)
    b3_9_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(9, 9), padding="same", activation=conv_activation, name="second_b3_9_conv")(b3_9_pool)

    # Final GlobalMaxPooling
    a3_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_1_globalpool", keepdims=True)(a3_1_CNN_2)
    a3_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_3_globalpool", keepdims=True)(a3_3_CNN_2)
    a3_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_5_globalpool", keepdims=True)(a3_5_CNN_2)
    a3_7_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_7_globalpool", keepdims=True)(a3_7_CNN_2)
    a3_9_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_9_globalpool", keepdims=True)(a3_9_CNN_2)

    b3_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_1_globalpool", keepdims=True)(b3_1_CNN_2)
    b3_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_3_globalpool", keepdims=True)(b3_3_CNN_2)
    b3_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_5_globalpool", keepdims=True)(b3_5_CNN_2)
    b3_7_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_7_globalpool", keepdims=True)(b3_7_CNN_2)
    b3_9_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_9_globalpool", keepdims=True)(b3_9_CNN_2)

    # Flatten the layers
    a3_1_flatten = layers.Flatten(name="flatten_a3_1")(a3_1_pool_2)
    a3_3_flatten = layers.Flatten(name="flatten_a3_3")(a3_3_pool_2)
    a3_5_flatten = layers.Flatten(name="flatten_a3_5")(a3_5_pool_2)
    a3_7_flatten = layers.Flatten(name="flatten_a3_7")(a3_7_pool_2)
    a3_9_flatten = layers.Flatten(name="flatten_a3_9")(a3_9_pool_2)

    b3_1_flatten = layers.Flatten(name="flatten_b3_1")(b3_1_pool_2)
    b3_3_flatten = layers.Flatten(name="flatten_b3_3")(b3_3_pool_2)
    b3_5_flatten = layers.Flatten(name="flatten_b3_5")(b3_5_pool_2)
    b3_7_flatten = layers.Flatten(name="flatten_b3_7")(b3_7_pool_2)
    b3_9_flatten = layers.Flatten(name="flatten_b3_9")(b3_9_pool_2)

    # Concatenation of all MaxPool outputs from all features and kernel-sizes
    cat = layers.Concatenate(name = "first_cat")([a3_1_flatten, a3_3_flatten, a3_5_flatten, a3_7_flatten, a3_9_flatten,
                                                  b3_1_flatten, b3_3_flatten, b3_5_flatten, b3_7_flatten, b3_9_flatten])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    # Dense layers after concatenation+dropout
    dense = layers.Dense(64, activation = dense_activation, kernel_regularizer = l2(l2_reg), name = "first_dense")(cat_dropout)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)

    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)

    # Model definition
    model = keras.Model(inputs = [a3, b3],
                        outputs = out)

    return model

# Corrected-alternative 2D baseline (interaction maps) architecture for CDR123 paired-chain sequences
def CNN_CDR123_2D_new_corrected(dropout_rate, n_maps, seed, conv_activation = "relu", dense_activation = "relu",
                                nr_of_filters_1 = 8, nr_of_filters_2 = 32, l2_reg = 0.001, max_lengths = None):

    # Max.length of the sequences from the dataset
    if max_lengths:
        a1_max = max_lengths[0]
        a2_max = max_lengths[1]
        a3_max = max_lengths[2]
        b1_max = max_lengths[3]
        b2_max = max_lengths[4]
        b3_max = max_lengths[5]
        pep_max = max_lengths[6]
    else:
        a1_max = 7
        a2_max = 8
        a3_max = 22
        b1_max = 6
        b2_max = 7
        b3_max = 23
        pep_max = 12

    # 2D-Input dimensions
    a1 = keras.Input(shape = (a1_max, pep_max, n_maps), name ="a1")
    b1 = keras.Input(shape = (b1_max, pep_max, n_maps), name ="b1")
    a2 = keras.Input(shape = (a2_max, pep_max, n_maps), name ="a2")
    b2 = keras.Input(shape = (b2_max, pep_max, n_maps), name ="b2")
    a3 = keras.Input(shape = (a3_max, pep_max, n_maps), name ="a3")
    b3 = keras.Input(shape = (b3_max, pep_max, n_maps), name ="b3")

    # Trained in a pan-specific setup (padding="same" for keeping original input sizes)

    # CNN layers for each feature and the different 5 kernel-sizes
    a1_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a1_1_conv")(a1)
    a1_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a1_3_conv")(a1)
    a1_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a1_5_conv")(a1)
    a1_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_a1_7_conv")(a1)
    a1_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_a1_9_conv")(a1)

    # CNN layers for each feature and the different 5 kernel-sizes
    b1_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b1_1_conv")(b1)
    b1_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b1_3_conv")(b1)
    b1_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b1_5_conv")(b1)
    b1_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_b1_7_conv")(b1)
    b1_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_b1_9_conv")(b1)

    # CNN layers for each feature and the different 5 kernel-sizes
    a2_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a2_1_conv")(a2)
    a2_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a2_3_conv")(a2)
    a2_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a2_5_conv")(a2)
    a2_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_a2_7_conv")(a2)
    a2_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_a2_9_conv")(a2)

    # CNN layers for each feature and the different 5 kernel-sizes
    b2_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b2_1_conv")(b2)
    b2_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b2_3_conv")(b2)
    b2_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b2_5_conv")(b2)
    b2_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_b2_7_conv")(b2)
    b2_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_b2_9_conv")(b2)

    # CNN layers for each feature and the different 5 kernel-sizes
    a3_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    a3_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_a3_7_conv")(a3)
    a3_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_a3_9_conv")(a3)

    # CNN layers for each feature and the different 5 kernel-sizes
    b3_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3)
    b3_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_b3_7_conv")(b3)
    b3_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_b3_9_conv")(b3)

    # Batch normalization after convolution
    a1_1_CNN = layers.BatchNormalization()(a1_1_CNN)
    a1_3_CNN = layers.BatchNormalization()(a1_3_CNN)
    a1_5_CNN = layers.BatchNormalization()(a1_5_CNN)
    a1_7_CNN = layers.BatchNormalization()(a1_7_CNN)
    a1_9_CNN = layers.BatchNormalization()(a1_9_CNN)

    b1_1_CNN = layers.BatchNormalization()(b1_1_CNN)
    b1_3_CNN = layers.BatchNormalization()(b1_3_CNN)
    b1_5_CNN = layers.BatchNormalization()(b1_5_CNN)
    b1_7_CNN = layers.BatchNormalization()(b1_7_CNN)
    b1_9_CNN = layers.BatchNormalization()(b1_9_CNN)

    a2_1_CNN = layers.BatchNormalization()(a2_1_CNN)
    a2_3_CNN = layers.BatchNormalization()(a2_3_CNN)
    a2_5_CNN = layers.BatchNormalization()(a2_5_CNN)
    a2_7_CNN = layers.BatchNormalization()(a2_7_CNN)
    a2_9_CNN = layers.BatchNormalization()(a2_9_CNN)

    b2_1_CNN = layers.BatchNormalization()(b2_1_CNN)
    b2_3_CNN = layers.BatchNormalization()(b2_3_CNN)
    b2_5_CNN = layers.BatchNormalization()(b2_5_CNN)
    b2_7_CNN = layers.BatchNormalization()(b2_7_CNN)
    b2_9_CNN = layers.BatchNormalization()(b2_9_CNN)

    a3_1_CNN = layers.BatchNormalization()(a3_1_CNN)
    a3_3_CNN = layers.BatchNormalization()(a3_3_CNN)
    a3_5_CNN = layers.BatchNormalization()(a3_5_CNN)
    a3_7_CNN = layers.BatchNormalization()(a3_7_CNN)
    a3_9_CNN = layers.BatchNormalization()(a3_9_CNN)

    b3_1_CNN = layers.BatchNormalization()(b3_1_CNN)
    b3_3_CNN = layers.BatchNormalization()(b3_3_CNN)
    b3_5_CNN = layers.BatchNormalization()(b3_5_CNN)
    b3_7_CNN = layers.BatchNormalization()(b3_7_CNN)
    b3_9_CNN = layers.BatchNormalization()(b3_9_CNN)

    # MaxPooling: downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (2, 2)
    a1_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a1_1_maxpool")(a1_1_CNN)
    a1_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a1_3_maxpool")(a1_3_CNN)
    a1_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a1_5_maxpool")(a1_5_CNN)
    a1_7_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a1_7_maxpool")(a1_7_CNN)
    a1_9_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a1_9_maxpool")(a1_9_CNN)

    b1_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b1_1_pool")(b1_1_CNN)
    b1_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b1_3_pool")(b1_3_CNN)
    b1_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b1_5_pool")(b1_5_CNN)
    b1_7_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b1_7_pool")(b1_7_CNN)
    b1_9_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b1_9_pool")(b1_9_CNN)

    a2_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a2_1_maxpool")(a2_1_CNN)
    a2_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a2_3_maxpool")(a2_3_CNN)
    a2_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a2_5_maxpool")(a2_5_CNN)
    a2_7_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a2_7_maxpool")(a2_7_CNN)
    a2_9_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a2_9_maxpool")(a2_9_CNN)

    b2_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b2_1_pool")(b2_1_CNN)
    b2_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b2_3_pool")(b2_3_CNN)
    b2_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b2_5_pool")(b2_5_CNN)
    b2_7_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b2_7_pool")(b2_7_CNN)
    b2_9_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b2_9_pool")(b2_9_CNN)

    a3_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_1_maxpool")(a3_1_CNN)
    a3_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_3_maxpool")(a3_3_CNN)
    a3_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_5_maxpool")(a3_5_CNN)
    a3_7_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_7_maxpool")(a3_7_CNN)
    a3_9_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_9_maxpool")(a3_9_CNN)

    b3_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_1_pool")(b3_1_CNN)
    b3_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_3_pool")(b3_3_CNN)
    b3_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_5_pool")(b3_5_CNN)
    b3_7_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_7_pool")(b3_7_CNN)
    b3_9_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_9_pool")(b3_9_CNN)

    # Second CNN layers for each feature and the different 5 kernel-sizes
    a1_1_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_a1_1_conv")(a1_1_pool)
    a1_3_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_a1_3_conv")(a1_3_pool)
    a1_5_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_a1_5_conv")(a1_5_pool)
    a1_7_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "second_a1_7_conv")(a1_7_pool)
    a1_9_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "second_a1_9_conv")(a1_9_pool)

    b1_1_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(1, 1), padding="same", activation=conv_activation, name="second_b1_1_conv")(b1_1_pool)
    b1_3_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(3, 3), padding="same", activation=conv_activation, name="second_b1_3_conv")(b1_3_pool)
    b1_5_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(5, 5), padding="same", activation=conv_activation, name="second_b1_5_conv")(b1_5_pool)
    b1_7_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(7, 7), padding="same", activation=conv_activation, name="second_b1_7_conv")(b1_7_pool)
    b1_9_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(9, 9), padding="same", activation=conv_activation, name="second_b1_9_conv")(b1_9_pool)

    a2_1_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_a2_1_conv")(a2_1_pool)
    a2_3_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_a2_3_conv")(a2_3_pool)
    a2_5_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_a2_5_conv")(a2_5_pool)
    a2_7_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "second_a2_7_conv")(a2_7_pool)
    a2_9_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "second_a2_9_conv")(a2_9_pool)

    b2_1_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(1, 1), padding="same", activation=conv_activation, name="second_b2_1_conv")(b2_1_pool)
    b2_3_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(3, 3), padding="same", activation=conv_activation, name="second_b2_3_conv")(b2_3_pool)
    b2_5_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(5, 5), padding="same", activation=conv_activation, name="second_b2_5_conv")(b2_5_pool)
    b2_7_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(7, 7), padding="same", activation=conv_activation, name="second_b2_7_conv")(b2_7_pool)
    b2_9_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(9, 9), padding="same", activation=conv_activation, name="second_b2_9_conv")(b2_9_pool)

    a3_1_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_a3_1_conv")(a3_1_pool)
    a3_3_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_a3_3_conv")(a3_3_pool)
    a3_5_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_a3_5_conv")(a3_5_pool)
    a3_7_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "second_a3_7_conv")(a3_7_pool)
    a3_9_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "second_a3_9_conv")(a3_9_pool)

    b3_1_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(1, 1), padding="same", activation=conv_activation, name="second_b3_1_conv")(b3_1_pool)
    b3_3_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(3, 3), padding="same", activation=conv_activation, name="second_b3_3_conv")(b3_3_pool)
    b3_5_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(5, 5), padding="same", activation=conv_activation, name="second_b3_5_conv")(b3_5_pool)
    b3_7_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(7, 7), padding="same", activation=conv_activation, name="second_b3_7_conv")(b3_7_pool)
    b3_9_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(9, 9), padding="same", activation=conv_activation, name="second_b3_9_conv")(b3_9_pool)

    # Final GlobalMaxPooling
    a1_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_a1_1_globalpool", keepdims=True)(a1_1_CNN_2)
    a1_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_a1_3_globalpool", keepdims=True)(a1_3_CNN_2)
    a1_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_a1_5_globalpool", keepdims=True)(a1_5_CNN_2)
    a1_7_pool_2 = layers.GlobalMaxPooling2D(name = "second_a1_7_globalpool", keepdims=True)(a1_7_CNN_2)
    a1_9_pool_2 = layers.GlobalMaxPooling2D(name = "second_a1_9_globalpool", keepdims=True)(a1_9_CNN_2)

    b1_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_b1_1_globalpool", keepdims=True)(b1_1_CNN_2)
    b1_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_b1_3_globalpool", keepdims=True)(b1_3_CNN_2)
    b1_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_b1_5_globalpool", keepdims=True)(b1_5_CNN_2)
    b1_7_pool_2 = layers.GlobalMaxPooling2D(name = "second_b1_7_globalpool", keepdims=True)(b1_7_CNN_2)
    b1_9_pool_2 = layers.GlobalMaxPooling2D(name = "second_b1_9_globalpool", keepdims=True)(b1_9_CNN_2)

    a2_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_a2_1_globalpool", keepdims=True)(a2_1_CNN_2)
    a2_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_a2_3_globalpool", keepdims=True)(a2_3_CNN_2)
    a2_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_a2_5_globalpool", keepdims=True)(a2_5_CNN_2)
    a2_7_pool_2 = layers.GlobalMaxPooling2D(name = "second_a2_7_globalpool", keepdims=True)(a2_7_CNN_2)
    a2_9_pool_2 = layers.GlobalMaxPooling2D(name = "second_a2_9_globalpool", keepdims=True)(a2_9_CNN_2)

    b2_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_b2_1_globalpool", keepdims=True)(b2_1_CNN_2)
    b2_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_b2_3_globalpool", keepdims=True)(b2_3_CNN_2)
    b2_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_b2_5_globalpool", keepdims=True)(b2_5_CNN_2)
    b2_7_pool_2 = layers.GlobalMaxPooling2D(name = "second_b2_7_globalpool", keepdims=True)(b2_7_CNN_2)
    b2_9_pool_2 = layers.GlobalMaxPooling2D(name = "second_b2_9_globalpool", keepdims=True)(b2_9_CNN_2)

    a3_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_1_globalpool", keepdims=True)(a3_1_CNN_2)
    a3_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_3_globalpool", keepdims=True)(a3_3_CNN_2)
    a3_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_5_globalpool", keepdims=True)(a3_5_CNN_2)
    a3_7_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_7_globalpool", keepdims=True)(a3_7_CNN_2)
    a3_9_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_9_globalpool", keepdims=True)(a3_9_CNN_2)

    b3_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_1_globalpool", keepdims=True)(b3_1_CNN_2)
    b3_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_3_globalpool", keepdims=True)(b3_3_CNN_2)
    b3_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_5_globalpool", keepdims=True)(b3_5_CNN_2)
    b3_7_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_7_globalpool", keepdims=True)(b3_7_CNN_2)
    b3_9_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_9_globalpool", keepdims=True)(b3_9_CNN_2)

    # Flatten the layers
    a1_1_flatten = layers.Flatten(name="flatten_a1_1")(a1_1_pool_2)
    a1_3_flatten = layers.Flatten(name="flatten_a1_3")(a1_3_pool_2)
    a1_5_flatten = layers.Flatten(name="flatten_a1_5")(a1_5_pool_2)
    a1_7_flatten = layers.Flatten(name="flatten_a1_7")(a1_7_pool_2)
    a1_9_flatten = layers.Flatten(name="flatten_a1_9")(a1_9_pool_2)

    b1_1_flatten = layers.Flatten(name="flatten_b1_1")(b1_1_pool_2)
    b1_3_flatten = layers.Flatten(name="flatten_b1_3")(b1_3_pool_2)
    b1_5_flatten = layers.Flatten(name="flatten_b1_5")(b1_5_pool_2)
    b1_7_flatten = layers.Flatten(name="flatten_b1_7")(b1_7_pool_2)
    b1_9_flatten = layers.Flatten(name="flatten_b1_9")(b1_9_pool_2)

    a2_1_flatten = layers.Flatten(name="flatten_a2_1")(a2_1_pool_2)
    a2_3_flatten = layers.Flatten(name="flatten_a2_3")(a2_3_pool_2)
    a2_5_flatten = layers.Flatten(name="flatten_a2_5")(a2_5_pool_2)
    a2_7_flatten = layers.Flatten(name="flatten_a2_7")(a2_7_pool_2)
    a2_9_flatten = layers.Flatten(name="flatten_a2_9")(a2_9_pool_2)

    b2_1_flatten = layers.Flatten(name="flatten_b2_1")(b2_1_pool_2)
    b2_3_flatten = layers.Flatten(name="flatten_b2_3")(b2_3_pool_2)
    b2_5_flatten = layers.Flatten(name="flatten_b2_5")(b2_5_pool_2)
    b2_7_flatten = layers.Flatten(name="flatten_b2_7")(b2_7_pool_2)
    b2_9_flatten = layers.Flatten(name="flatten_b2_9")(b2_9_pool_2)

    a3_1_flatten = layers.Flatten(name="flatten_a3_1")(a3_1_pool_2)
    a3_3_flatten = layers.Flatten(name="flatten_a3_3")(a3_3_pool_2)
    a3_5_flatten = layers.Flatten(name="flatten_a3_5")(a3_5_pool_2)
    a3_7_flatten = layers.Flatten(name="flatten_a3_7")(a3_7_pool_2)
    a3_9_flatten = layers.Flatten(name="flatten_a3_9")(a3_9_pool_2)

    b3_1_flatten = layers.Flatten(name="flatten_b3_1")(b3_1_pool_2)
    b3_3_flatten = layers.Flatten(name="flatten_b3_3")(b3_3_pool_2)
    b3_5_flatten = layers.Flatten(name="flatten_b3_5")(b3_5_pool_2)
    b3_7_flatten = layers.Flatten(name="flatten_b3_7")(b3_7_pool_2)
    b3_9_flatten = layers.Flatten(name="flatten_b3_9")(b3_9_pool_2)

    # Concatenation of all MaxPool outputs from all features and kernel-sizes
    cat = layers.Concatenate(name = "first_cat")([a1_1_flatten, a1_3_flatten, a1_5_flatten, a1_7_flatten, a1_9_flatten,
                                                  b1_1_flatten, b1_3_flatten, b1_5_flatten, b1_7_flatten, b1_9_flatten,
                                                  a2_1_flatten, a2_3_flatten, a2_5_flatten, a2_7_flatten, a2_9_flatten,
                                                  b2_1_flatten, b2_3_flatten, b2_5_flatten, b2_7_flatten, b2_9_flatten,
                                                  a3_1_flatten, a3_3_flatten, a3_5_flatten, a3_7_flatten, a3_9_flatten,
                                                  b3_1_flatten, b3_3_flatten, b3_5_flatten, b3_7_flatten, b3_9_flatten])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    # Dense layers after concatenation+dropout
    dense = layers.Dense(64, activation = dense_activation, kernel_regularizer = l2(l2_reg), name = "first_dense")(cat_dropout)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)

    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)

    # Model definition
    model = keras.Model(inputs = [a1, b1, a2, b2, a3, b3],
                        outputs = out)

    return model

# Corrected-alternative 2D baseline (interaction maps) architecture for CDR123 paired-chain sequences
def CNN_CDR123_opt2D_new_corrected(dropout_rate, n_maps, seed, conv_activation = "relu", dense_activation = "relu",
                                   nr_of_filters_1 = 8, nr_of_filters_2 = 32, l2_reg = 0.001, max_lengths = None):

    # Max.length of the sequences from the dataset
    if max_lengths:
        a1_max = max_lengths[0]
        a2_max = max_lengths[1]
        a3_max = max_lengths[2]
        b1_max = max_lengths[3]
        b2_max = max_lengths[4]
        b3_max = max_lengths[5]
        pep_max = max_lengths[6]
    else:
        a1_max = 7
        a2_max = 8
        a3_max = 22
        b1_max = 6
        b2_max = 7
        b3_max = 23
        pep_max = 12

    # 2D-Input dimensions
    a1 = keras.Input(shape = (a1_max, pep_max, n_maps), name ="a1")
    b1 = keras.Input(shape = (b1_max, pep_max, n_maps), name ="b1")
    a2 = keras.Input(shape = (a2_max, pep_max, n_maps), name ="a2")
    b2 = keras.Input(shape = (b2_max, pep_max, n_maps), name ="b2")
    a3 = keras.Input(shape = (a3_max, pep_max, n_maps), name ="a3")
    b3 = keras.Input(shape = (b3_max, pep_max, n_maps), name ="b3")

    # Trained in a pan-specific setup (padding="same" for keeping original input sizes)

    # CNN layers for each feature and the different 5 kernel-sizes
    a1_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a1_1_conv")(a1)
    a1_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a1_3_conv")(a1)
    a1_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a1_5_conv")(a1)

    # CNN layers for each feature and the different 5 kernel-sizes
    b1_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b1_1_conv")(b1)
    b1_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b1_3_conv")(b1)
    b1_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b1_5_conv")(b1)

    # CNN layers for each feature and the different 5 kernel-sizes
    a2_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a2_1_conv")(a2)
    a2_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a2_3_conv")(a2)
    a2_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a2_5_conv")(a2)

    # CNN layers for each feature and the different 5 kernel-sizes
    b2_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b2_1_conv")(b2)
    b2_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b2_3_conv")(b2)
    b2_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b2_5_conv")(a2)

    # CNN layers for each feature and the different 5 kernel-sizes
    a3_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    a3_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_a3_7_conv")(a3)
    a3_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_a3_9_conv")(a3)

    # CNN layers for each feature and the different 5 kernel-sizes
    b3_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3)
    b3_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_b3_7_conv")(b3)
    b3_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_b3_9_conv")(b3)

    # Batch normalization after convolution
    a1_1_CNN = layers.BatchNormalization()(a1_1_CNN)
    a1_3_CNN = layers.BatchNormalization()(a1_3_CNN)
    a1_5_CNN = layers.BatchNormalization()(a1_5_CNN)

    b1_1_CNN = layers.BatchNormalization()(b1_1_CNN)
    b1_3_CNN = layers.BatchNormalization()(b1_3_CNN)
    b1_5_CNN = layers.BatchNormalization()(b1_5_CNN)

    a2_1_CNN = layers.BatchNormalization()(a2_1_CNN)
    a2_3_CNN = layers.BatchNormalization()(a2_3_CNN)
    a2_5_CNN = layers.BatchNormalization()(a2_5_CNN)

    b2_1_CNN = layers.BatchNormalization()(b2_1_CNN)
    b2_3_CNN = layers.BatchNormalization()(b2_3_CNN)
    b2_5_CNN = layers.BatchNormalization()(b2_5_CNN)

    a3_1_CNN = layers.BatchNormalization()(a3_1_CNN)
    a3_3_CNN = layers.BatchNormalization()(a3_3_CNN)
    a3_5_CNN = layers.BatchNormalization()(a3_5_CNN)
    a3_7_CNN = layers.BatchNormalization()(a3_7_CNN)
    a3_9_CNN = layers.BatchNormalization()(a3_9_CNN)

    b3_1_CNN = layers.BatchNormalization()(b3_1_CNN)
    b3_3_CNN = layers.BatchNormalization()(b3_3_CNN)
    b3_5_CNN = layers.BatchNormalization()(b3_5_CNN)
    b3_7_CNN = layers.BatchNormalization()(b3_7_CNN)
    b3_9_CNN = layers.BatchNormalization()(b3_9_CNN)

    # MaxPooling: downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (2, 2)
    a1_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a1_1_maxpool")(a1_1_CNN)
    a1_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a1_3_maxpool")(a1_3_CNN)
    a1_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a1_5_maxpool")(a1_5_CNN)

    b1_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b1_1_maxpool")(b1_1_CNN)
    b1_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b1_3_maxpool")(b1_3_CNN)
    b1_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b1_5_maxpool")(b1_5_CNN)

    a2_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a2_1_maxpool")(a2_1_CNN)
    a2_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a2_3_maxpool")(a2_3_CNN)
    a2_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a2_5_maxpool")(a2_5_CNN)

    b2_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b2_1_maxpool")(b2_1_CNN)
    b2_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b2_3_maxpool")(b2_3_CNN)
    b2_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b2_5_maxpool")(b2_5_CNN)

    a3_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_1_maxpool")(a3_1_CNN)
    a3_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_3_maxpool")(a3_3_CNN)
    a3_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_5_maxpool")(a3_5_CNN)
    a3_7_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_7_maxpool")(a3_7_CNN)
    a3_9_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_9_maxpool")(a3_9_CNN)

    b3_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_1_maxpool")(b3_1_CNN)
    b3_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_3_maxpool")(b3_3_CNN)
    b3_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_5_maxpool")(b3_5_CNN)
    b3_7_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_7_maxpool")(b3_7_CNN)
    b3_9_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_9_maxpool")(b3_9_CNN)

    # Second CNN layers for each feature and the different 5 kernel-sizes
    a1_1_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_a1_1_conv")(a1_1_pool)
    a1_3_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_a1_3_conv")(a1_3_pool)
    a1_5_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_a1_5_conv")(a1_5_pool)

    b1_1_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(1, 1), padding="same", activation=conv_activation, name="second_b1_1_conv")(b1_1_pool)
    b1_3_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(3, 3), padding="same", activation=conv_activation, name="second_b1_3_conv")(b1_3_pool)
    b1_5_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_b1_5_conv")(b1_5_pool)
    
    a2_1_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_a2_1_conv")(a2_1_pool)
    a2_3_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_a2_3_conv")(a2_3_pool)
    a2_5_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_a2_5_conv")(a2_5_pool)

    b2_1_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(1, 1), padding="same", activation=conv_activation, name="second_b2_1_conv")(b2_1_pool)
    b2_3_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(3, 3), padding="same", activation=conv_activation, name="second_b2_3_conv")(b2_3_pool)
    b2_5_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_b2_5_conv")(b2_5_pool)

    a3_1_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_a3_1_conv")(a3_1_pool)
    a3_3_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_a3_3_conv")(a3_3_pool)
    a3_5_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_a3_5_conv")(a3_5_pool)
    a3_7_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "second_a3_7_conv")(a3_7_pool)
    a3_9_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "second_a3_9_conv")(a3_9_pool)

    b3_1_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(1, 1), padding="same", activation=conv_activation, name="second_b3_1_conv")(b3_1_pool)
    b3_3_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(3, 3), padding="same", activation=conv_activation, name="second_b3_3_conv")(b3_3_pool)
    b3_5_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(5, 5), padding="same", activation=conv_activation, name="second_b3_5_conv")(b3_5_pool)
    b3_7_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "second_b3_7_conv")(b3_7_pool)
    b3_9_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "second_b3_9_conv")(b3_9_pool)

    # Final GlobalMaxPooling
    a1_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_a1_1_globalpool", keepdims=True)(a1_1_CNN_2)
    a1_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_a1_3_globalpool", keepdims=True)(a1_3_CNN_2)
    a1_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_a1_5_globalpool", keepdims=True)(a1_5_CNN_2)

    b1_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_b1_1_globalpool", keepdims=True)(b1_1_CNN_2)
    b1_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_b1_3_globalpool", keepdims=True)(b1_3_CNN_2)
    b1_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_b1_5_globalpool", keepdims=True)(b1_5_CNN_2)

    a2_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_a2_1_globalpool", keepdims=True)(a2_1_CNN_2)
    a2_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_a2_3_globalpool", keepdims=True)(a2_3_CNN_2)
    a2_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_a2_5_globalpool", keepdims=True)(a2_5_CNN_2)

    b2_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_b2_1_globalpool", keepdims=True)(b2_1_CNN_2)
    b2_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_b2_3_globalpool", keepdims=True)(b2_3_CNN_2)
    b2_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_b2_5_globalpool", keepdims=True)(b2_5_CNN_2)

    a3_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_1_globalpool", keepdims=True)(a3_1_CNN_2)
    a3_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_3_globalpool", keepdims=True)(a3_3_CNN_2)
    a3_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_5_globalpool", keepdims=True)(a3_5_CNN_2)
    a3_7_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_7_globalpool", keepdims=True)(a3_7_CNN_2)
    a3_9_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_9_globalpool", keepdims=True)(a3_9_CNN_2)

    b3_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_1_globalpool", keepdims=True)(b3_1_CNN_2)
    b3_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_3_globalpool", keepdims=True)(b3_3_CNN_2)
    b3_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_5_globalpool", keepdims=True)(b3_5_CNN_2)
    b3_7_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_7_globalpool", keepdims=True)(b3_7_CNN_2)
    b3_9_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_9_globalpool", keepdims=True)(b3_9_CNN_2)

    # Flatten the layers
    a1_1_flatten = layers.Flatten(name="flatten_a1_1")(a1_1_pool_2)
    a1_3_flatten = layers.Flatten(name="flatten_a1_3")(a1_3_pool_2)
    a1_5_flatten = layers.Flatten(name="flatten_a1_5")(a1_5_pool_2)

    b1_1_flatten = layers.Flatten(name="flatten_b1_1")(b1_1_pool_2)
    b1_3_flatten = layers.Flatten(name="flatten_b1_3")(b1_3_pool_2)
    b1_5_flatten = layers.Flatten(name="flatten_b1_5")(b1_5_pool_2)

    a2_1_flatten = layers.Flatten(name="flatten_a2_1")(a2_1_pool_2)
    a2_3_flatten = layers.Flatten(name="flatten_a2_3")(a2_3_pool_2)
    a2_5_flatten = layers.Flatten(name="flatten_a2_5")(a2_5_pool_2)

    b2_1_flatten = layers.Flatten(name="flatten_b2_1")(b2_1_pool_2)
    b2_3_flatten = layers.Flatten(name="flatten_b2_3")(b2_3_pool_2)
    b2_5_flatten = layers.Flatten(name="flatten_b2_5")(b2_5_pool_2)

    a3_1_flatten = layers.Flatten(name="flatten_a3_1")(a3_1_pool_2)
    a3_3_flatten = layers.Flatten(name="flatten_a3_3")(a3_3_pool_2)
    a3_5_flatten = layers.Flatten(name="flatten_a3_5")(a3_5_pool_2)
    a3_7_flatten = layers.Flatten(name="flatten_a3_7")(a3_7_pool_2)
    a3_9_flatten = layers.Flatten(name="flatten_a3_9")(a3_9_pool_2)

    b3_1_flatten = layers.Flatten(name="flatten_b3_1")(b3_1_pool_2)
    b3_3_flatten = layers.Flatten(name="flatten_b3_3")(b3_3_pool_2)
    b3_5_flatten = layers.Flatten(name="flatten_b3_5")(b3_5_pool_2)
    b3_7_flatten = layers.Flatten(name="flatten_b3_7")(b3_7_pool_2)
    b3_9_flatten = layers.Flatten(name="flatten_b3_9")(b3_9_pool_2)

    # Concatenation of all MaxPool outputs from all features and kernel-sizes
    cat = layers.Concatenate(name = "first_cat")([a1_1_flatten, a1_3_flatten, a1_5_flatten, 
                                                  b1_1_flatten, b1_3_flatten, b1_5_flatten, 
                                                  a2_1_flatten, a2_3_flatten, a2_5_flatten, 
                                                  b2_1_flatten, b2_3_flatten, b2_5_flatten, 
                                                  a3_1_flatten, a3_3_flatten, a3_5_flatten, a3_7_flatten, a3_9_flatten, 
                                                  b3_1_flatten, b3_3_flatten, b3_5_flatten, b3_7_flatten, b3_9_flatten])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    # Dense layers after concatenation+dropout
    dense = layers.Dense(64, activation = dense_activation, kernel_regularizer = l2(l2_reg), name = "first_dense")(cat_dropout)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)

    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)

    # Model definition
    model = keras.Model(inputs = [a1, b1, a2, b2, a3, b3],
                        outputs = out)

    return model

# Corrected-alternative 2D baseline (interaction maps) architecture for CDR3 paired-chain sequences
def CNN_CDR3_opt2D_new_corrected(dropout_rate, n_maps, seed, conv_activation = "relu", dense_activation = "relu",
                                 nr_of_filters_1 = 8, nr_of_filters_2 = 32, l2_reg = 0.001, max_lengths = None):

    # Max.length of the sequences from the dataset
    if max_lengths:
        a3_max = max_lengths[2]
        b3_max = max_lengths[5]
        pep_max = max_lengths[6]
    else:
        a3_max = 22
        b3_max = 23
        pep_max = 12

    # 2D-Input dimensions
    a3 = keras.Input(shape = (a3_max, pep_max, n_maps), name ="a3")
    b3 = keras.Input(shape = (b3_max, pep_max, n_maps), name ="b3")

    # Trained in a pan-specific setup (padding="same" for keeping original input sizes)

    # CNN layers for each feature and the different 5 kernel-sizes
    a3_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    a3_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_a3_7_conv")(a3)
    a3_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_a3_9_conv")(a3)

    # CNN layers for each feature and the different 5 kernel-sizes
    b3_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3)
    b3_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_b3_7_conv")(b3)
    b3_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_b3_9_conv")(b3)

    # Batch normalization after convolution
    a3_1_CNN = layers.BatchNormalization()(a3_1_CNN)
    a3_3_CNN = layers.BatchNormalization()(a3_3_CNN)
    a3_5_CNN = layers.BatchNormalization()(a3_5_CNN)
    a3_7_CNN = layers.BatchNormalization()(a3_7_CNN)
    a3_9_CNN = layers.BatchNormalization()(a3_9_CNN)

    b3_1_CNN = layers.BatchNormalization()(b3_1_CNN)
    b3_3_CNN = layers.BatchNormalization()(b3_3_CNN)
    b3_5_CNN = layers.BatchNormalization()(b3_5_CNN)
    b3_7_CNN = layers.BatchNormalization()(b3_7_CNN)
    b3_9_CNN = layers.BatchNormalization()(b3_9_CNN)

    # MaxPooling: downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (2, 2)
    a3_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_1_maxpool")(a3_1_CNN)
    a3_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_3_maxpool")(a3_3_CNN)
    a3_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_5_maxpool")(a3_5_CNN)
    a3_7_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_7_maxpool")(a3_7_CNN)
    a3_9_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3_9_maxpool")(a3_9_CNN)

    b3_1_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_1_maxpool")(b3_1_CNN)
    b3_3_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_3_maxpool")(b3_3_CNN)
    b3_5_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_5_maxpool")(b3_5_CNN)
    b3_7_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_7_maxpool")(b3_7_CNN)
    b3_9_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3_9_maxpool")(b3_9_CNN)

    # Second CNN layers for each feature and the different 5 kernel-sizes
    a3_1_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_a3_1_conv")(a3_1_pool)
    a3_3_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_a3_3_conv")(a3_3_pool)
    a3_5_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_a3_5_conv")(a3_5_pool)
    a3_7_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "second_a3_7_conv")(a3_7_pool)
    a3_9_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "second_a3_9_conv")(a3_9_pool)

    b3_1_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(1, 1), padding="same", activation=conv_activation, name="second_b3_1_conv")(b3_1_pool)
    b3_3_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(3, 3), padding="same", activation=conv_activation, name="second_b3_3_conv")(b3_3_pool)
    b3_5_CNN_2 = layers.Conv2D(filters=nr_of_filters_2, kernel_size=(5, 5), padding="same", activation=conv_activation, name="second_b3_5_conv")(b3_5_pool)
    b3_7_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "second_b3_7_conv")(b3_7_pool)
    b3_9_CNN_2 = layers.Conv2D(filters = nr_of_filters_2, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "second_b3_9_conv")(b3_9_pool)

    # Final GlobalMaxPooling
    a3_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_1_globalpool", keepdims=True)(a3_1_CNN_2)
    a3_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_3_globalpool", keepdims=True)(a3_3_CNN_2)
    a3_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_5_globalpool", keepdims=True)(a3_5_CNN_2)
    a3_7_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_7_globalpool", keepdims=True)(a3_7_CNN_2)
    a3_9_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_9_globalpool", keepdims=True)(a3_9_CNN_2)

    b3_1_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_1_globalpool", keepdims=True)(b3_1_CNN_2)
    b3_3_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_3_globalpool", keepdims=True)(b3_3_CNN_2)
    b3_5_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_5_globalpool", keepdims=True)(b3_5_CNN_2)
    b3_7_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_7_globalpool", keepdims=True)(b3_7_CNN_2)
    b3_9_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_9_globalpool", keepdims=True)(b3_9_CNN_2)

    # Flatten the layers
    a3_1_flatten = layers.Flatten(name="flatten_a3_1")(a3_1_pool_2)
    a3_3_flatten = layers.Flatten(name="flatten_a3_3")(a3_3_pool_2)
    a3_5_flatten = layers.Flatten(name="flatten_a3_5")(a3_5_pool_2)
    a3_7_flatten = layers.Flatten(name="flatten_a3_7")(a3_7_pool_2)
    a3_9_flatten = layers.Flatten(name="flatten_a3_9")(a3_9_pool_2)

    b3_1_flatten = layers.Flatten(name="flatten_b3_1")(b3_1_pool_2)
    b3_3_flatten = layers.Flatten(name="flatten_b3_3")(b3_3_pool_2)
    b3_5_flatten = layers.Flatten(name="flatten_b3_5")(b3_5_pool_2)
    b3_7_flatten = layers.Flatten(name="flatten_b3_7")(b3_7_pool_2)
    b3_9_flatten = layers.Flatten(name="flatten_b3_9")(b3_9_pool_2)

    # Concatenation of all MaxPool outputs from all features and kernel-sizes
    cat = layers.Concatenate(name = "first_cat")([a3_1_flatten, a3_3_flatten, a3_5_flatten, a3_7_flatten, a3_9_flatten, 
                                                  b3_1_flatten, b3_3_flatten, b3_5_flatten, b3_7_flatten, b3_9_flatten])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    # Dense layers after concatenation+dropout
    dense = layers.Dense(64, activation = dense_activation, kernel_regularizer = l2(l2_reg), name = "first_dense")(cat_dropout)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)

    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)

    # Model definition
    model = keras.Model(inputs = [a3, b3],
                        outputs = out)

    return model

# Mixed architectures fusing 1D and 2D models
def CNN_mixed_1D_2D(n_maps, seed, dropout_rate = 0.6, conv_activation = "relu", dense_activation = "relu", 
                    embed_dim = 20, nr_of_filters_1D = 32, nr_of_filters_1_2D = 8, nr_of_filters_2_2D = 32, l2_reg = 0.001, max_lengths = None):

    # Max.length of the sequences from the dataset
    if max_lengths:
        a1_max = max_lengths[0]
        a2_max = max_lengths[1]
        a3_max = max_lengths[2]
        b1_max = max_lengths[3]
        b2_max = max_lengths[4]
        b3_max = max_lengths[5]
        pep_max = max_lengths[6]
    else:
        a1_max = 7
        a2_max = 8
        a3_max = 22
        b1_max = 6
        b2_max = 7
        b3_max = 23
        pep_max = 12

    # 2D-Input dimensions
    a1pep = keras.Input(shape = (a1_max, pep_max, n_maps), name ="a1pep")
    b1pep = keras.Input(shape = (b1_max, pep_max, n_maps), name ="b1pep")
    a2pep = keras.Input(shape = (a2_max, pep_max, n_maps), name ="a2pep")
    b2pep = keras.Input(shape = (b2_max, pep_max, n_maps), name ="b2pep")
    a3pep = keras.Input(shape = (a3_max, pep_max, n_maps), name ="a3pep")
    b3pep = keras.Input(shape = (b3_max, pep_max, n_maps), name ="b3pep")

    # 1D-Input dimensions
    pep = keras.Input(shape = (pep_max, embed_dim), name ="pep")
    a1 = keras.Input(shape = (a1_max, embed_dim), name ="a1")
    a2 = keras.Input(shape = (a2_max, embed_dim), name ="a2")
    a3 = keras.Input(shape = (a3_max, embed_dim), name ="a3")
    b1 = keras.Input(shape = (b1_max, embed_dim), name ="b1")
    b2 = keras.Input(shape = (b2_max, embed_dim), name ="b2")
    b3 = keras.Input(shape = (b3_max, embed_dim), name ="b3")
    
    # 1D Convolutional Layers Architecture

    pep_1_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_pep_1_1D_conv")(pep)
    pep_3_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_pep_3_1D_conv")(pep)
    pep_5_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_pep_5_1D_conv")(pep)
    pep_7_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_pep_7_1D_conv")(pep)
    pep_9_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_pep_9_1D_conv")(pep)
    
    a1_1_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a1_1_1D_conv")(a1)
    a1_3_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a1_3_1D_conv")(a1)
    a1_5_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a1_5_1D_conv")(a1)
    a1_7_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a1_7_1D_conv")(a1)
    a1_9_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a1_9_1D_conv")(a1)
    
    a2_1_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a2_1_1D_conv")(a2)
    a2_3_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a2_3_1D_conv")(a2)
    a2_5_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a2_5_1D_conv")(a2)
    a2_7_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a2_7_1D_conv")(a2)
    a2_9_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a2_9_1D_conv")(a2)
    
    a3_1_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a3_1_1D_conv")(a3)
    a3_3_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a3_3_1D_conv")(a3)
    a3_5_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a3_5_1D_conv")(a3)
    a3_7_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a3_7_1D_conv")(a3)
    a3_9_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a3_9_1D_conv")(a3)
    
    b1_1_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b1_1_1D_conv")(b1)
    b1_3_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b1_3_1D_conv")(b1)
    b1_5_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b1_5_1D_conv")(b1)
    b1_7_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b1_7_1D_conv")(b1)
    b1_9_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b1_9_1D_conv")(b1)
    
    b2_1_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b2_1_1D_conv")(b2)
    b2_3_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b2_3_1D_conv")(b2)
    b2_5_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b2_5_1D_conv")(b2)
    b2_7_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b2_7_1D_conv")(b2)
    b2_9_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b2_9_1D_conv")(b2)
    
    b3_1_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b3_1_1D_conv")(b3)
    b3_3_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b3_3_1D_conv")(b3)
    b3_5_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b3_5_1D_conv")(b3)
    b3_7_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b3_7_1D_conv")(b3)
    b3_9_1D_CNN = layers.Conv1D(filters = nr_of_filters_1D, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b3_9_1D_conv")(b3) 
    
    # GlobalMaxPooling: takes the maximum value along the entire sequence for each feature map (16 filters)
    pep_1_1D_pool = layers.GlobalMaxPooling1D(name = "first_pep_1_1D_pool")(pep_1_1D_CNN)
    pep_3_1D_pool = layers.GlobalMaxPooling1D(name = "first_pep_3_1D_pool")(pep_3_1D_CNN)
    pep_5_1D_pool = layers.GlobalMaxPooling1D(name = "first_pep_5_1D_pool")(pep_5_1D_CNN)
    pep_7_1D_pool = layers.GlobalMaxPooling1D(name = "first_pep_7_1D_pool")(pep_7_1D_CNN)
    pep_9_1D_pool = layers.GlobalMaxPooling1D(name = "first_pep_9_1D_pool")(pep_9_1D_CNN)
    
    a1_1_1D_pool = layers.GlobalMaxPooling1D(name = "first_a1_1_1D_pool")(a1_1_1D_CNN)
    a1_3_1D_pool = layers.GlobalMaxPooling1D(name = "first_a1_3_1D_pool")(a1_3_1D_CNN)
    a1_5_1D_pool = layers.GlobalMaxPooling1D(name = "first_a1_5_1D_pool")(a1_5_1D_CNN)
    a1_7_1D_pool = layers.GlobalMaxPooling1D(name = "first_a1_7_1D_pool")(a1_7_1D_CNN)
    a1_9_1D_pool = layers.GlobalMaxPooling1D(name = "first_a1_9_1D_pool")(a1_9_1D_CNN)
    
    a2_1_1D_pool = layers.GlobalMaxPooling1D(name = "first_a2_1_1D_pool")(a2_1_1D_CNN)
    a2_3_1D_pool = layers.GlobalMaxPooling1D(name = "first_a2_3_1D_pool")(a2_3_1D_CNN)
    a2_5_1D_pool = layers.GlobalMaxPooling1D(name = "first_a2_5_1D_pool")(a2_5_1D_CNN)
    a2_7_1D_pool = layers.GlobalMaxPooling1D(name = "first_a2_7_1D_pool")(a2_7_1D_CNN)
    a2_9_1D_pool = layers.GlobalMaxPooling1D(name = "first_a2_9_1D_pool")(a2_9_1D_CNN)
    
    a3_1_1D_pool = layers.GlobalMaxPooling1D(name = "first_a3_1_1D_pool")(a3_1_1D_CNN)
    a3_3_1D_pool = layers.GlobalMaxPooling1D(name = "first_a3_3_1D_pool")(a3_3_1D_CNN)
    a3_5_1D_pool = layers.GlobalMaxPooling1D(name = "first_a3_5_1D_pool")(a3_5_1D_CNN)
    a3_7_1D_pool = layers.GlobalMaxPooling1D(name = "first_a3_7_1D_pool")(a3_7_1D_CNN)
    a3_9_1D_pool = layers.GlobalMaxPooling1D(name = "first_a3_9_1D_pool")(a3_9_1D_CNN)
    
    b1_1_1D_pool = layers.GlobalMaxPooling1D(name = "first_b1_1_1D_pool")(b1_1_1D_CNN)
    b1_3_1D_pool = layers.GlobalMaxPooling1D(name = "first_b1_3_1D_pool")(b1_3_1D_CNN)
    b1_5_1D_pool = layers.GlobalMaxPooling1D(name = "first_b1_5_1D_pool")(b1_5_1D_CNN)
    b1_7_1D_pool = layers.GlobalMaxPooling1D(name = "first_b1_7_1D_pool")(b1_7_1D_CNN)
    b1_9_1D_pool = layers.GlobalMaxPooling1D(name = "first_b1_9_1D_pool")(b1_9_1D_CNN)
    
    b2_1_1D_pool = layers.GlobalMaxPooling1D(name = "first_b2_1_1D_pool")(b2_1_1D_CNN)
    b2_3_1D_pool = layers.GlobalMaxPooling1D(name = "first_b2_3_1D_pool")(b2_3_1D_CNN)
    b2_5_1D_pool = layers.GlobalMaxPooling1D(name = "first_b2_5_1D_pool")(b2_5_1D_CNN)
    b2_7_1D_pool = layers.GlobalMaxPooling1D(name = "first_b2_7_1D_pool")(b2_7_1D_CNN)
    b2_9_1D_pool = layers.GlobalMaxPooling1D(name = "first_b2_9_1D_pool")(b2_9_1D_CNN)
    
    b3_1_1D_pool = layers.GlobalMaxPooling1D(name = "first_b3_1_1D_pool")(b3_1_1D_CNN)
    b3_3_1D_pool = layers.GlobalMaxPooling1D(name = "first_b3_3_1D_pool")(b3_3_1D_CNN)
    b3_5_1D_pool = layers.GlobalMaxPooling1D(name = "first_b3_5_1D_pool")(b3_5_1D_CNN)
    b3_7_1D_pool = layers.GlobalMaxPooling1D(name = "first_b3_7_1D_pool")(b3_7_1D_CNN)
    b3_9_1D_pool = layers.GlobalMaxPooling1D(name = "first_b3_9_1D_pool")(b3_9_1D_CNN)

    # 2D Convolutional Layers Structure

    a1pep_1_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a1pep_1_2D_conv")(a1pep)
    a1pep_3_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a1pep_3_2D_conv")(a1pep)
    a1pep_5_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a1pep_5_2D_conv")(a1pep)
    
    b1pep_1_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b1pep_1_2D_conv")(b1pep)
    b1pep_3_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b1pep_3_2D_conv")(b1pep)
    b1pep_5_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b1pep_5_2D_conv")(b1pep)

    a2pep_1_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a2pep_1_2D_conv")(a2pep)
    a2pep_3_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a2pep_3_2D_conv")(a2pep)
    a2pep_5_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a2pep_5_2D_conv")(a2pep)
    
    b2pep_1_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b2pep_1_2D_conv")(b2pep)
    b2pep_3_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b2pep_3_2D_conv")(b2pep)
    b2pep_5_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b2pep_5_2D_conv")(b2pep)

    a3pep_1_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a3pep_1_2D_conv")(a3pep)
    a3pep_3_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a3pep_3_2D_conv")(a3pep)
    a3pep_5_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a3pep_5_2D_conv")(a3pep)
    a3pep_7_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_a3pep_7_2D_conv")(a3pep)
    a3pep_9_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_a3pep_9_2D_conv")(a3pep)

    b3pep_1_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b3pep_1_2D_conv")(b3pep)
    b3pep_3_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b3pep_3_2D_conv")(b3pep)
    b3pep_5_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b3pep_5_2D_conv")(b3pep)
    b3pep_7_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_b3pep_7_2D_conv")(b3pep)
    b3pep_9_2D_CNN = layers.Conv2D(filters = nr_of_filters_1_2D, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_b3pep_9_2D_conv")(b3pep)

    # Batch normalization after convolution
    a1pep_1_2D_CNN = layers.BatchNormalization()(a1pep_1_2D_CNN)
    a1pep_3_2D_CNN = layers.BatchNormalization()(a1pep_3_2D_CNN)
    a1pep_5_2D_CNN = layers.BatchNormalization()(a1pep_5_2D_CNN)

    b1pep_1_2D_CNN = layers.BatchNormalization()(b1pep_1_2D_CNN)
    b1pep_3_2D_CNN = layers.BatchNormalization()(b1pep_3_2D_CNN)
    b1pep_5_2D_CNN = layers.BatchNormalization()(b1pep_5_2D_CNN)

    a2pep_1_2D_CNN = layers.BatchNormalization()(a2pep_1_2D_CNN)
    a2pep_3_2D_CNN = layers.BatchNormalization()(a2pep_3_2D_CNN)
    a2pep_5_2D_CNN = layers.BatchNormalization()(a2pep_5_2D_CNN)

    b2pep_1_2D_CNN = layers.BatchNormalization()(b2pep_1_2D_CNN)
    b2pep_3_2D_CNN = layers.BatchNormalization()(b2pep_3_2D_CNN)
    b2pep_5_2D_CNN = layers.BatchNormalization()(b2pep_5_2D_CNN)

    a3pep_1_2D_CNN = layers.BatchNormalization()(a3pep_1_2D_CNN)
    a3pep_3_2D_CNN = layers.BatchNormalization()(a3pep_3_2D_CNN)
    a3pep_5_2D_CNN = layers.BatchNormalization()(a3pep_5_2D_CNN)
    a3pep_7_2D_CNN = layers.BatchNormalization()(a3pep_7_2D_CNN)
    a3pep_9_2D_CNN = layers.BatchNormalization()(a3pep_9_2D_CNN)

    b3pep_1_2D_CNN = layers.BatchNormalization()(b3pep_1_2D_CNN)
    b3pep_3_2D_CNN = layers.BatchNormalization()(b3pep_3_2D_CNN)
    b3pep_5_2D_CNN = layers.BatchNormalization()(b3pep_5_2D_CNN)
    b3pep_7_2D_CNN = layers.BatchNormalization()(b3pep_7_2D_CNN)
    b3pep_9_2D_CNN = layers.BatchNormalization()(b3pep_9_2D_CNN)


    # MaxPooling: downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (2, 2)
    a1pep_1_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a1pep_1_2D_maxpool")(a1pep_1_2D_CNN)
    a1pep_3_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a1pep_3_2D_maxpool")(a1pep_3_2D_CNN)
    a1pep_5_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a1pep_5_2D_maxpool")(a1pep_5_2D_CNN)

    b1pep_1_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b1pep_1_2D_maxpool")(b1pep_1_2D_CNN)
    b1pep_3_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b1pep_3_2D_maxpool")(b1pep_3_2D_CNN)
    b1pep_5_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b1pep_5_2D_maxpool")(b1pep_5_2D_CNN)

    a2pep_1_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a2pep_1_2D_maxpool")(a2pep_1_2D_CNN)
    a2pep_3_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a2pep_3_2D_maxpool")(a2pep_3_2D_CNN)
    a2pep_5_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a2pep_5_2D_maxpool")(a2pep_5_2D_CNN)

    b2pep_1_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b2pep_1_2D_maxpool")(b2pep_1_2D_CNN)
    b2pep_3_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b2pep_3_2D_maxpool")(b2pep_3_2D_CNN)
    b2pep_5_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b2pep_5_2D_maxpool")(b2pep_5_2D_CNN)

    a3pep_1_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3pep_1_2D_maxpool")(a3pep_1_2D_CNN)
    a3pep_3_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3pep_3_2D_maxpool")(a3pep_3_2D_CNN)
    a3pep_5_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3pep_5_2D_maxpool")(a3pep_5_2D_CNN)
    a3pep_7_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3pep_7_2D_maxpool")(a3pep_7_2D_CNN)
    a3pep_9_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_a3pep_9_2D_maxpool")(a3pep_9_2D_CNN)

    b3pep_1_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3pep_1_2D_maxpool")(b3pep_1_2D_CNN)
    b3pep_3_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3pep_3_2D_maxpool")(b3pep_3_2D_CNN)
    b3pep_5_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3pep_5_2D_maxpool")(b3pep_5_2D_CNN)
    b3pep_7_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3pep_7_2D_maxpool")(b3pep_7_2D_CNN)
    b3pep_9_2D_pool = layers.MaxPooling2D(pool_size=(2, 2), padding = "same", name = "first_b3pep_9_2D_maxpool")(b3pep_9_2D_CNN)

    # Second CNN layers for each feature and the different 5 kernel-sizes
    a1pep_1_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_a1pep_1_2D_conv")(a1pep_1_2D_pool)
    a1pep_3_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_a1pep_3_2D_conv")(a1pep_3_2D_pool)
    a1pep_5_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_a1pep_5_2D_conv")(a1pep_5_2D_pool)

    b1pep_1_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_b1pep_1_2D_conv")(b1pep_1_2D_pool)
    b1pep_3_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_b1pep_3_2D_conv")(b1pep_3_2D_pool)
    b1pep_5_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_b1pep_5_2D_conv")(b1pep_5_2D_pool)

    a2pep_1_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_a2pep_1_2D_conv")(a2pep_1_2D_pool)
    a2pep_3_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_a2pep_3_2D_conv")(a2pep_3_2D_pool)
    a2pep_5_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_a2pep_5_2D_conv")(a2pep_5_2D_pool)

    b2pep_1_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_b2pep_1_2D_conv")(b2pep_1_2D_pool)
    b2pep_3_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_b2pep_3_2D_conv")(b2pep_3_2D_pool)
    b2pep_5_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_b2pep_5_2D_conv")(b2pep_5_2D_pool)

    a3pep_1_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_a3pep_1_2D_conv")(a3pep_1_2D_pool)
    a3pep_3_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_a3pep_3_2D_conv")(a3pep_3_2D_pool)
    a3pep_5_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_a3pep_5_2D_conv")(a3pep_5_2D_pool)
    a3pep_7_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "second_a3pep_7_2D_conv")(a3pep_7_2D_pool)
    a3pep_9_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "second_a3pep_9_2D_conv")(a3pep_9_2D_pool)

    b3pep_1_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "second_b3pep_1_2D_conv")(b3pep_1_2D_pool)
    b3pep_3_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "second_b3pep_3_2D_conv")(b3pep_3_2D_pool)
    b3pep_5_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "second_b3pep_5_2D_conv")(b3pep_5_2D_pool)
    b3pep_7_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "second_b3pep_7_2D_conv")(b3pep_7_2D_pool)
    b3pep_9_2D_CNN_2 = layers.Conv2D(filters = nr_of_filters_2_2D, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "second_b3pep_9_2D_conv")(b3pep_9_2D_pool)
    
    # Final GlobalMaxPooling
    a1pep_1_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_a1_1_globalpool", keepdims=True)(a1pep_1_2D_CNN_2)
    a1pep_3_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_a1_3_globalpool", keepdims=True)(a1pep_3_2D_CNN_2)
    a1pep_5_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_a1_5_globalpool", keepdims=True)(a1pep_5_2D_CNN_2)

    b1pep_1_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_b1_1_globalpool", keepdims=True)(b1pep_1_2D_CNN_2)
    b1pep_3_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_b1_3_globalpool", keepdims=True)(b1pep_3_2D_CNN_2)
    b1pep_5_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_b1_5_globalpool", keepdims=True)(b1pep_5_2D_CNN_2)

    a2pep_1_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_a2_1_globalpool", keepdims=True)(a2pep_1_2D_CNN_2)
    a2pep_3_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_a2_3_globalpool", keepdims=True)(a2pep_3_2D_CNN_2)
    a2pep_5_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_a2_5_globalpool", keepdims=True)(a2pep_5_2D_CNN_2)

    b2pep_1_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_b2_1_globalpool", keepdims=True)(b2pep_1_2D_CNN_2)
    b2pep_3_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_b2_3_globalpool", keepdims=True)(b2pep_3_2D_CNN_2)
    b2pep_5_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_b2_5_globalpool", keepdims=True)(b2pep_5_2D_CNN_2)

    a3pep_1_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_1_globalpool", keepdims=True)(a3pep_1_2D_CNN_2)
    a3pep_3_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_3_globalpool", keepdims=True)(a3pep_3_2D_CNN_2)
    a3pep_5_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_5_globalpool", keepdims=True)(a3pep_5_2D_CNN_2)
    a3pep_7_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_7_globalpool", keepdims=True)(a3pep_7_2D_CNN_2)
    a3pep_9_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_a3_9_globalpool", keepdims=True)(a3pep_9_2D_CNN_2)

    b3pep_1_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_1_globalpool", keepdims=True)(b3pep_1_2D_CNN_2)
    b3pep_3_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_3_globalpool", keepdims=True)(b3pep_3_2D_CNN_2)
    b3pep_5_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_5_globalpool", keepdims=True)(b3pep_5_2D_CNN_2)
    b3pep_7_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_7_globalpool", keepdims=True)(b3pep_7_2D_CNN_2)
    b3pep_9_2D_pool_2 = layers.GlobalMaxPooling2D(name = "second_b3_9_globalpool", keepdims=True)(b3pep_9_2D_CNN_2)

    # Flatten 2D-CNN layers before concatenation
    flatten_a1pep_1_2D = layers.Flatten(name="flatten_a1pep_1_2D")(a1pep_1_2D_pool_2)
    flatten_a1pep_3_2D = layers.Flatten(name="flatten_a1pep_3_2D")(a1pep_3_2D_pool_2)   
    flatten_a1pep_5_2D = layers.Flatten(name="flatten_a1pep_5_2D")(a1pep_5_2D_pool_2)

    flatten_b1pep_1_2D = layers.Flatten(name="flatten_b1pep_1_2D")(b1pep_1_2D_pool_2)
    flatten_b1pep_3_2D = layers.Flatten(name="flatten_b1pep_3_2D")(b1pep_3_2D_pool_2)   
    flatten_b1pep_5_2D = layers.Flatten(name="flatten_b1pep_5_2D")(b1pep_5_2D_pool_2)
    
    flatten_a2pep_1_2D = layers.Flatten(name="flatten_a2pep_1_2D")(a2pep_1_2D_pool_2)
    flatten_a2pep_3_2D = layers.Flatten(name="flatten_a2pep_3_2D")(a2pep_3_2D_pool_2)   
    flatten_a2pep_5_2D = layers.Flatten(name="flatten_a2pep_5_2D")(a2pep_5_2D_pool_2)

    flatten_b2pep_1_2D = layers.Flatten(name="flatten_b2pep_1_2D")(b2pep_1_2D_pool_2)
    flatten_b2pep_3_2D = layers.Flatten(name="flatten_b2pep_3_2D")(b2pep_3_2D_pool_2)   
    flatten_b2pep_5_2D = layers.Flatten(name="flatten_b2pep_5_2D")(b2pep_5_2D_pool_2)

    flatten_a3pep_1_2D = layers.Flatten(name="flatten_a3pep_1_2D")(a3pep_1_2D_pool_2)
    flatten_a3pep_3_2D = layers.Flatten(name="flatten_a3pep_3_2D")(a3pep_3_2D_pool_2)   
    flatten_a3pep_5_2D = layers.Flatten(name="flatten_a3pep_5_2D")(a3pep_5_2D_pool_2)
    flatten_a3pep_7_2D = layers.Flatten(name="flatten_a3pep_7_2D")(a3pep_7_2D_pool_2)   
    flatten_a3pep_9_2D = layers.Flatten(name="flatten_a3pep_9_2D")(a3pep_9_2D_pool_2)
    
    flatten_b3pep_1_2D = layers.Flatten(name="flatten_b3pep_1_2D")(b3pep_1_2D_pool_2)
    flatten_b3pep_3_2D = layers.Flatten(name="flatten_b3pep_3_2D")(b3pep_3_2D_pool_2)   
    flatten_b3pep_5_2D = layers.Flatten(name="flatten_b3pep_5_2D")(b3pep_5_2D_pool_2)
    flatten_b3pep_7_2D = layers.Flatten(name="flatten_b3pep_7_2D")(b3pep_7_2D_pool_2)   
    flatten_b3pep_9_2D = layers.Flatten(name="flatten_b3pep_9_2D")(b3pep_9_2D_pool_2)

    # Concatenation of 2D MaxPool outputs from all features and kernel-sizes
    cat_1D = layers.concatenate([pep_1_1D_pool, pep_3_1D_pool, pep_5_1D_pool, pep_7_1D_pool, pep_9_1D_pool,
                                 a1_1_1D_pool, a1_3_1D_pool, a1_5_1D_pool, a1_7_1D_pool, a1_9_1D_pool,
                                 a2_1_1D_pool, a2_3_1D_pool, a2_5_1D_pool, a2_7_1D_pool, a2_9_1D_pool,
                                 a3_1_1D_pool, a3_3_1D_pool, a3_5_1D_pool, a3_7_1D_pool, a3_9_1D_pool,
                                 b1_1_1D_pool, b1_3_1D_pool, b1_5_1D_pool, b1_7_1D_pool, b1_9_1D_pool,
                                 b2_1_1D_pool, b2_3_1D_pool, b2_5_1D_pool, b2_7_1D_pool, b2_9_1D_pool,
                                 b3_1_1D_pool, b3_3_1D_pool, b3_5_1D_pool, b3_7_1D_pool, b3_9_1D_pool])

    # Concatenation of 2D MaxPool outputs from all features and kernel-sizes
    cat_2D = layers.concatenate([flatten_a1pep_1_2D, flatten_a1pep_3_2D, flatten_a1pep_5_2D, 
                                 flatten_b1pep_1_2D, flatten_b1pep_3_2D, flatten_b1pep_5_2D, 
                                 flatten_a2pep_1_2D, flatten_a2pep_3_2D, flatten_a2pep_5_2D, 
                                 flatten_b2pep_1_2D, flatten_b2pep_3_2D, flatten_b2pep_5_2D, 
                                 flatten_a3pep_1_2D, flatten_a3pep_3_2D, flatten_a3pep_5_2D, flatten_a3pep_7_2D, flatten_a3pep_9_2D,
                                 flatten_b3pep_1_2D, flatten_b3pep_3_2D, flatten_b3pep_5_2D, flatten_b3pep_7_2D, flatten_b3pep_9_2D])

    # Dropout and dense layer after concatenation of the 1D-CNN layers
    cat_1D_dropout = layers.Dropout(dropout_rate, seed = seed)(cat_1D)
    dense_1D = layers.Dense(64, activation = dense_activation, name = "first_1D_dense")(cat_1D_dropout)

    # Dropout and dense layer after concatenation of the 2D-CNN layers
    cat_2D_dropout = layers.Dropout(dropout_rate, seed = seed)(cat_2D)
    dense_2D = layers.Dense(64, activation = dense_activation, kernel_regularizer = l2(l2_reg), name = "first_2D_dense")(cat_2D_dropout)

    # Concatenate both separated dense layers into one for a common second dense layer before the output layer
    cat_both = layers.concatenate([dense_1D, dense_2D])
    joint_dense = layers.Dense(64, activation = dense_activation, name = "joint_intermedium_dense")(cat_both)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_combined_dense")(joint_dense)
    
    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)
    
    # Model definition
    model = keras.Model(inputs = [a1pep, b1pep, a2pep, b2pep, a3pep, b3pep, pep, a1, a2, a3, b1, b2, b3],
                        outputs = out)
    
    return model

# 2D CNN including CDR123 maps, but individually introduced in the 2D CNN and maintaining the 1D baseline architecture
def CNN_CDR123_2D_individual_maps(dropout_rate, seed, conv_activation = "relu", dense_activation = "relu", 
                                  nr_of_filters_1 = 16, max_lengths = None, l2_reg = 0.001):

    # Max.length of the sequences from the dataset
    if max_lengths:
        a1_max = max_lengths[0]
        a2_max = max_lengths[1]
        a3_max = max_lengths[2]
        b1_max = max_lengths[3]
        b2_max = max_lengths[4]
        b3_max = max_lengths[5]
        pep_max = max_lengths[6]
    else:
        a1_max = 7
        a2_max = 8
        a3_max = 22
        b1_max = 6
        b2_max = 7
        b3_max = 23
        pep_max = 12

    # 2D-Input dimensions for a1 CDR123 maps
    a1pep_mass = keras.Input(shape = (a1_max, pep_max, 1), name ="a1pep_mass")
    a1pep_ip = keras.Input(shape = (a1_max, pep_max, 1), name ="a1pep_ip")
    a1pep_hpho = keras.Input(shape = (a1_max, pep_max, 1), name ="a1pep_hpho")
    a1pep_hphi = keras.Input(shape = (a1_max, pep_max, 1), name ="a1pep_hphi")
    a1pep_blosum = keras.Input(shape = (a1_max, pep_max, 1), name ="a1pep_blosum")
    
    # 2D-Input dimensions for a2 CDR123 maps
    a2pep_mass = keras.Input(shape=(a2_max, pep_max, 1), name="a2pep_mass")
    a2pep_ip = keras.Input(shape=(a2_max, pep_max, 1), name="a2pep_ip")
    a2pep_hpho = keras.Input(shape=(a2_max, pep_max, 1), name="a2pep_hpho")
    a2pep_hphi = keras.Input(shape=(a2_max, pep_max, 1), name="a2pep_hphi")
    a2pep_blosum = keras.Input(shape=(a2_max, pep_max, 1), name="a2pep_blosum")

    # 2D-Input dimensions for a3 CDR123 maps
    a3pep_mass = keras.Input(shape=(a3_max, pep_max, 1), name="a3pep_mass")
    a3pep_ip = keras.Input(shape=(a3_max, pep_max, 1), name="a3pep_ip")
    a3pep_hpho = keras.Input(shape=(a3_max, pep_max, 1), name="a3pep_hpho")
    a3pep_hphi = keras.Input(shape=(a3_max, pep_max, 1), name="a3pep_hphi")
    a3pep_blosum = keras.Input(shape=(a3_max, pep_max, 1), name="a3pep_blosum")

    # 2D-Input dimensions for b1 CDR123 maps
    b1pep_mass = keras.Input(shape=(b1_max, pep_max, 1), name="b1pep_mass")
    b1pep_ip = keras.Input(shape=(b1_max, pep_max, 1), name="b1pep_ip")
    b1pep_hpho = keras.Input(shape=(b1_max, pep_max, 1), name="b1pep_hpho")
    b1pep_hphi = keras.Input(shape=(b1_max, pep_max, 1), name="b1pep_hphi")
    b1pep_blosum = keras.Input(shape=(b1_max, pep_max, 1), name="b1pep_blosum")

    # 2D-Input dimensions for b2 CDR123 maps
    b2pep_mass = keras.Input(shape=(b2_max, pep_max, 1), name="b2pep_mass")
    b2pep_ip = keras.Input(shape=(b2_max, pep_max, 1), name="b2pep_ip")
    b2pep_hpho = keras.Input(shape=(b2_max, pep_max, 1), name="b2pep_hpho")
    b2pep_hphi = keras.Input(shape=(b2_max, pep_max, 1), name="b2pep_hphi")
    b2pep_blosum = keras.Input(shape=(b2_max, pep_max, 1), name="b2pep_blosum")

    # 2D-Input dimensions for b3 CDR123 maps
    b3pep_mass = keras.Input(shape=(b3_max, pep_max, 1), name="b3pep_mass")
    b3pep_ip = keras.Input(shape=(b3_max, pep_max, 1), name="b3pep_ip")
    b3pep_hpho = keras.Input(shape=(b3_max, pep_max, 1), name="b3pep_hpho")
    b3pep_hphi = keras.Input(shape=(b3_max, pep_max, 1), name="b3pep_hphi")
    b3pep_blosum = keras.Input(shape=(b3_max, pep_max, 1), name="b3pep_blosum")

    # 2D-CNN layers for a1 maps
    a1pep_1_mass_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "a1pep_mass_1_conv")(a1pep_mass)
    a1pep_3_mass_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "a1pep_mass_3_conv")(a1pep_mass)
    a1pep_5_mass_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "a1pep_mass_5_conv")(a1pep_mass)
    a1pep_1_ip_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "a1pep_ip_1_conv")(a1pep_ip)
    a1pep_3_ip_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "a1pep_ip_3_conv")(a1pep_ip)
    a1pep_5_ip_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "a1pep_ip_5_conv")(a1pep_ip)
    a1pep_1_hpho_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "a1pep_hpho_1_conv")(a1pep_hpho)
    a1pep_3_hpho_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "a1pep_hpho_3_conv")(a1pep_hpho)
    a1pep_5_hpho_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "a1pep_hpho_5_conv")(a1pep_hpho)
    a1pep_1_hphi_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "a1pep_hphi_1_conv")(a1pep_hphi)
    a1pep_3_hphi_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "a1pep_hphi_3_conv")(a1pep_hphi)
    a1pep_5_hphi_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "a1pep_hphi_5_conv")(a1pep_hphi)
    a1pep_1_blosum_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "a1pep_blosum_1_conv")(a1pep_blosum)
    a1pep_3_blosum_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "a1pep_blosum_3_conv")(a1pep_blosum)
    a1pep_5_blosum_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "a1pep_blosum_5_conv")(a1pep_blosum)

    # 2D-CNN layers for a2 maps
    a2pep_1_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="a2pep_mass_1_conv")(a2pep_mass)
    a2pep_3_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="a2pep_mass_3_conv")(a2pep_mass)
    a2pep_5_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="a2pep_mass_5_conv")(a2pep_mass)
    a2pep_1_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="a2pep_ip_1_conv")(a2pep_ip)
    a2pep_3_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="a2pep_ip_3_conv")(a2pep_ip)
    a2pep_5_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="a2pep_ip_5_conv")(a2pep_ip)
    a2pep_1_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="a2pep_hpho_1_conv")(a2pep_hpho)
    a2pep_3_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="a2pep_hpho_3_conv")(a2pep_hpho)
    a2pep_5_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="a2pep_hpho_5_conv")(a2pep_hpho)
    a2pep_1_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="a2pep_hphi_1_conv")(a2pep_hphi)
    a2pep_3_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="a2pep_hphi_3_conv")(a2pep_hphi)
    a2pep_5_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="a2pep_hphi_5_conv")(a2pep_hphi)
    a2pep_1_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="a2pep_blosum_1_conv")(a2pep_blosum)
    a2pep_3_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="a2pep_blosum_3_conv")(a2pep_blosum)
    a2pep_5_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="a2pep_blosum_5_conv")(a2pep_blosum)

    # 2D-CNN layers for a3 maps
    a3pep_1_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="a3pep_mass_1_conv")(a3pep_mass)
    a3pep_3_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="a3pep_mass_3_conv")(a3pep_mass)
    a3pep_5_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="a3pep_mass_5_conv")(a3pep_mass)
    a3pep_7_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(7, 7), padding="same", activation=conv_activation, name="a3pep_mass_7_conv")(a3pep_mass)
    a3pep_9_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(9, 9), padding="same", activation=conv_activation, name="a3pep_mass_9_conv")(a3pep_mass)
    a3pep_1_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="a3pep_ip_1_conv")(a3pep_ip)
    a3pep_3_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="a3pep_ip_3_conv")(a3pep_ip)
    a3pep_5_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="a3pep_ip_5_conv")(a3pep_ip)
    a3pep_7_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(7, 7), padding="same", activation=conv_activation, name="a3pep_ip_7_conv")(a3pep_ip)
    a3pep_9_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(9, 9), padding="same", activation=conv_activation, name="a3pep_ip_9_conv")(a3pep_ip)
    a3pep_1_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="a3pep_hpho_1_conv")(a3pep_hpho)
    a3pep_3_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="a3pep_hpho_3_conv")(a3pep_hpho)
    a3pep_5_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="a3pep_hpho_5_conv")(a3pep_hpho)
    a3pep_7_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(7, 7), padding="same", activation=conv_activation, name="a3pep_hpho_7_conv")(a3pep_hpho)
    a3pep_9_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(9, 9), padding="same", activation=conv_activation, name="a3pep_hpho_9_conv")(a3pep_hpho)
    a3pep_1_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="a3pep_hphi_1_conv")(a3pep_hphi)
    a3pep_3_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="a3pep_hphi_3_conv")(a3pep_hphi)
    a3pep_5_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="a3pep_hphi_5_conv")(a3pep_hphi)
    a3pep_7_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(7, 7), padding="same", activation=conv_activation, name="a3pep_hphi_7_conv")(a3pep_hphi)
    a3pep_9_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(9, 9), padding="same", activation=conv_activation, name="a3pep_hphi_9_conv")(a3pep_hphi)
    a3pep_1_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="a3pep_blosum_1_conv")(a3pep_blosum)
    a3pep_3_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="a3pep_blosum_3_conv")(a3pep_blosum)
    a3pep_5_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="a3pep_blosum_5_conv")(a3pep_blosum)
    a3pep_7_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(7, 7), padding="same", activation=conv_activation, name="a3pep_blosum_7_conv")(a3pep_blosum)
    a3pep_9_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(9, 9), padding="same", activation=conv_activation, name="a3pep_blosum_9_conv")(a3pep_blosum)

    # 2D-CNN layers for b1 maps
    b1pep_1_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b1pep_mass_1_conv")(b1pep_mass)
    b1pep_3_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b1pep_mass_3_conv")(b1pep_mass)
    b1pep_5_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b1pep_mass_5_conv")(b1pep_mass)
    b1pep_1_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b1pep_ip_1_conv")(b1pep_ip)
    b1pep_3_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b1pep_ip_3_conv")(b1pep_ip)
    b1pep_5_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b1pep_ip_5_conv")(b1pep_ip)
    b1pep_1_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b1pep_hpho_1_conv")(b1pep_hpho)
    b1pep_3_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b1pep_hpho_3_conv")(b1pep_hpho)
    b1pep_5_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b1pep_hpho_5_conv")(b1pep_hpho)
    b1pep_1_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b1pep_hphi_1_conv")(b1pep_hphi)
    b1pep_3_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b1pep_hphi_3_conv")(b1pep_hphi)
    b1pep_5_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b1pep_hphi_5_conv")(b1pep_hphi)
    b1pep_1_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b1pep_blosum_1_conv")(b1pep_blosum)
    b1pep_3_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b1pep_blosum_3_conv")(b1pep_blosum)
    b1pep_5_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b1pep_blosum_5_conv")(b1pep_blosum)

    # 2D-CNN layers for b2 maps
    b2pep_1_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b2pep_mass_1_conv")(b2pep_mass)
    b2pep_3_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b2pep_mass_3_conv")(b2pep_mass)
    b2pep_5_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b2pep_mass_5_conv")(b2pep_mass)
    b2pep_1_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b2pep_ip_1_conv")(b2pep_ip)
    b2pep_3_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b2pep_ip_3_conv")(b2pep_ip)
    b2pep_5_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b2pep_ip_5_conv")(b2pep_ip)
    b2pep_1_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b2pep_hpho_1_conv")(b2pep_hpho)
    b2pep_3_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b2pep_hpho_3_conv")(b2pep_hpho)
    b2pep_5_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b2pep_hpho_5_conv")(b2pep_hpho)
    b2pep_1_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b2pep_hphi_1_conv")(b2pep_hphi)
    b2pep_3_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b2pep_hphi_3_conv")(b2pep_hphi)
    b2pep_5_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b2pep_hphi_5_conv")(b2pep_hphi)
    b2pep_1_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b2pep_blosum_1_conv")(b2pep_blosum)
    b2pep_3_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b2pep_blosum_3_conv")(b2pep_blosum)
    b2pep_5_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b2pep_blosum_5_conv")(b2pep_blosum)

    # 2D-CNN layers for b3 maps
    b3pep_1_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b3pep_mass_1_conv")(b3pep_mass)
    b3pep_3_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b3pep_mass_3_conv")(b3pep_mass)
    b3pep_5_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b3pep_mass_5_conv")(b3pep_mass)
    b3pep_7_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(7, 7), padding="same", activation=conv_activation, name="b3pep_mass_7_conv")(b3pep_mass)
    b3pep_9_mass_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(9, 9), padding="same", activation=conv_activation, name="b3pep_mass_9_conv")(b3pep_mass)
    b3pep_1_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b3pep_ip_1_conv")(b3pep_ip)
    b3pep_3_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b3pep_ip_3_conv")(b3pep_ip)
    b3pep_5_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b3pep_ip_5_conv")(b3pep_ip)
    b3pep_7_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(7, 7), padding="same", activation=conv_activation, name="b3pep_ip_7_conv")(b3pep_ip)
    b3pep_9_ip_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(9, 9), padding="same", activation=conv_activation, name="b3pep_ip_9_conv")(b3pep_ip)
    b3pep_1_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b3pep_hpho_1_conv")(b3pep_hpho)
    b3pep_3_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b3pep_hpho_3_conv")(b3pep_hpho)
    b3pep_5_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b3pep_hpho_5_conv")(b3pep_hpho)
    b3pep_7_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(7, 7), padding="same", activation=conv_activation, name="b3pep_hpho_7_conv")(b3pep_hpho)
    b3pep_9_hpho_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(9, 9), padding="same", activation=conv_activation, name="b3pep_hpho_9_conv")(b3pep_hpho)
    b3pep_1_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b3pep_hphi_1_conv")(b3pep_hphi)
    b3pep_3_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b3pep_hphi_3_conv")(b3pep_hphi)
    b3pep_5_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b3pep_hphi_5_conv")(b3pep_hphi)
    b3pep_7_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(7, 7), padding="same", activation=conv_activation, name="b3pep_hphi_7_conv")(b3pep_hphi)
    b3pep_9_hphi_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(9, 9), padding="same", activation=conv_activation, name="b3pep_hphi_9_conv")(b3pep_hphi)
    b3pep_1_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(1, 1), padding="same", activation=conv_activation, name="b3pep_blosum_1_conv")(b3pep_blosum)
    b3pep_3_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(3, 3), padding="same", activation=conv_activation, name="b3pep_blosum_3_conv")(b3pep_blosum)
    b3pep_5_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(5, 5), padding="same", activation=conv_activation, name="b3pep_blosum_5_conv")(b3pep_blosum)
    b3pep_7_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(7, 7), padding="same", activation=conv_activation, name="b3pep_blosum_7_conv")(b3pep_blosum)
    b3pep_9_blosum_CNN = layers.Conv2D(filters=nr_of_filters_1, kernel_size=(9, 9), padding="same", activation=conv_activation, name="b3pep_blosum_9_conv")(b3pep_blosum)

    # Batch normalization after convolution for a1 maps
    a1pep_1_mass_CNN = layers.BatchNormalization()(a1pep_1_mass_CNN)
    a1pep_3_mass_CNN = layers.BatchNormalization()(a1pep_3_mass_CNN)
    a1pep_5_mass_CNN = layers.BatchNormalization()(a1pep_5_mass_CNN)
    a1pep_1_ip_CNN = layers.BatchNormalization()(a1pep_1_ip_CNN)
    a1pep_3_ip_CNN = layers.BatchNormalization()(a1pep_3_ip_CNN)
    a1pep_5_ip_CNN = layers.BatchNormalization()(a1pep_5_ip_CNN)
    a1pep_1_hpho_CNN = layers.BatchNormalization()(a1pep_1_hpho_CNN)
    a1pep_3_hpho_CNN = layers.BatchNormalization()(a1pep_3_hpho_CNN)
    a1pep_5_hpho_CNN = layers.BatchNormalization()(a1pep_5_hpho_CNN)
    a1pep_1_hphi_CNN = layers.BatchNormalization()(a1pep_1_hphi_CNN)
    a1pep_3_hphi_CNN = layers.BatchNormalization()(a1pep_3_hphi_CNN)
    a1pep_5_hphi_CNN = layers.BatchNormalization()(a1pep_5_hphi_CNN)
    a1pep_1_blosum_CNN = layers.BatchNormalization()(a1pep_1_blosum_CNN)
    a1pep_3_blosum_CNN = layers.BatchNormalization()(a1pep_3_blosum_CNN)
    a1pep_5_blosum_CNN = layers.BatchNormalization()(a1pep_5_blosum_CNN)

    # Batch normalization after convolution for a2 maps
    a2pep_1_mass_CNN = layers.BatchNormalization()(a2pep_1_mass_CNN)
    a2pep_3_mass_CNN = layers.BatchNormalization()(a2pep_3_mass_CNN)
    a2pep_5_mass_CNN = layers.BatchNormalization()(a2pep_5_mass_CNN)
    a2pep_1_ip_CNN = layers.BatchNormalization()(a2pep_1_ip_CNN)
    a2pep_3_ip_CNN = layers.BatchNormalization()(a2pep_3_ip_CNN)
    a2pep_5_ip_CNN = layers.BatchNormalization()(a2pep_5_ip_CNN)
    a2pep_1_hpho_CNN = layers.BatchNormalization()(a2pep_1_hpho_CNN)
    a2pep_3_hpho_CNN = layers.BatchNormalization()(a2pep_3_hpho_CNN)
    a2pep_5_hpho_CNN = layers.BatchNormalization()(a2pep_5_hpho_CNN)
    a2pep_1_hphi_CNN = layers.BatchNormalization()(a2pep_1_hphi_CNN)
    a2pep_3_hphi_CNN = layers.BatchNormalization()(a2pep_3_hphi_CNN)
    a2pep_5_hphi_CNN = layers.BatchNormalization()(a2pep_5_hphi_CNN)
    a2pep_1_blosum_CNN = layers.BatchNormalization()(a2pep_1_blosum_CNN)
    a2pep_3_blosum_CNN = layers.BatchNormalization()(a2pep_3_blosum_CNN)
    a2pep_5_blosum_CNN = layers.BatchNormalization()(a2pep_5_blosum_CNN)

    # Batch normalization after convolution for a3 maps
    a3pep_1_mass_CNN = layers.BatchNormalization()(a3pep_1_mass_CNN)
    a3pep_3_mass_CNN = layers.BatchNormalization()(a3pep_3_mass_CNN)
    a3pep_5_mass_CNN = layers.BatchNormalization()(a3pep_5_mass_CNN)
    a3pep_7_mass_CNN = layers.BatchNormalization()(a3pep_7_mass_CNN)
    a3pep_9_mass_CNN = layers.BatchNormalization()(a3pep_9_mass_CNN)
    a3pep_1_ip_CNN = layers.BatchNormalization()(a3pep_1_ip_CNN)
    a3pep_3_ip_CNN = layers.BatchNormalization()(a3pep_3_ip_CNN)
    a3pep_5_ip_CNN = layers.BatchNormalization()(a3pep_5_ip_CNN)
    a3pep_7_ip_CNN = layers.BatchNormalization()(a3pep_7_ip_CNN)
    a3pep_9_ip_CNN = layers.BatchNormalization()(a3pep_9_ip_CNN)
    a3pep_1_hpho_CNN = layers.BatchNormalization()(a3pep_1_hpho_CNN)
    a3pep_3_hpho_CNN = layers.BatchNormalization()(a3pep_3_hpho_CNN)
    a3pep_5_hpho_CNN = layers.BatchNormalization()(a3pep_5_hpho_CNN)
    a3pep_7_hpho_CNN = layers.BatchNormalization()(a3pep_7_hpho_CNN)
    a3pep_9_hpho_CNN = layers.BatchNormalization()(a3pep_9_hpho_CNN)
    a3pep_1_hphi_CNN = layers.BatchNormalization()(a3pep_1_hphi_CNN)
    a3pep_3_hphi_CNN = layers.BatchNormalization()(a3pep_3_hphi_CNN)
    a3pep_5_hphi_CNN = layers.BatchNormalization()(a3pep_5_hphi_CNN)
    a3pep_7_hphi_CNN = layers.BatchNormalization()(a3pep_7_hphi_CNN)
    a3pep_9_hphi_CNN = layers.BatchNormalization()(a3pep_9_hphi_CNN)
    a3pep_1_blosum_CNN = layers.BatchNormalization()(a3pep_1_blosum_CNN)
    a3pep_3_blosum_CNN = layers.BatchNormalization()(a3pep_3_blosum_CNN)
    a3pep_5_blosum_CNN = layers.BatchNormalization()(a3pep_5_blosum_CNN)
    a3pep_7_blosum_CNN = layers.BatchNormalization()(a3pep_7_blosum_CNN)
    a3pep_9_blosum_CNN = layers.BatchNormalization()(a3pep_9_blosum_CNN)

    # Batch normalization after convolution for b1 maps
    b1pep_1_mass_CNN = layers.BatchNormalization()(b1pep_1_mass_CNN)
    b1pep_3_mass_CNN = layers.BatchNormalization()(b1pep_3_mass_CNN)
    b1pep_5_mass_CNN = layers.BatchNormalization()(b1pep_5_mass_CNN)
    b1pep_1_ip_CNN = layers.BatchNormalization()(b1pep_1_ip_CNN)
    b1pep_3_ip_CNN = layers.BatchNormalization()(b1pep_3_ip_CNN)
    b1pep_5_ip_CNN = layers.BatchNormalization()(b1pep_5_ip_CNN)
    b1pep_1_hpho_CNN = layers.BatchNormalization()(b1pep_1_hpho_CNN)
    b1pep_3_hpho_CNN = layers.BatchNormalization()(b1pep_3_hpho_CNN)
    b1pep_5_hpho_CNN = layers.BatchNormalization()(b1pep_5_hpho_CNN)
    b1pep_1_hphi_CNN = layers.BatchNormalization()(b1pep_1_hphi_CNN)
    b1pep_3_hphi_CNN = layers.BatchNormalization()(b1pep_3_hphi_CNN)
    b1pep_5_hphi_CNN = layers.BatchNormalization()(b1pep_5_hphi_CNN)
    b1pep_1_blosum_CNN = layers.BatchNormalization()(b1pep_1_blosum_CNN)
    b1pep_3_blosum_CNN = layers.BatchNormalization()(b1pep_3_blosum_CNN)
    b1pep_5_blosum_CNN = layers.BatchNormalization()(b1pep_5_blosum_CNN)

    # Batch normalization after convolution for a2 maps
    b2pep_1_mass_CNN = layers.BatchNormalization()(b2pep_1_mass_CNN)
    b2pep_3_mass_CNN = layers.BatchNormalization()(b2pep_3_mass_CNN)
    b2pep_5_mass_CNN = layers.BatchNormalization()(b2pep_5_mass_CNN)
    b2pep_1_ip_CNN = layers.BatchNormalization()(b2pep_1_ip_CNN)
    b2pep_3_ip_CNN = layers.BatchNormalization()(b2pep_3_ip_CNN)
    b2pep_5_ip_CNN = layers.BatchNormalization()(b2pep_5_ip_CNN)
    b2pep_1_hpho_CNN = layers.BatchNormalization()(b2pep_1_hpho_CNN)
    b2pep_3_hpho_CNN = layers.BatchNormalization()(b2pep_3_hpho_CNN)
    b2pep_5_hpho_CNN = layers.BatchNormalization()(b2pep_5_hpho_CNN)
    b2pep_1_hphi_CNN = layers.BatchNormalization()(b2pep_1_hphi_CNN)
    b2pep_3_hphi_CNN = layers.BatchNormalization()(b2pep_3_hphi_CNN)
    b2pep_5_hphi_CNN = layers.BatchNormalization()(b2pep_5_hphi_CNN)
    b2pep_1_blosum_CNN = layers.BatchNormalization()(b2pep_1_blosum_CNN)
    b2pep_3_blosum_CNN = layers.BatchNormalization()(b2pep_3_blosum_CNN)
    b2pep_5_blosum_CNN = layers.BatchNormalization()(b2pep_5_blosum_CNN)

    # Batch normalization after convolution for a3 maps
    b3pep_1_mass_CNN = layers.BatchNormalization()(b3pep_1_mass_CNN)
    b3pep_3_mass_CNN = layers.BatchNormalization()(b3pep_3_mass_CNN)
    b3pep_5_mass_CNN = layers.BatchNormalization()(b3pep_5_mass_CNN)
    b3pep_7_mass_CNN = layers.BatchNormalization()(b3pep_7_mass_CNN)
    b3pep_9_mass_CNN = layers.BatchNormalization()(b3pep_9_mass_CNN)
    b3pep_1_ip_CNN = layers.BatchNormalization()(b3pep_1_ip_CNN)
    b3pep_3_ip_CNN = layers.BatchNormalization()(b3pep_3_ip_CNN)
    b3pep_5_ip_CNN = layers.BatchNormalization()(b3pep_5_ip_CNN)
    b3pep_7_ip_CNN = layers.BatchNormalization()(b3pep_7_ip_CNN)
    b3pep_9_ip_CNN = layers.BatchNormalization()(b3pep_9_ip_CNN)
    b3pep_1_hpho_CNN = layers.BatchNormalization()(b3pep_1_hpho_CNN)
    b3pep_3_hpho_CNN = layers.BatchNormalization()(b3pep_3_hpho_CNN)
    b3pep_5_hpho_CNN = layers.BatchNormalization()(b3pep_5_hpho_CNN)
    b3pep_7_hpho_CNN = layers.BatchNormalization()(b3pep_7_hpho_CNN)
    b3pep_9_hpho_CNN = layers.BatchNormalization()(b3pep_9_hpho_CNN)
    b3pep_1_hphi_CNN = layers.BatchNormalization()(b3pep_1_hphi_CNN)
    b3pep_3_hphi_CNN = layers.BatchNormalization()(b3pep_3_hphi_CNN)
    b3pep_5_hphi_CNN = layers.BatchNormalization()(b3pep_5_hphi_CNN)
    b3pep_7_hphi_CNN = layers.BatchNormalization()(b3pep_7_hphi_CNN)
    b3pep_9_hphi_CNN = layers.BatchNormalization()(b3pep_9_hphi_CNN)
    b3pep_1_blosum_CNN = layers.BatchNormalization()(b3pep_1_blosum_CNN)
    b3pep_3_blosum_CNN = layers.BatchNormalization()(b3pep_3_blosum_CNN)
    b3pep_5_blosum_CNN = layers.BatchNormalization()(b3pep_5_blosum_CNN)
    b3pep_7_blosum_CNN = layers.BatchNormalization()(b3pep_7_blosum_CNN)
    b3pep_9_blosum_CNN = layers.BatchNormalization()(b3pep_9_blosum_CNN)

    # GlobalMaxPooling for the a1 maps
    a1pep_1_mass_pool = layers.GlobalMaxPooling2D(name = "a1pep_mass_1_pool", keepdims=True)(a1pep_1_mass_CNN)
    a1pep_3_mass_pool = layers.GlobalMaxPooling2D(name = "a1pep_mass_3_pool", keepdims=True)(a1pep_3_mass_CNN)
    a1pep_5_mass_pool = layers.GlobalMaxPooling2D(name = "a1pep_mass_5_pool", keepdims=True)(a1pep_5_mass_CNN)
    a1pep_1_ip_pool = layers.GlobalMaxPooling2D(name = "a1pep_ip_1_pool", keepdims=True)(a1pep_1_ip_CNN)
    a1pep_3_ip_pool = layers.GlobalMaxPooling2D(name = "a1pep_ip_3_pool", keepdims=True)(a1pep_3_ip_CNN)
    a1pep_5_ip_pool = layers.GlobalMaxPooling2D(name = "a1pep_ip_5_pool", keepdims=True)(a1pep_5_ip_CNN)
    a1pep_1_hpho_pool = layers.GlobalMaxPooling2D(name = "a1pep_hpho_1_pool", keepdims=True)(a1pep_1_hpho_CNN)
    a1pep_3_hpho_pool = layers.GlobalMaxPooling2D(name = "a1pep_hpho_3_pool", keepdims=True)(a1pep_3_hpho_CNN)
    a1pep_5_hpho_pool = layers.GlobalMaxPooling2D(name = "a1pep_hpho_5_pool", keepdims=True)(a1pep_5_hpho_CNN)
    a1pep_1_hphi_pool = layers.GlobalMaxPooling2D(name = "a1pep_hphi_1_pool", keepdims=True)(a1pep_1_hphi_CNN)
    a1pep_3_hphi_pool = layers.GlobalMaxPooling2D(name = "a1pep_hphi_3_pool", keepdims=True)(a1pep_3_hphi_CNN)
    a1pep_5_hphi_pool = layers.GlobalMaxPooling2D(name = "a1pep_hphi_5_pool", keepdims=True)(a1pep_5_hphi_CNN)
    a1pep_1_blosum_pool = layers.GlobalMaxPooling2D(name = "a1pep_blosum_1_pool", keepdims=True)(a1pep_1_blosum_CNN)
    a1pep_3_blosum_pool = layers.GlobalMaxPooling2D(name = "a1pep_blosum_3_pool", keepdims=True)(a1pep_3_blosum_CNN)
    a1pep_5_blosum_pool = layers.GlobalMaxPooling2D(name = "a1pep_5_blosum_pool", keepdims=True)(a1pep_5_blosum_CNN)

    # GlobalMaxPooling for the a2 maps
    a2pep_1_mass_pool = layers.GlobalMaxPooling2D(name="a2pep_mass_1_pool", keepdims=True)(a2pep_1_mass_CNN)
    a2pep_3_mass_pool = layers.GlobalMaxPooling2D(name="a2pep_mass_3_pool", keepdims=True)(a2pep_3_mass_CNN)
    a2pep_5_mass_pool = layers.GlobalMaxPooling2D(name="a2pep_mass_5_pool", keepdims=True)(a2pep_5_mass_CNN)
    a2pep_1_ip_pool = layers.GlobalMaxPooling2D(name="a2pep_ip_1_pool", keepdims=True)(a2pep_1_ip_CNN)
    a2pep_3_ip_pool = layers.GlobalMaxPooling2D(name="a2pep_ip_3_pool", keepdims=True)(a2pep_3_ip_CNN)
    a2pep_5_ip_pool = layers.GlobalMaxPooling2D(name="a2pep_ip_5_pool", keepdims=True)(a2pep_5_ip_CNN)
    a2pep_1_hpho_pool = layers.GlobalMaxPooling2D(name="a2pep_hpho_1_pool", keepdims=True)(a2pep_1_hpho_CNN)
    a2pep_3_hpho_pool = layers.GlobalMaxPooling2D(name="a2pep_hpho_3_pool", keepdims=True)(a2pep_3_hpho_CNN)
    a2pep_5_hpho_pool = layers.GlobalMaxPooling2D(name="a2pep_hpho_5_pool", keepdims=True)(a2pep_5_hpho_CNN)
    a2pep_1_hphi_pool = layers.GlobalMaxPooling2D(name="a2pep_hphi_1_pool", keepdims=True)(a2pep_1_hphi_CNN)
    a2pep_3_hphi_pool = layers.GlobalMaxPooling2D(name="a2pep_hphi_3_pool", keepdims=True)(a2pep_3_hphi_CNN)
    a2pep_5_hphi_pool = layers.GlobalMaxPooling2D(name="a2pep_hphi_5_pool", keepdims=True)(a2pep_5_hphi_CNN)
    a2pep_1_blosum_pool = layers.GlobalMaxPooling2D(name="a2pep_blosum_1_pool", keepdims=True)(a2pep_1_blosum_CNN)
    a2pep_3_blosum_pool = layers.GlobalMaxPooling2D(name="a2pep_blosum_3_pool", keepdims=True)(a2pep_3_blosum_CNN)
    a2pep_5_blosum_pool = layers.GlobalMaxPooling2D(name="a2pep_blosum_5_pool", keepdims=True)(a2pep_5_blosum_CNN)

    # GlobalMaxPooling for the a3 maps
    a3pep_1_mass_pool = layers.GlobalMaxPooling2D(name="a3pep_mass_1_pool", keepdims=True)(a3pep_1_mass_CNN)
    a3pep_3_mass_pool = layers.GlobalMaxPooling2D(name="a3pep_mass_3_pool", keepdims=True)(a3pep_3_mass_CNN)
    a3pep_5_mass_pool = layers.GlobalMaxPooling2D(name="a3pep_mass_5_pool", keepdims=True)(a3pep_5_mass_CNN)
    a3pep_7_mass_pool = layers.GlobalMaxPooling2D(name="a3pep_mass_7_pool", keepdims=True)(a3pep_7_mass_CNN)
    a3pep_9_mass_pool = layers.GlobalMaxPooling2D(name="a3pep_mass_9_pool", keepdims=True)(a3pep_9_mass_CNN)
    a3pep_1_ip_pool = layers.GlobalMaxPooling2D(name="a3pep_ip_1_pool", keepdims=True)(a3pep_1_ip_CNN)
    a3pep_3_ip_pool = layers.GlobalMaxPooling2D(name="a3pep_ip_3_pool", keepdims=True)(a3pep_3_ip_CNN)
    a3pep_5_ip_pool = layers.GlobalMaxPooling2D(name="a3pep_ip_5_pool", keepdims=True)(a3pep_5_ip_CNN)
    a3pep_7_ip_pool = layers.GlobalMaxPooling2D(name="a3pep_ip_7_pool", keepdims=True)(a3pep_7_ip_CNN)
    a3pep_9_ip_pool = layers.GlobalMaxPooling2D(name="a3pep_ip_9_pool", keepdims=True)(a3pep_9_ip_CNN)
    a3pep_1_hpho_pool = layers.GlobalMaxPooling2D(name="a3pep_hpho_1_pool", keepdims=True)(a3pep_1_hpho_CNN)
    a3pep_3_hpho_pool = layers.GlobalMaxPooling2D(name="a3pep_hpho_3_pool", keepdims=True)(a3pep_3_hpho_CNN)
    a3pep_5_hpho_pool = layers.GlobalMaxPooling2D(name="a3pep_hpho_5_pool", keepdims=True)(a3pep_5_hpho_CNN)
    a3pep_7_hpho_pool = layers.GlobalMaxPooling2D(name="a3pep_hpho_7_pool", keepdims=True)(a3pep_7_hpho_CNN)
    a3pep_9_hpho_pool = layers.GlobalMaxPooling2D(name="a3pep_hpho_9_pool", keepdims=True)(a3pep_9_hpho_CNN)
    a3pep_1_hphi_pool = layers.GlobalMaxPooling2D(name="a3pep_hphi_1_pool", keepdims=True)(a3pep_1_hphi_CNN)
    a3pep_3_hphi_pool = layers.GlobalMaxPooling2D(name="a3pep_hphi_3_pool", keepdims=True)(a3pep_3_hphi_CNN)
    a3pep_5_hphi_pool = layers.GlobalMaxPooling2D(name="a3pep_hphi_5_pool", keepdims=True)(a3pep_5_hphi_CNN)
    a3pep_7_hphi_pool = layers.GlobalMaxPooling2D(name="a3pep_hphi_7_pool", keepdims=True)(a3pep_7_hphi_CNN)
    a3pep_9_hphi_pool = layers.GlobalMaxPooling2D(name="a3pep_hphi_9_pool", keepdims=True)(a3pep_9_hphi_CNN)
    a3pep_1_blosum_pool = layers.GlobalMaxPooling2D(name="a3pep_blosum_1_pool", keepdims=True)(a3pep_1_blosum_CNN)
    a3pep_3_blosum_pool = layers.GlobalMaxPooling2D(name="a3pep_blosum_3_pool", keepdims=True)(a3pep_3_blosum_CNN)
    a3pep_5_blosum_pool = layers.GlobalMaxPooling2D(name="a3pep_blosum_5_pool", keepdims=True)(a3pep_5_blosum_CNN)
    a3pep_7_blosum_pool = layers.GlobalMaxPooling2D(name="a3pep_blosum_7_pool", keepdims=True)(a3pep_7_blosum_CNN)
    a3pep_9_blosum_pool = layers.GlobalMaxPooling2D(name="a3pep_blosum_9_pool", keepdims=True)(a3pep_9_blosum_CNN)

    # GlobalMaxPooling for the b1 maps
    b1pep_1_mass_pool = layers.GlobalMaxPooling2D(name="b1pep_mass_1_pool", keepdims=True)(b1pep_1_mass_CNN)
    b1pep_3_mass_pool = layers.GlobalMaxPooling2D(name="b1pep_mass_3_pool", keepdims=True)(b1pep_3_mass_CNN)
    b1pep_5_mass_pool = layers.GlobalMaxPooling2D(name="b1pep_mass_5_pool", keepdims=True)(b1pep_5_mass_CNN)
    b1pep_1_ip_pool = layers.GlobalMaxPooling2D(name="b1pep_ip_1_pool", keepdims=True)(b1pep_1_ip_CNN)
    b1pep_3_ip_pool = layers.GlobalMaxPooling2D(name="b1pep_ip_3_pool", keepdims=True)(b1pep_3_ip_CNN)
    b1pep_5_ip_pool = layers.GlobalMaxPooling2D(name="b1pep_ip_5_pool", keepdims=True)(b1pep_5_ip_CNN)
    b1pep_1_hpho_pool = layers.GlobalMaxPooling2D(name="b1pep_hpho_1_pool", keepdims=True)(b1pep_1_hpho_CNN)
    b1pep_3_hpho_pool = layers.GlobalMaxPooling2D(name="b1pep_hpho_3_pool", keepdims=True)(b1pep_3_hpho_CNN)
    b1pep_5_hpho_pool = layers.GlobalMaxPooling2D(name="b1pep_hpho_5_pool", keepdims=True)(b1pep_5_hpho_CNN)
    b1pep_1_hphi_pool = layers.GlobalMaxPooling2D(name="b1pep_hphi_1_pool", keepdims=True)(b1pep_1_hphi_CNN)
    b1pep_3_hphi_pool = layers.GlobalMaxPooling2D(name="b1pep_hphi_3_pool", keepdims=True)(b1pep_3_hphi_CNN)
    b1pep_5_hphi_pool = layers.GlobalMaxPooling2D(name="b1pep_hphi_5_pool", keepdims=True)(b1pep_5_hphi_CNN)
    b1pep_1_blosum_pool = layers.GlobalMaxPooling2D(name="b1pep_blosum_1_pool", keepdims=True)(b1pep_1_blosum_CNN)
    b1pep_3_blosum_pool = layers.GlobalMaxPooling2D(name="b1pep_blosum_3_pool", keepdims=True)(b1pep_3_blosum_CNN)
    b1pep_5_blosum_pool = layers.GlobalMaxPooling2D(name="b1pep_blosum_5_pool", keepdims=True)(b1pep_5_blosum_CNN)

    # GlobalMaxPooling for the b2 maps
    b2pep_1_mass_pool = layers.GlobalMaxPooling2D(name="b2pep_mass_1_pool", keepdims=True)(b2pep_1_mass_CNN)
    b2pep_3_mass_pool = layers.GlobalMaxPooling2D(name="b2pep_mass_3_pool", keepdims=True)(b2pep_3_mass_CNN)
    b2pep_5_mass_pool = layers.GlobalMaxPooling2D(name="b2pep_mass_5_pool", keepdims=True)(b2pep_5_mass_CNN)
    b2pep_1_ip_pool = layers.GlobalMaxPooling2D(name="b2pep_ip_1_pool", keepdims=True)(b2pep_1_ip_CNN)
    b2pep_3_ip_pool = layers.GlobalMaxPooling2D(name="b2pep_ip_3_pool", keepdims=True)(b2pep_3_ip_CNN)
    b2pep_5_ip_pool = layers.GlobalMaxPooling2D(name="b2pep_ip_5_pool", keepdims=True)(b2pep_5_ip_CNN)
    b2pep_1_hpho_pool = layers.GlobalMaxPooling2D(name="b2pep_hpho_1_pool", keepdims=True)(b2pep_1_hpho_CNN)
    b2pep_3_hpho_pool = layers.GlobalMaxPooling2D(name="b2pep_hpho_3_pool", keepdims=True)(b2pep_3_hpho_CNN)
    b2pep_5_hpho_pool = layers.GlobalMaxPooling2D(name="b2pep_hpho_5_pool", keepdims=True)(b2pep_5_hpho_CNN)
    b2pep_1_hphi_pool = layers.GlobalMaxPooling2D(name="b2pep_hphi_1_pool", keepdims=True)(b2pep_1_hphi_CNN)
    b2pep_3_hphi_pool = layers.GlobalMaxPooling2D(name="b2pep_hphi_3_pool", keepdims=True)(b2pep_3_hphi_CNN)
    b2pep_5_hphi_pool = layers.GlobalMaxPooling2D(name="b2pep_hphi_5_pool", keepdims=True)(b2pep_5_hphi_CNN)
    b2pep_1_blosum_pool = layers.GlobalMaxPooling2D(name="b2pep_blosum_1_pool", keepdims=True)(b2pep_1_blosum_CNN)
    b2pep_3_blosum_pool = layers.GlobalMaxPooling2D(name="b2pep_blosum_3_pool", keepdims=True)(b2pep_3_blosum_CNN)
    b2pep_5_blosum_pool = layers.GlobalMaxPooling2D(name="b2pep_blosum_5_pool", keepdims=True)(b2pep_5_blosum_CNN)

    # GlobalMaxPooling for the b3 maps
    b3pep_1_mass_pool = layers.GlobalMaxPooling2D(name="b3pep_mass_1_pool", keepdims=True)(b3pep_1_mass_CNN)
    b3pep_3_mass_pool = layers.GlobalMaxPooling2D(name="b3pep_mass_3_pool", keepdims=True)(b3pep_3_mass_CNN)
    b3pep_5_mass_pool = layers.GlobalMaxPooling2D(name="b3pep_mass_5_pool", keepdims=True)(b3pep_5_mass_CNN)
    b3pep_7_mass_pool = layers.GlobalMaxPooling2D(name="b3pep_mass_7_pool", keepdims=True)(b3pep_7_mass_CNN)
    b3pep_9_mass_pool = layers.GlobalMaxPooling2D(name="b3pep_mass_9_pool", keepdims=True)(b3pep_9_mass_CNN)
    b3pep_1_ip_pool = layers.GlobalMaxPooling2D(name="b3pep_ip_1_pool", keepdims=True)(b3pep_1_ip_CNN)
    b3pep_3_ip_pool = layers.GlobalMaxPooling2D(name="b3pep_ip_3_pool", keepdims=True)(b3pep_3_ip_CNN)
    b3pep_5_ip_pool = layers.GlobalMaxPooling2D(name="b3pep_ip_5_pool", keepdims=True)(b3pep_5_ip_CNN)
    b3pep_7_ip_pool = layers.GlobalMaxPooling2D(name="b3pep_ip_7_pool", keepdims=True)(b3pep_7_ip_CNN)
    b3pep_9_ip_pool = layers.GlobalMaxPooling2D(name="b3pep_ip_9_pool", keepdims=True)(b3pep_9_ip_CNN)
    b3pep_1_hpho_pool = layers.GlobalMaxPooling2D(name="b3pep_hpho_1_pool", keepdims=True)(b3pep_1_hpho_CNN)
    b3pep_3_hpho_pool = layers.GlobalMaxPooling2D(name="b3pep_hpho_3_pool", keepdims=True)(b3pep_3_hpho_CNN)
    b3pep_5_hpho_pool = layers.GlobalMaxPooling2D(name="b3pep_hpho_5_pool", keepdims=True)(b3pep_5_hpho_CNN)
    b3pep_7_hpho_pool = layers.GlobalMaxPooling2D(name="b3pep_hpho_7_pool", keepdims=True)(b3pep_7_hpho_CNN)
    b3pep_9_hpho_pool = layers.GlobalMaxPooling2D(name="b3pep_hpho_9_pool", keepdims=True)(b3pep_9_hpho_CNN)
    b3pep_1_hphi_pool = layers.GlobalMaxPooling2D(name="b3pep_hphi_1_pool", keepdims=True)(b3pep_1_hphi_CNN)
    b3pep_3_hphi_pool = layers.GlobalMaxPooling2D(name="b3pep_hphi_3_pool", keepdims=True)(b3pep_3_hphi_CNN)
    b3pep_5_hphi_pool = layers.GlobalMaxPooling2D(name="b3pep_hphi_5_pool", keepdims=True)(b3pep_5_hphi_CNN)
    b3pep_7_hphi_pool = layers.GlobalMaxPooling2D(name="b3pep_hphi_7_pool", keepdims=True)(b3pep_7_hphi_CNN)
    b3pep_9_hphi_pool = layers.GlobalMaxPooling2D(name="b3pep_hphi_9_pool", keepdims=True)(b3pep_9_hphi_CNN)
    b3pep_1_blosum_pool = layers.GlobalMaxPooling2D(name="b3pep_blosum_1_pool", keepdims=True)(b3pep_1_blosum_CNN)
    b3pep_3_blosum_pool = layers.GlobalMaxPooling2D(name="b3pep_blosum_3_pool", keepdims=True)(b3pep_3_blosum_CNN)
    b3pep_5_blosum_pool = layers.GlobalMaxPooling2D(name="b3pep_blosum_5_pool", keepdims=True)(b3pep_5_blosum_CNN)
    b3pep_7_blosum_pool = layers.GlobalMaxPooling2D(name="b3pep_blosum_7_pool", keepdims=True)(b3pep_7_blosum_CNN)
    b3pep_9_blosum_pool = layers.GlobalMaxPooling2D(name="b3pep_blosum_9_pool", keepdims=True)(b3pep_9_blosum_CNN)

    # Flatten for the a1 maps
    a1pep_1_mass_flatten = layers.Flatten(name="a1pep_mass_1_flatten")(a1pep_1_mass_pool)
    a1pep_3_mass_flatten = layers.Flatten(name="a1pep_mass_3_flatten")(a1pep_3_mass_pool)
    a1pep_5_mass_flatten = layers.Flatten(name="a1pep_mass_5_flatten")(a1pep_5_mass_pool)
    a1pep_1_ip_flatten = layers.Flatten(name="a1pep_ip_1_flatten")(a1pep_1_ip_pool)
    a1pep_3_ip_flatten = layers.Flatten(name="a1pep_ip_3_flatten")(a1pep_3_ip_pool)
    a1pep_5_ip_flatten = layers.Flatten(name="a1pep_ip_5_flatten")(a1pep_5_ip_pool)
    a1pep_1_hpho_flatten = layers.Flatten(name="a1pep_hpho_1_flatten")(a1pep_1_hpho_pool)
    a1pep_3_hpho_flatten = layers.Flatten(name="a1pep_hpho_3_flatten")(a1pep_3_hpho_pool)
    a1pep_5_hpho_flatten = layers.Flatten(name="a1pep_hpho_5_flatten")(a1pep_5_hpho_pool)
    a1pep_1_hphi_flatten = layers.Flatten(name="a1pep_hphi_1_flatten")(a1pep_1_hphi_pool)
    a1pep_3_hphi_flatten = layers.Flatten(name="a1pep_hphi_3_flatten")(a1pep_3_hphi_pool)
    a1pep_5_hphi_flatten = layers.Flatten(name="a1pep_hphi_5_flatten")(a1pep_5_hphi_pool)
    a1pep_1_blosum_flatten = layers.Flatten(name="a1pep_blosum_1_flatten")(a1pep_1_blosum_pool)
    a1pep_3_blosum_flatten = layers.Flatten(name="a1pep_blosum_3_flatten")(a1pep_3_blosum_pool)
    a1pep_5_blosum_flatten = layers.Flatten(name="a1pep_blosum_5_flatten")(a1pep_5_blosum_pool)

    # Flatten for the a2 maps
    a2pep_1_mass_flatten = layers.Flatten(name="a2pep_mass_1_flatten")(a2pep_1_mass_pool)
    a2pep_3_mass_flatten = layers.Flatten(name="a2pep_mass_3_flatten")(a2pep_3_mass_pool)
    a2pep_5_mass_flatten = layers.Flatten(name="a2pep_mass_5_flatten")(a2pep_5_mass_pool)
    a2pep_1_ip_flatten = layers.Flatten(name="a2pep_ip_1_flatten")(a2pep_1_ip_pool)
    a2pep_3_ip_flatten = layers.Flatten(name="a2pep_ip_3_flatten")(a2pep_3_ip_pool)
    a2pep_5_ip_flatten = layers.Flatten(name="a2pep_ip_5_flatten")(a2pep_5_ip_pool)
    a2pep_1_hpho_flatten = layers.Flatten(name="a2pep_hpho_1_flatten")(a2pep_1_hpho_pool)
    a2pep_3_hpho_flatten = layers.Flatten(name="a2pep_hpho_3_flatten")(a2pep_3_hpho_pool)
    a2pep_5_hpho_flatten = layers.Flatten(name="a2pep_hpho_5_flatten")(a2pep_5_hpho_pool)
    a2pep_1_hphi_flatten = layers.Flatten(name="a2pep_hphi_1_flatten")(a2pep_1_hphi_pool)
    a2pep_3_hphi_flatten = layers.Flatten(name="a2pep_hphi_3_flatten")(a2pep_3_hphi_pool)
    a2pep_5_hphi_flatten = layers.Flatten(name="a2pep_hphi_5_flatten")(a2pep_5_hphi_pool)
    a2pep_1_blosum_flatten = layers.Flatten(name="a2pep_blosum_1_flatten")(a2pep_1_blosum_pool)
    a2pep_3_blosum_flatten = layers.Flatten(name="a2pep_blosum_3_flatten")(a2pep_3_blosum_pool)
    a2pep_5_blosum_flatten = layers.Flatten(name="a2pep_blosum_5_flatten")(a2pep_5_blosum_pool)

    # Flatten for the a3 maps
    a3pep_1_mass_flatten = layers.Flatten(name="a3pep_mass_1_flatten")(a3pep_1_mass_pool)
    a3pep_3_mass_flatten = layers.Flatten(name="a3pep_mass_3_flatten")(a3pep_3_mass_pool)
    a3pep_5_mass_flatten = layers.Flatten(name="a3pep_mass_5_flatten")(a3pep_5_mass_pool)
    a3pep_7_mass_flatten = layers.Flatten(name="a3pep_mass_7_flatten")(a3pep_7_mass_pool)
    a3pep_9_mass_flatten = layers.Flatten(name="a3pep_mass_9_flatten")(a3pep_9_mass_pool)
    a3pep_1_ip_flatten = layers.Flatten(name="a3pep_ip_1_flatten")(a3pep_1_ip_pool)
    a3pep_3_ip_flatten = layers.Flatten(name="a3pep_ip_3_flatten")(a3pep_3_ip_pool)
    a3pep_5_ip_flatten = layers.Flatten(name="a3pep_ip_5_flatten")(a3pep_5_ip_pool)
    a3pep_7_ip_flatten = layers.Flatten(name="a3pep_ip_7_flatten")(a3pep_7_ip_pool)
    a3pep_9_ip_flatten = layers.Flatten(name="a3pep_ip_9_flatten")(a3pep_9_ip_pool)
    a3pep_1_hpho_flatten = layers.Flatten(name="a3pep_hpho_1_flatten")(a3pep_1_hpho_pool)
    a3pep_3_hpho_flatten = layers.Flatten(name="a3pep_hpho_3_flatten")(a3pep_3_hpho_pool)
    a3pep_5_hpho_flatten = layers.Flatten(name="a3pep_hpho_5_flatten")(a3pep_5_hpho_pool)
    a3pep_7_hpho_flatten = layers.Flatten(name="a3pep_hpho_7_flatten")(a3pep_7_hpho_pool)
    a3pep_9_hpho_flatten = layers.Flatten(name="a3pep_hpho_9_flatten")(a3pep_9_hpho_pool)
    a3pep_1_hphi_flatten = layers.Flatten(name="a3pep_hphi_1_flatten")(a3pep_1_hphi_pool)
    a3pep_3_hphi_flatten = layers.Flatten(name="a3pep_hphi_3_flatten")(a3pep_3_hphi_pool)
    a3pep_5_hphi_flatten = layers.Flatten(name="a3pep_hphi_5_flatten")(a3pep_5_hphi_pool)
    a3pep_7_hphi_flatten = layers.Flatten(name="a3pep_hphi_7_flatten")(a3pep_7_hphi_pool)
    a3pep_9_hphi_flatten = layers.Flatten(name="a3pep_hphi_9_flatten")(a3pep_9_hphi_pool)
    a3pep_1_blosum_flatten = layers.Flatten(name="a3pep_blosum_1_flatten")(a3pep_1_blosum_pool)
    a3pep_3_blosum_flatten = layers.Flatten(name="a3pep_blosum_3_flatten")(a3pep_3_blosum_pool)
    a3pep_5_blosum_flatten = layers.Flatten(name="a3pep_blosum_5_flatten")(a3pep_5_blosum_pool)
    a3pep_7_blosum_flatten = layers.Flatten(name="a3pep_blosum_7_flatten")(a3pep_7_blosum_pool)
    a3pep_9_blosum_flatten = layers.Flatten(name="a3pep_blosum_9_flatten")(a3pep_9_blosum_pool)

    # Flatten for the b1 maps
    b1pep_1_mass_flatten = layers.Flatten(name="b1pep_mass_1_flatten")(b1pep_1_mass_pool)
    b1pep_3_mass_flatten = layers.Flatten(name="b1pep_mass_3_flatten")(b1pep_3_mass_pool)
    b1pep_5_mass_flatten = layers.Flatten(name="b1pep_mass_5_flatten")(b1pep_5_mass_pool)
    b1pep_1_ip_flatten = layers.Flatten(name="b1pep_ip_1_flatten")(b1pep_1_ip_pool)
    b1pep_3_ip_flatten = layers.Flatten(name="b1pep_ip_3_flatten")(b1pep_3_ip_pool)
    b1pep_5_ip_flatten = layers.Flatten(name="b1pep_ip_5_flatten")(b1pep_5_ip_pool)
    b1pep_1_hpho_flatten = layers.Flatten(name="b1pep_hpho_1_flatten")(b1pep_1_hpho_pool)
    b1pep_3_hpho_flatten = layers.Flatten(name="b1pep_hpho_3_flatten")(b1pep_3_hpho_pool)
    b1pep_5_hpho_flatten = layers.Flatten(name="b1pep_hpho_5_flatten")(b1pep_5_hpho_pool)
    b1pep_1_hphi_flatten = layers.Flatten(name="b1pep_hphi_1_flatten")(b1pep_1_hphi_pool)
    b1pep_3_hphi_flatten = layers.Flatten(name="b1pep_hphi_3_flatten")(b1pep_3_hphi_pool)
    b1pep_5_hphi_flatten = layers.Flatten(name="b1pep_hphi_5_flatten")(b1pep_5_hphi_pool)
    b1pep_1_blosum_flatten = layers.Flatten(name="b1pep_blosum_1_flatten")(b1pep_1_blosum_pool)
    b1pep_3_blosum_flatten = layers.Flatten(name="b1pep_blosum_3_flatten")(b1pep_3_blosum_pool)
    b1pep_5_blosum_flatten = layers.Flatten(name="b1pep_blosum_5_flatten")(b1pep_5_blosum_pool)

    # Flatten for the b2 maps
    b2pep_1_mass_flatten = layers.Flatten(name="b2pep_mass_1_flatten")(b2pep_1_mass_pool)
    b2pep_3_mass_flatten = layers.Flatten(name="b2pep_mass_3_flatten")(b2pep_3_mass_pool)
    b2pep_5_mass_flatten = layers.Flatten(name="b2pep_mass_5_flatten")(b2pep_5_mass_pool)
    b2pep_1_ip_flatten = layers.Flatten(name="b2pep_ip_1_flatten")(b2pep_1_ip_pool)
    b2pep_3_ip_flatten = layers.Flatten(name="b2pep_ip_3_flatten")(b2pep_3_ip_pool)
    b2pep_5_ip_flatten = layers.Flatten(name="b2pep_ip_5_flatten")(b2pep_5_ip_pool)
    b2pep_1_hpho_flatten = layers.Flatten(name="b2pep_hpho_1_flatten")(b2pep_1_hpho_pool)
    b2pep_3_hpho_flatten = layers.Flatten(name="b2pep_hpho_3_flatten")(b2pep_3_hpho_pool)
    b2pep_5_hpho_flatten = layers.Flatten(name="b2pep_hpho_5_flatten")(b2pep_5_hpho_pool)
    b2pep_1_hphi_flatten = layers.Flatten(name="b2pep_hphi_1_flatten")(b2pep_1_hphi_pool)
    b2pep_3_hphi_flatten = layers.Flatten(name="b2pep_hphi_3_flatten")(b2pep_3_hphi_pool)
    b2pep_5_hphi_flatten = layers.Flatten(name="b2pep_hphi_5_flatten")(b2pep_5_hphi_pool)
    b2pep_1_blosum_flatten = layers.Flatten(name="b2pep_blosum_1_flatten")(b2pep_1_blosum_pool)
    b2pep_3_blosum_flatten = layers.Flatten(name="b2pep_blosum_3_flatten")(b2pep_3_blosum_pool)
    b2pep_5_blosum_flatten = layers.Flatten(name="b2pep_blosum_5_flatten")(b2pep_5_blosum_pool)

    # Flatten for the b3 maps
    b3pep_1_mass_flatten = layers.Flatten(name="b3pep_mass_1_flatten")(b3pep_1_mass_pool)
    b3pep_3_mass_flatten = layers.Flatten(name="b3pep_mass_3_flatten")(b3pep_3_mass_pool)
    b3pep_5_mass_flatten = layers.Flatten(name="b3pep_mass_5_flatten")(b3pep_5_mass_pool)
    b3pep_7_mass_flatten = layers.Flatten(name="b3pep_mass_7_flatten")(b3pep_7_mass_pool)
    b3pep_9_mass_flatten = layers.Flatten(name="b3pep_mass_9_flatten")(b3pep_9_mass_pool)
    b3pep_1_ip_flatten = layers.Flatten(name="b3pep_ip_1_flatten")(b3pep_1_ip_pool)
    b3pep_3_ip_flatten = layers.Flatten(name="b3pep_ip_3_flatten")(b3pep_3_ip_pool)
    b3pep_5_ip_flatten = layers.Flatten(name="b3pep_ip_5_flatten")(b3pep_5_ip_pool)
    b3pep_7_ip_flatten = layers.Flatten(name="b3pep_ip_7_flatten")(b3pep_7_ip_pool)
    b3pep_9_ip_flatten = layers.Flatten(name="b3pep_ip_9_flatten")(b3pep_9_ip_pool)
    b3pep_1_hpho_flatten = layers.Flatten(name="b3pep_hpho_1_flatten")(b3pep_1_hpho_pool)
    b3pep_3_hpho_flatten = layers.Flatten(name="b3pep_hpho_3_flatten")(b3pep_3_hpho_pool)
    b3pep_5_hpho_flatten = layers.Flatten(name="b3pep_hpho_5_flatten")(b3pep_5_hpho_pool)
    b3pep_7_hpho_flatten = layers.Flatten(name="b3pep_hpho_7_flatten")(b3pep_7_hpho_pool)
    b3pep_9_hpho_flatten = layers.Flatten(name="b3pep_hpho_9_flatten")(b3pep_9_hpho_pool)
    b3pep_1_hphi_flatten = layers.Flatten(name="b3pep_hphi_1_flatten")(b3pep_1_hphi_pool)
    b3pep_3_hphi_flatten = layers.Flatten(name="b3pep_hphi_3_flatten")(b3pep_3_hphi_pool)
    b3pep_5_hphi_flatten = layers.Flatten(name="b3pep_hphi_5_flatten")(b3pep_5_hphi_pool)
    b3pep_7_hphi_flatten = layers.Flatten(name="b3pep_hphi_7_flatten")(b3pep_7_hphi_pool)
    b3pep_9_hphi_flatten = layers.Flatten(name="b3pep_hphi_9_flatten")(b3pep_9_hphi_pool)
    b3pep_1_blosum_flatten = layers.Flatten(name="b3pep_blosum_1_flatten")(b3pep_1_blosum_pool)
    b3pep_3_blosum_flatten = layers.Flatten(name="b3pep_blosum_3_flatten")(b3pep_3_blosum_pool)
    b3pep_5_blosum_flatten = layers.Flatten(name="b3pep_blosum_5_flatten")(b3pep_5_blosum_pool)
    b3pep_7_blosum_flatten = layers.Flatten(name="b3pep_blosum_7_flatten")(b3pep_7_blosum_pool)
    b3pep_9_blosum_flatten = layers.Flatten(name="b3pep_blosum_9_flatten")(b3pep_9_blosum_pool)

    # Concatenation of all MaxPool outputs from all features and kernel-sizes
    cat = layers.Concatenate(name = "first_cat")([
            a1pep_1_mass_flatten, a1pep_3_mass_flatten, a1pep_5_mass_flatten,
            a1pep_1_ip_flatten, a1pep_3_ip_flatten, a1pep_5_ip_flatten,
            a1pep_1_hpho_flatten, a1pep_3_hpho_flatten, a1pep_5_hpho_flatten,
            a1pep_1_hphi_flatten, a1pep_3_hphi_flatten, a1pep_5_hphi_flatten,
            a1pep_1_blosum_flatten, a1pep_3_blosum_flatten, a1pep_5_blosum_flatten,
            
            a2pep_1_mass_flatten, a2pep_3_mass_flatten, a2pep_5_mass_flatten,
            a2pep_1_ip_flatten, a2pep_3_ip_flatten, a2pep_5_ip_flatten,
            a2pep_1_hpho_flatten, a2pep_3_hpho_flatten, a2pep_5_hpho_flatten,
            a2pep_1_hphi_flatten, a2pep_3_hphi_flatten, a2pep_5_hphi_flatten,
            a2pep_1_blosum_flatten, a2pep_3_blosum_flatten, a2pep_5_blosum_flatten,
            
            a3pep_1_mass_flatten, a3pep_3_mass_flatten, a3pep_5_mass_flatten, a3pep_7_mass_flatten, a3pep_9_mass_flatten,
            a3pep_1_ip_flatten, a3pep_3_ip_flatten, a3pep_5_ip_flatten, a3pep_7_ip_flatten, a3pep_9_ip_flatten,
            a3pep_1_hpho_flatten, a3pep_3_hpho_flatten, a3pep_5_hpho_flatten, a3pep_7_hpho_flatten, a3pep_9_hpho_flatten,
            a3pep_1_hphi_flatten, a3pep_3_hphi_flatten, a3pep_5_hphi_flatten, a3pep_7_hphi_flatten, a3pep_9_hphi_flatten,
            a3pep_1_blosum_flatten, a3pep_3_blosum_flatten, a3pep_5_blosum_flatten, a3pep_7_blosum_flatten, a3pep_9_blosum_flatten,
            
            b1pep_1_mass_flatten, b1pep_3_mass_flatten, b1pep_5_mass_flatten,
            b1pep_1_ip_flatten, b1pep_3_ip_flatten, b1pep_5_ip_flatten,
            b1pep_1_hpho_flatten, b1pep_3_hpho_flatten, b1pep_5_hpho_flatten,
            b1pep_1_hphi_flatten, b1pep_3_hphi_flatten, b1pep_5_hphi_flatten,
            b1pep_1_blosum_flatten, b1pep_3_blosum_flatten, b1pep_5_blosum_flatten,
            
            b2pep_1_mass_flatten, b2pep_3_mass_flatten, b2pep_5_mass_flatten,
            b2pep_1_ip_flatten, b2pep_3_ip_flatten, b2pep_5_ip_flatten,
            b2pep_1_hpho_flatten, b2pep_3_hpho_flatten, b2pep_5_hpho_flatten,
            b2pep_1_hphi_flatten, b2pep_3_hphi_flatten, b2pep_5_hphi_flatten,
            b2pep_1_blosum_flatten, b2pep_3_blosum_flatten, b2pep_5_blosum_flatten,
            
            b3pep_1_mass_flatten, b3pep_3_mass_flatten, b3pep_5_mass_flatten, b3pep_7_mass_flatten, b3pep_9_mass_flatten,
            b3pep_1_ip_flatten, b3pep_3_ip_flatten, b3pep_5_ip_flatten, b3pep_7_ip_flatten, b3pep_9_ip_flatten,
            b3pep_1_hpho_flatten, b3pep_3_hpho_flatten, b3pep_5_hpho_flatten, b3pep_7_hpho_flatten, b3pep_9_hpho_flatten,
            b3pep_1_hphi_flatten, b3pep_3_hphi_flatten, b3pep_5_hphi_flatten, b3pep_7_hphi_flatten, b3pep_9_hphi_flatten,
            b3pep_1_blosum_flatten, b3pep_3_blosum_flatten, b3pep_5_blosum_flatten, b3pep_7_blosum_flatten, b3pep_9_blosum_flatten])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)
    
    # Dense layers after concatenation+dropout
    dense = layers.Dense(64, activation = dense_activation, kernel_regularizer = l2(l2_reg), name = "first_dense")(cat_dropout)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)
    
    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)
    
    # Model definition
    model = keras.Model(inputs = [a1pep_mass, a1pep_ip, a1pep_hpho, a1pep_hphi, a1pep_blosum,
                                  a2pep_mass, a2pep_ip, a2pep_hpho, a2pep_hphi, a2pep_blosum,
                                  a3pep_mass, a3pep_ip, a3pep_hpho, a3pep_hphi, a3pep_blosum,
                                  b1pep_mass, b1pep_ip, b1pep_hpho, b1pep_hphi, b1pep_blosum,
                                  b2pep_mass, b2pep_ip, b2pep_hpho, b2pep_hphi, b2pep_blosum,
                                  b3pep_mass, b3pep_ip, b3pep_hpho, b3pep_hphi, b3pep_blosum],
                        outputs = out)
    
    return model

# 'Old' redefined 2D baseline (interaction maps) architecture for CDR3 paired-chain sequences
def CNN_CDR123_2D_old_redefined(dropout_rate, n_maps, seed, conv_activation = "relu", dense_activation = "relu", 
                                nr_of_filters_1 = 32, max_lengths = None, l2_reg = 0.001):

    # Max.length of the sequences from the dataset
    if max_lengths:
        a1_max = max_lengths[0]
        a2_max = max_lengths[1]
        a3_max = max_lengths[2]
        b1_max = max_lengths[3]
        b2_max = max_lengths[4]
        b3_max = max_lengths[5]
        pep_max = max_lengths[6]
    else:
        a1_max = 7
        a2_max = 8
        a3_max = 22
        b1_max = 6
        b2_max = 7
        b3_max = 23
        pep_max = 12

    # 2D-Input dimensions
    a1 = keras.Input(shape = (a1_max, pep_max, n_maps), name ="a1")
    b1 = keras.Input(shape = (b1_max, pep_max, n_maps), name ="b1")
    a2 = keras.Input(shape = (a2_max, pep_max, n_maps), name ="a2")
    b2 = keras.Input(shape = (b2_max, pep_max, n_maps), name ="b2")
    a3 = keras.Input(shape = (a3_max, pep_max, n_maps), name ="a3")
    b3 = keras.Input(shape = (b3_max, pep_max, n_maps), name ="b3")

    # Trained in a pan-specific setup (padding="same" for keeping original input sizes)

    # CNN layers for each feature and the different 5 kernel-sizes
    a1_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a1_1_conv")(a1)
    a1_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a1_3_conv")(a1)
    a1_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a1_5_conv")(a1)
    # a1_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_a1_7_conv")(a1)
    # a1_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_a1_9_conv")(a1)

    b1_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b1_1_conv")(b1)
    b1_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b1_3_conv")(b1)
    b1_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b1_5_conv")(b1)
    # b1_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_b1_7_conv")(b1)
    # b1_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_b1_9_conv")(b1)

    a2_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a2_1_conv")(a2)
    a2_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a2_3_conv")(a2)
    a2_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a2_5_conv")(a2)
    # a2_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_a2_7_conv")(a2)
    # a2_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_a2_9_conv")(a2)

    b2_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b2_1_conv")(b2)
    b2_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b2_3_conv")(b2)
    b2_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b2_5_conv")(b2)
    # b2_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_b2_7_conv")(b2)
    # b2_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_b2_9_conv")(b2) 

    a3_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    a3_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_a3_7_conv")(a3)
    a3_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_a3_9_conv")(a3)

    b3_1_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (1, 1), padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_3_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (3, 3), padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_5_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (5, 5), padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3)
    b3_7_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (7, 7), padding = "same", activation = conv_activation, name = "first_b3_7_conv")(b3)
    b3_9_CNN = layers.Conv2D(filters = nr_of_filters_1, kernel_size = (9, 9), padding = "same", activation = conv_activation, name = "first_b3_9_conv")(b3) 

    # Batch Normalization
    a1_1_CNN = layers.BatchNormalization()(a1_1_CNN)
    a1_3_CNN = layers.BatchNormalization()(a1_3_CNN)
    a1_5_CNN = layers.BatchNormalization()(a1_5_CNN)
    # a1_7_CNN = layers.BatchNormalization()(a1_7_CNN)
    # a1_9_CNN = layers.BatchNormalization()(a1_9_CNN)

    b1_1_CNN = layers.BatchNormalization()(b1_1_CNN)
    b1_3_CNN = layers.BatchNormalization()(b1_3_CNN)
    b1_5_CNN = layers.BatchNormalization()(b1_5_CNN)
    # b1_7_CNN = layers.BatchNormalization()(b1_7_CNN)
    # b1_9_CNN = layers.BatchNormalization()(b1_9_CNN)

    a2_1_CNN = layers.BatchNormalization()(a2_1_CNN)
    a2_3_CNN = layers.BatchNormalization()(a2_3_CNN)
    a2_5_CNN = layers.BatchNormalization()(a2_5_CNN)
    # a2_7_CNN = layers.BatchNormalization()(a2_7_CNN)
    # a2_9_CNN = layers.BatchNormalization()(a2_9_CNN)

    b2_1_CNN = layers.BatchNormalization()(b2_1_CNN)
    b2_3_CNN = layers.BatchNormalization()(b2_3_CNN)
    b2_5_CNN = layers.BatchNormalization()(b2_5_CNN)
    # b2_7_CNN = layers.BatchNormalization()(b2_7_CNN)
    # b2_9_CNN = layers.BatchNormalization()(b2_9_CNN)

    a3_1_CNN = layers.BatchNormalization()(a3_1_CNN)
    a3_3_CNN = layers.BatchNormalization()(a3_3_CNN)
    a3_5_CNN = layers.BatchNormalization()(a3_5_CNN)
    a3_7_CNN = layers.BatchNormalization()(a3_7_CNN)
    a3_9_CNN = layers.BatchNormalization()(a3_9_CNN)

    b3_1_CNN = layers.BatchNormalization()(b3_1_CNN)
    b3_3_CNN = layers.BatchNormalization()(b3_3_CNN)
    b3_5_CNN = layers.BatchNormalization()(b3_5_CNN)
    b3_7_CNN = layers.BatchNormalization()(b3_7_CNN)
    b3_9_CNN = layers.BatchNormalization()(b3_9_CNN)
    
    # GlobalMaxPooling
    a1_1_pool = layers.GlobalMaxPooling2D(name = "first_a1_1_pool", keepdims=True)(a1_1_CNN)
    a1_3_pool = layers.GlobalMaxPooling2D(name = "first_a1_3_pool", keepdims=True)(a1_3_CNN)
    a1_5_pool = layers.GlobalMaxPooling2D(name = "first_a1_5_pool", keepdims=True)(a1_5_CNN)
    # a1_7_pool = layers.GlobalMaxPooling2D(name = "first_a1_7_pool", keepdims=True)(a1_7_CNN)
    # a1_9_pool = layers.GlobalMaxPooling2D(name = "first_a1_9_pool", keepdims=True)(a1_9_CNN)
      
    b1_1_pool = layers.GlobalMaxPooling2D(name = "first_b1_1_pool", keepdims=True)(b1_1_CNN)
    b1_3_pool = layers.GlobalMaxPooling2D(name = "first_b1_3_pool", keepdims=True)(b1_3_CNN)
    b1_5_pool = layers.GlobalMaxPooling2D(name = "first_b1_5_pool", keepdims=True)(b1_5_CNN)
    # b1_7_pool = layers.GlobalMaxPooling2D(name = "first_b1_7_pool", keepdims=True)(b1_7_CNN)
    # b1_9_pool = layers.GlobalMaxPooling2D(name = "first_b1_9_pool", keepdims=True)(b1_9_CNN)

    a2_1_pool = layers.GlobalMaxPooling2D(name = "first_a2_1_pool", keepdims=True)(a2_1_CNN)
    a2_3_pool = layers.GlobalMaxPooling2D(name = "first_a2_3_pool", keepdims=True)(a2_3_CNN)
    a2_5_pool = layers.GlobalMaxPooling2D(name = "first_a2_5_pool", keepdims=True)(a2_5_CNN)
    # a2_7_pool = layers.GlobalMaxPooling2D(name = "first_a2_7_pool", keepdims=True)(a2_7_CNN)
    # a2_9_pool = layers.GlobalMaxPooling2D(name = "first_a2_9_pool", keepdims=True)(a2_9_CNN)
      
    b2_1_pool = layers.GlobalMaxPooling2D(name = "first_b2_1_pool", keepdims=True)(b2_1_CNN)
    b2_3_pool = layers.GlobalMaxPooling2D(name = "first_b2_3_pool", keepdims=True)(b2_3_CNN)
    b2_5_pool = layers.GlobalMaxPooling2D(name = "first_b2_5_pool", keepdims=True)(b2_5_CNN)
    # b2_7_pool = layers.GlobalMaxPooling2D(name = "first_b2_7_pool", keepdims=True)(b2_7_CNN)
    # b2_9_pool = layers.GlobalMaxPooling2D(name = "first_b2_9_pool", keepdims=True)(b2_9_CNN)

    a3_1_pool = layers.GlobalMaxPooling2D(name = "first_a3_1_pool", keepdims=True)(a3_1_CNN)
    a3_3_pool = layers.GlobalMaxPooling2D(name = "first_a3_3_pool", keepdims=True)(a3_3_CNN)
    a3_5_pool = layers.GlobalMaxPooling2D(name = "first_a3_5_pool", keepdims=True)(a3_5_CNN)
    a3_7_pool = layers.GlobalMaxPooling2D(name = "first_a3_7_pool", keepdims=True)(a3_7_CNN)
    a3_9_pool = layers.GlobalMaxPooling2D(name = "first_a3_9_pool", keepdims=True)(a3_9_CNN)
      
    b3_1_pool = layers.GlobalMaxPooling2D(name = "first_b3_1_pool", keepdims=True)(b3_1_CNN)
    b3_3_pool = layers.GlobalMaxPooling2D(name = "first_b3_3_pool", keepdims=True)(b3_3_CNN)
    b3_5_pool = layers.GlobalMaxPooling2D(name = "first_b3_5_pool", keepdims=True)(b3_5_CNN)
    b3_7_pool = layers.GlobalMaxPooling2D(name = "first_b3_7_pool", keepdims=True)(b3_7_CNN)
    b3_9_pool = layers.GlobalMaxPooling2D(name = "first_b3_9_pool", keepdims=True)(b3_9_CNN)

    # Flatten the layers
    a1_1_flatten = layers.Flatten(name="flatten_a1_1")(a1_1_pool)
    a1_3_flatten = layers.Flatten(name="flatten_a1_3")(a1_3_pool)
    a1_5_flatten = layers.Flatten(name="flatten_a1_5")(a1_5_pool)
    # a1_7_flatten = layers.Flatten(name="flatten_a1_7")(a1_7_pool)
    # a1_9_flatten = layers.Flatten(name="flatten_a1_9")(a1_9_pool)

    b1_1_flatten = layers.Flatten(name="flatten_b1_1")(b1_1_pool)
    b1_3_flatten = layers.Flatten(name="flatten_b1_3")(b1_3_pool)
    b1_5_flatten = layers.Flatten(name="flatten_b1_5")(b1_5_pool)
    # b1_7_flatten = layers.Flatten(name="flatten_b1_7")(b1_7_pool)
    # b1_9_flatten = layers.Flatten(name="flatten_b1_9")(b1_9_pool)

    a2_1_flatten = layers.Flatten(name="flatten_a2_1")(a2_1_pool)
    a2_3_flatten = layers.Flatten(name="flatten_a2_3")(a2_3_pool)
    a2_5_flatten = layers.Flatten(name="flatten_a2_5")(a2_5_pool)
    # a2_7_flatten = layers.Flatten(name="flatten_a2_7")(a2_7_pool)
    # a2_9_flatten = layers.Flatten(name="flatten_a2_9")(a2_9_pool)

    b2_1_flatten = layers.Flatten(name="flatten_b2_1")(b2_1_pool)
    b2_3_flatten = layers.Flatten(name="flatten_b2_2")(b2_3_pool)
    b2_5_flatten = layers.Flatten(name="flatten_b2_3")(b2_5_pool)
    # b2_7_flatten = layers.Flatten(name="flatten_b2_4")(b2_7_pool)
    # b2_9_flatten = layers.Flatten(name="flatten_b2_5")(b2_9_pool)

    a3_1_flatten = layers.Flatten(name="flatten_a3_1")(a3_1_pool)
    a3_3_flatten = layers.Flatten(name="flatten_a3_3")(a3_3_pool)
    a3_5_flatten = layers.Flatten(name="flatten_a3_5")(a3_5_pool)
    a3_7_flatten = layers.Flatten(name="flatten_a3_7")(a3_7_pool)
    a3_9_flatten = layers.Flatten(name="flatten_a3_9")(a3_9_pool)

    b3_1_flatten = layers.Flatten(name="flatten_b3_1")(b3_1_pool)
    b3_3_flatten = layers.Flatten(name="flatten_b3_3")(b3_3_pool)
    b3_5_flatten = layers.Flatten(name="flatten_b3_5")(b3_5_pool)
    b3_7_flatten = layers.Flatten(name="flatten_b3_7")(b3_7_pool)
    b3_9_flatten = layers.Flatten(name="flatten_b3_9")(b3_9_pool)

    # Concatenation of all MaxPool outputs from all features and kernel-sizes
    cat = layers.Concatenate(name = "first_cat")([a1_1_flatten, a1_3_flatten, a1_5_flatten, 
                                                  b1_1_flatten, b1_3_flatten, b1_5_flatten,
                                                  a2_1_flatten, a2_3_flatten, a2_5_flatten,
                                                  b2_1_flatten, b2_3_flatten, b2_5_flatten, 
                                                  a3_1_flatten, a3_3_flatten, a3_5_flatten, a3_7_flatten, a3_9_flatten,
                                                  b3_1_flatten, b3_3_flatten, b3_5_flatten, b3_7_flatten, b3_9_flatten])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)
    
    # Dense layers after concatenation+dropout
    dense = layers.Dense(64, activation = dense_activation, kernel_regularizer = l2(l2_reg), name = "first_dense")(cat_dropout)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)
    
    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)
    
    # Model definition
    model = keras.Model(inputs = [a1, b1, a2, b2, a3, b3],
                        outputs = out)
    
    return model
