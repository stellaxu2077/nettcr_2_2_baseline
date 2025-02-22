#!/usr/bin/env python

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import numpy as np

def CNN_CDR123_global_max(dropout_rate,
                          seed,
                          peptide_shape,
                          a1_shape,
                          a2_shape,
                          a3_shape,
                          b1_shape,
                          b2_shape,
                          b3_shape,
                          convolution_filters_count,
                          hidden_units_count,
                          mixed_precision,
                          pep_conv_activation,
                          cdr_conv_activation):

    # Activation
    dense_activation = 'sigmoid'

    # Inputs
    pep = keras.Input(shape = peptide_shape, name ='pep')
    a1  = keras.Input(shape = a1_shape, name ='a1')
    a2  = keras.Input(shape = a2_shape, name ='a2')
    a3  = keras.Input(shape = a3_shape, name ='a3')
    b1  = keras.Input(shape = b1_shape, name ='b1')
    b2  = keras.Input(shape = b2_shape, name ='b2')
    b3  = keras.Input(shape = b3_shape, name ='b3')

    # Convolutional Layers
    pep_1_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 1, padding = 'same', activation = pep_conv_activation)(pep)
    pep_3_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 3, padding = 'same', activation = pep_conv_activation)(pep)
    pep_5_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 5, padding = 'same', activation = pep_conv_activation)(pep)
    pep_7_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 7, padding = 'same', activation = pep_conv_activation)(pep)
    pep_9_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 9, padding = 'same', activation = pep_conv_activation)(pep)

    a1_1_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 1, padding = 'same', activation = cdr_conv_activation)(a1)
    a1_3_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 3, padding = 'same', activation = cdr_conv_activation)(a1)
    a1_5_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 5, padding = 'same', activation = cdr_conv_activation)(a1)
    a1_7_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 7, padding = 'same', activation = cdr_conv_activation)(a1)
    a1_9_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 9, padding = 'same', activation = cdr_conv_activation)(a1)

    a2_1_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 1, padding = 'same', activation = cdr_conv_activation)(a2)
    a2_3_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 3, padding = 'same', activation = cdr_conv_activation)(a2)
    a2_5_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 5, padding = 'same', activation = cdr_conv_activation)(a2)
    a2_7_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 7, padding = 'same', activation = cdr_conv_activation)(a2)
    a2_9_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 9, padding = 'same', activation = cdr_conv_activation)(a2)

    a3_1_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 1, padding = 'same', activation = cdr_conv_activation)(a3)
    a3_3_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 3, padding = 'same', activation = cdr_conv_activation)(a3)
    a3_5_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 5, padding = 'same', activation = cdr_conv_activation)(a3)
    a3_7_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 7, padding = 'same', activation = cdr_conv_activation)(a3)
    a3_9_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 9, padding = 'same', activation = cdr_conv_activation)(a3)

    b1_1_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 1, padding = 'same', activation = cdr_conv_activation)(b1)
    b1_3_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 3, padding = 'same', activation = cdr_conv_activation)(b1)
    b1_5_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 5, padding = 'same', activation = cdr_conv_activation)(b1)
    b1_7_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 7, padding = 'same', activation = cdr_conv_activation)(b1)
    b1_9_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 9, padding = 'same', activation = cdr_conv_activation)(b1)

    b2_1_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 1, padding = 'same', activation = cdr_conv_activation)(b2)
    b2_3_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 3, padding = 'same', activation = cdr_conv_activation)(b2)
    b2_5_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 5, padding = 'same', activation = cdr_conv_activation)(b2)
    b2_7_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 7, padding = 'same', activation = cdr_conv_activation)(b2)
    b2_9_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 9, padding = 'same', activation = cdr_conv_activation)(b2)

    b3_1_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 1, padding = 'same', activation = cdr_conv_activation)(b3)
    b3_3_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 3, padding = 'same', activation = cdr_conv_activation)(b3)
    b3_5_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 5, padding = 'same', activation = cdr_conv_activation)(b3)
    b3_7_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 7, padding = 'same', activation = cdr_conv_activation)(b3)
    b3_9_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 9, padding = 'same', activation = cdr_conv_activation)(b3)

    # Max Pooling
    pep_1_pool = layers.GlobalMaxPooling1D()(pep_1_CNN)
    pep_3_pool = layers.GlobalMaxPooling1D()(pep_3_CNN)
    pep_5_pool = layers.GlobalMaxPooling1D()(pep_5_CNN)
    pep_7_pool = layers.GlobalMaxPooling1D()(pep_7_CNN)
    pep_9_pool = layers.GlobalMaxPooling1D()(pep_9_CNN)

    a1_1_pool = layers.GlobalMaxPooling1D()(a1_1_CNN)
    a1_3_pool = layers.GlobalMaxPooling1D()(a1_3_CNN)
    a1_5_pool = layers.GlobalMaxPooling1D()(a1_5_CNN)
    a1_7_pool = layers.GlobalMaxPooling1D()(a1_7_CNN)
    a1_9_pool = layers.GlobalMaxPooling1D()(a1_9_CNN)

    a2_1_pool = layers.GlobalMaxPooling1D()(a2_1_CNN)
    a2_3_pool = layers.GlobalMaxPooling1D()(a2_3_CNN)
    a2_5_pool = layers.GlobalMaxPooling1D()(a2_5_CNN)
    a2_7_pool = layers.GlobalMaxPooling1D()(a2_7_CNN)
    a2_9_pool = layers.GlobalMaxPooling1D()(a2_9_CNN)

    a3_1_pool = layers.GlobalMaxPooling1D()(a3_1_CNN)
    a3_3_pool = layers.GlobalMaxPooling1D()(a3_3_CNN)
    a3_5_pool = layers.GlobalMaxPooling1D()(a3_5_CNN)
    a3_7_pool = layers.GlobalMaxPooling1D()(a3_7_CNN)
    a3_9_pool = layers.GlobalMaxPooling1D()(a3_9_CNN)

    b1_1_pool = layers.GlobalMaxPooling1D()(b1_1_CNN)
    b1_3_pool = layers.GlobalMaxPooling1D()(b1_3_CNN)
    b1_5_pool = layers.GlobalMaxPooling1D()(b1_5_CNN)
    b1_7_pool = layers.GlobalMaxPooling1D()(b1_7_CNN)
    b1_9_pool = layers.GlobalMaxPooling1D()(b1_9_CNN)

    b2_1_pool = layers.GlobalMaxPooling1D()(b2_1_CNN)
    b2_3_pool = layers.GlobalMaxPooling1D()(b2_3_CNN)
    b2_5_pool = layers.GlobalMaxPooling1D()(b2_5_CNN)
    b2_7_pool = layers.GlobalMaxPooling1D()(b2_7_CNN)
    b2_9_pool = layers.GlobalMaxPooling1D()(b2_9_CNN)

    b3_1_pool = layers.GlobalMaxPooling1D()(b3_1_CNN)
    b3_3_pool = layers.GlobalMaxPooling1D()(b3_3_CNN)
    b3_5_pool = layers.GlobalMaxPooling1D()(b3_5_CNN)
    b3_7_pool = layers.GlobalMaxPooling1D()(b3_7_CNN)
    b3_9_pool = layers.GlobalMaxPooling1D()(b3_9_CNN)

    # Concatenate all max pooling layers to a single layer
    cat = layers.Concatenate()([pep_1_pool, pep_3_pool, pep_5_pool, pep_7_pool, pep_9_pool,
                                a1_1_pool, a1_3_pool, a1_5_pool, a1_7_pool, a1_9_pool,
                                a2_1_pool, a2_3_pool, a2_5_pool, a2_7_pool, a2_9_pool,
                                a3_1_pool, a3_3_pool, a3_5_pool, a3_7_pool, a3_9_pool,
                                b1_1_pool, b1_3_pool, b1_5_pool, b1_7_pool, b1_9_pool,
                                b2_1_pool, b2_3_pool, b2_5_pool, b2_7_pool, b2_9_pool,
                                b3_1_pool, b3_3_pool, b3_5_pool, b3_7_pool, b3_9_pool])

    # Dropout - Required to prevent overfitting
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    # Dense layer
    dense = layers.Dense(units = hidden_units_count, activation = dense_activation)(cat_dropout)

    # Output layer
    out = layers.Dense(units = 1)(dense)

    if mixed_precision:
        out = layers.Activation(activation = "sigmoid", dtype='float32')(out)
    else:
        out = layers.Activation(activation = "sigmoid")(out)

    # Prepare model object
    model = keras.Model(inputs = [pep, a1, a2, a3, b1, b2, b3],
                        outputs = out)

    return model



def CNN_CDR3_global_max(dropout_rate,
                          seed,
                          peptide_shape,
                          a3_shape,
                          b3_shape,
                          convolution_filters_count,
                          hidden_units_count,
                          mixed_precision,
                          pep_conv_activation,
                          cdr_conv_activation):

    # Activation
    dense_activation = 'sigmoid'

    # Inputs
    pep = keras.Input(shape = peptide_shape, name ='pep')
    a3  = keras.Input(shape = a3_shape, name ='a3')
    b3  = keras.Input(shape = b3_shape, name ='b3')

    # Convolutional Layers
    pep_1_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 1, padding = 'same', activation = pep_conv_activation)(pep)
    pep_3_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 3, padding = 'same', activation = pep_conv_activation)(pep)
    pep_5_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 5, padding = 'same', activation = pep_conv_activation)(pep)
    pep_7_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 7, padding = 'same', activation = pep_conv_activation)(pep)
    pep_9_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 9, padding = 'same', activation = pep_conv_activation)(pep)

    a3_1_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 1, padding = 'same', activation = cdr_conv_activation)(a3)
    a3_3_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 3, padding = 'same', activation = cdr_conv_activation)(a3)
    a3_5_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 5, padding = 'same', activation = cdr_conv_activation)(a3)
    a3_7_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 7, padding = 'same', activation = cdr_conv_activation)(a3)
    a3_9_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 9, padding = 'same', activation = cdr_conv_activation)(a3)

    b3_1_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 1, padding = 'same', activation = cdr_conv_activation)(b3)
    b3_3_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 3, padding = 'same', activation = cdr_conv_activation)(b3)
    b3_5_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 5, padding = 'same', activation = cdr_conv_activation)(b3)
    b3_7_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 7, padding = 'same', activation = cdr_conv_activation)(b3)
    b3_9_CNN = layers.Conv1D(filters = convolution_filters_count, kernel_size = 9, padding = 'same', activation = cdr_conv_activation)(b3)

    # Max Pooling
    pep_1_pool = layers.GlobalMaxPooling1D()(pep_1_CNN)
    pep_3_pool = layers.GlobalMaxPooling1D()(pep_3_CNN)
    pep_5_pool = layers.GlobalMaxPooling1D()(pep_5_CNN)
    pep_7_pool = layers.GlobalMaxPooling1D()(pep_7_CNN)
    pep_9_pool = layers.GlobalMaxPooling1D()(pep_9_CNN)

    a3_1_pool = layers.GlobalMaxPooling1D()(a3_1_CNN)
    a3_3_pool = layers.GlobalMaxPooling1D()(a3_3_CNN)
    a3_5_pool = layers.GlobalMaxPooling1D()(a3_5_CNN)
    a3_7_pool = layers.GlobalMaxPooling1D()(a3_7_CNN)
    a3_9_pool = layers.GlobalMaxPooling1D()(a3_9_CNN)

    b3_1_pool = layers.GlobalMaxPooling1D()(b3_1_CNN)
    b3_3_pool = layers.GlobalMaxPooling1D()(b3_3_CNN)
    b3_5_pool = layers.GlobalMaxPooling1D()(b3_5_CNN)
    b3_7_pool = layers.GlobalMaxPooling1D()(b3_7_CNN)
    b3_9_pool = layers.GlobalMaxPooling1D()(b3_9_CNN)

    # Concatenate all max pooling layers to a single layer
    cat = layers.Concatenate()([pep_1_pool, pep_3_pool, pep_5_pool, pep_7_pool, pep_9_pool,
                                a3_1_pool, a3_3_pool, a3_5_pool, a3_7_pool, a3_9_pool,
                                b3_1_pool, b3_3_pool, b3_5_pool, b3_7_pool, b3_9_pool])

    # Dropout - Required to prevent overfitting
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    # Dense layer
    dense = layers.Dense(units = hidden_units_count, activation = dense_activation)(cat_dropout)

    # Output layer
    out = layers.Dense(units = 1)(dense)

    if mixed_precision:
        out = layers.Activation(activation = "sigmoid", dtype='float32')(out)
    else:
        out = layers.Activation(activation = "sigmoid")(out)

    # Prepare model object
    model = keras.Model(inputs = [pep, a3, b3],
                        outputs = out)

    return model






def ff_CDR123(dropout_rate,
              seed,
              peptide_shape,
              a3_shape,
              b3_shape,
              hidden_units_count,
              mixed_precision):

    # Activation
    dense_activation = 'sigmoid'

    # Inputs
    pep = keras.Input(shape = peptide_shape, name = 'pep')
    a3 = keras.Input(shape = a3_shape, name = 'a3')
    b3 = keras.Input(shape = b3_shape, name = 'b3')

    # Concatenate all input layers to a single layer
    cat = layers.Concatenate()([pep, a3, b3])

    # Dropout - Required to prevent overfitting
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    # Dense layer
    dense = layers.Dense(units = hidden_units_count, activation = dense_activation)(cat_dropout)

    # Output layer
    out = layers.Dense(units = 1)(dense)

    if mixed_precision:
        out = layers.Activation(activation = "sigmoid", dtype='float32')(out)
    else:
        out = layers.Activation(activation = "sigmoid")(out)

    # Prepare model object
    model = keras.Model(inputs = [pep, a3, b3],
                        outputs = out)

    return model



def ff_CDR3(dropout_rate,
              seed,
              peptide_shape,
              a3_shape,
              b3_shape,
              hidden_units_count,
              mixed_precision):

    # Activation
    dense_activation = 'sigmoid'

    # Inputs
    pep = keras.Input(shape = peptide_shape, name = 'pep')
    a3 = keras.Input(shape = a3_shape, name = 'a3')
    b3 = keras.Input(shape = b3_shape, name = 'b3')

    # Concatenate all input layers to a single layer
    cat = layers.Concatenate()([pep, a3, b3])

    # Dropout - Required to prevent overfitting
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    # Dense layer
    dense = layers.Dense(units = hidden_units_count, activation = dense_activation)(cat_dropout)

    # Output layer
    out = layers.Dense(units = 1)(dense)

    if mixed_precision:
        out = layers.Activation(activation = "sigmoid", dtype='float32')(out)
    else:
        out = layers.Activation(activation = "sigmoid")(out)

    # Prepare model object
    model = keras.Model(inputs = [pep, a3, b3],
                        outputs = out)

    return model

