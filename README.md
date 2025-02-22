# TCR-Peptide Binding Prediction Project

This repository contains scripts for processing and analyzing TCR (T-cell receptor) and peptide binding data using various embedding models.

## Main Scripts

### Embedding Generation Scripts
- `/data_process`: Generates embeddings for TCR sequences
  - Supports ESM2 model for embedding generation
  - Processes CDR1, CDR2, and CDR3 regions or only CDR3
  - Saves embeddings in pickle format

### Training and Predicting Scripts
- `/train_pred`: Uses embeddings for TCR sequences as the input of the CNN model. 
  - Generates prediction scores between 0 and 1.

### Data Analysis Scripts
- `/performance`: Calculates AUC metrics for prediction results
  - Computes AUC 0.1 scores for individual peptides
  - Calculates weighted average AUC scores
  - Generates detailed metrics reports

## Dependencies
- numpy
- pandas
- scikit-learn
- ESM embeddings library
- Bio-embeddings package

## Data Format
Input data should include:
- TCR alpha and beta chain sequences
- CDR region start and end positions
- Binding information (binder column)
