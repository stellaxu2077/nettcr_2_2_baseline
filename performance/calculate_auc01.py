import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import os

# Set up command line argument parser
parser = argparse.ArgumentParser(description="Process and save metrics.")
parser.add_argument("-d", "--data", required=True, help="Path to the input CSV file.")
args = parser.parse_args()

# Get input file path and name
input_file = args.data
if not os.path.exists(input_file):
    raise FileNotFoundError(f"File '{input_file}' not found.")

# Read all prediction results
predictions = pd.read_csv(input_file, sep='\t')

# Calculate total AUC 0.1
fpr, tpr, thresholds = roc_curve(predictions['binder'], predictions['prediction'])
total_auc_01 = roc_auc_score(predictions['binder'], predictions['prediction'], max_fpr=0.1)
#print(f"Total AUC 0.1: {total_auc_01}")

total_positive_count = predictions['binder'].sum()
#print(f"Total Positive Peptides: {total_positive_count}")

peptide_auc_01 = {}
peptide_positive_counts = {}

# Calculate AUC 0.1 for each peptide
unique_peptides = predictions['peptide'].unique()

for peptide in unique_peptides:
    peptide_data = predictions[predictions['peptide'] == peptide]

    positive_count = (peptide_data['binder'] == 1).sum()
    peptide_positive_counts[peptide] = positive_count

    if len(peptide_data['binder'].unique()) > 1:  # Make sure there are both positive and negative samples
        fpr, tpr, thresholds = roc_curve(peptide_data['binder'], peptide_data['prediction'])
        auc_01 = roc_auc_score(peptide_data['binder'], peptide_data['prediction'], max_fpr=0.1)
        peptide_auc_01[peptide] = auc_01
    else:
        peptide_auc_01[peptide] = None  # Only one class, cannot calculate AUC

# Combine results into a DataFrame
results_df = pd.DataFrame({
    'Peptide': unique_peptides,
    'Positive_Count': [peptide_positive_counts[peptide] for peptide in unique_peptides],
    'AUC_0.1': [peptide_auc_01[peptide] for peptide in unique_peptides]
})

total_row = pd.DataFrame({
    'Peptide': ['Total'],
    'Positive_Count': [total_positive_count],
    'AUC_0.1': [total_auc_01]
})

valid_peptides = results_df.dropna(subset=['AUC_0.1'])
weighted_sum = (valid_peptides['AUC_0.1'] * valid_peptides['Positive_Count']).sum()
total_weight = valid_peptides['Positive_Count'].sum()
weighted_average_auc_01 = weighted_sum / total_weight if total_weight != 0 else None

unweighted_average_auc_01 = valid_peptides['AUC_0.1'].mean() if not valid_peptides.empty else None

weighted_average_row = pd.DataFrame({
    'Peptide': ['Weighted Average'],
    'Positive_Count': [None],
    'AUC_0.1': [weighted_average_auc_01]
})

unweighted_average_row = pd.DataFrame({
    'Peptide': ['Unweighted Average'],
    'Positive_Count': [None],
    'AUC_0.1': [unweighted_average_auc_01]
})

# Combine total row with peptide-specific metrics
#results_df = pd.concat([total_row, results_df], ignore_index=True)
results_df = pd.concat([total_row, results_df, weighted_average_row, unweighted_average_row], ignore_index=True)

# Sort by positive counts
sorted_results_df = results_df.sort_values(by='Positive_Count', ascending=False)
sorted_results_df.reset_index(drop=True, inplace=True)
print(sorted_results_df)

# Save results for further analysis or plotting
base_name = os.path.basename(input_file).replace("_predictions.tsv", "")
output_file = f"{base_name}_auc01.tsv"

sorted_results_df.to_csv(output_file, sep='\t', index=False)
print(f"Metrics saved to '{output_file}'.")
#sorted_results_df.to_csv('peptide_metrics.csv', index=False)
#print("Metrics saved to 'peptide_metrics.csv'.")


