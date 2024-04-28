import os
import pandas as pd
#this is for merging the test dataset.
# Get the current working directory where the script is located
current_directory = os.path.dirname(os.path.realpath(__file__))

# Build the paths to the CSV files
path_to_test_csv = os.path.join(current_directory,'Toxic', 'test.csv')
path_to_submission_csv = os.path.join(current_directory, 'Toxic', 'submission.csv')

# Read the data into pandas dataframes
test_df = pd.read_csv(path_to_test_csv)
submission_df = pd.read_csv(path_to_submission_csv)

# combine the data on the 'id' column
merged_df = pd.merge(test_df, submission_df, on='id')

# Save the merged dataframe to a new CSV file in the current directory
merged_csv_path = os.path.join(current_directory, 'Toxic','merged_test_submission.csv')
merged_df.to_csv(merged_csv_path, index=False)

print(f"The merged CSV file has been saved to: {merged_csv_path}")