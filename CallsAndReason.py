import pandas as pd
from collections import Counter

# Define the path to your datasets
calls_file_path = 'datasets/calls.csv'
reasons_file_path = 'datasets/reasons.csv'

# Load the datasets
calls_df = pd.read_csv(calls_file_path)
reasons_df = pd.read_csv(reasons_file_path)

# Define a function to get the most frequent Mainreason
def get_most_frequent_mainreason(mainreason_list):
    counter = Counter(mainreason_list)
    most_common = counter.most_common(1)
    return most_common[0][0]  # Return the most frequent Mainreason

# Group the 'reasons' dataset by 'callreportnum1' and aggregate the 'Mainreason' entries
reasons_grouped_df = reasons_df.groupby('callreportnum1')['Mainreason'].agg(list).reset_index()

# Apply the function to get the most frequent Mainreason for each 'callreportnum1'
reasons_grouped_df['Mainreason'] = reasons_grouped_df['Mainreason'].apply(get_most_frequent_mainreason)

# Merge the 'calls' dataset with the aggregated 'reasons' dataset
merged_df = pd.merge(calls_df, reasons_grouped_df, left_on='callreportnum', right_on='callreportnum1', how='inner')

# Drop the 'callreportnum1' column as it is redundant after merging
merged_df.drop(columns=['callreportnum1'], inplace=True)

# Display the first few rows of the new merged DataFrame
# Write the dataset to a CSV file if needed

referrals_df = pd.read_csv('datasets/referrals.csv')


# Group the 'referrals' dataset by 'Callreportnum' and count the number of occurrences of each unique 'Callreportnum'
referrals_count_df = referrals_df.groupby('Callreportnum2').size().reset_index(name='Number_of_Referrals')

# Merge the 'mergedCallsReduced' dataset with the 'referrals_count_df' dataset
merged_df = pd.merge(merged_df, referrals_count_df, left_on='callreportnum', right_on='Callreportnum2', how='left')

merged_df.drop(columns=['Callreportnum2'], inplace=True)

# Fill NaN values in the 'Number_of_Referrals' column with 0
merged_df['Number_of_Referrals'].fillna(0, inplace=True)
merged_df['Number_of_Referrals'] = merged_df['Number_of_Referrals'].astype(int)  # Convert to integer

# Display the first few rows of the new merged DataFrame
print(merged_df.head())

# Save the new merged DataFrame to a CSV file if needed
merged_df.to_csv('datasets/mergedCallsReduced.csv', index=False)