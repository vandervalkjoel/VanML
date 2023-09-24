import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged CSV file
# file_path = 'datasets/mergedCalls.csv'
df = pd.read_csv(file_path)

# Group by 'Mainreason' and calculate the mean 'CallLength' and count for each group
mainreason_grouped_df = df.groupby('Mainreason').agg({'CallLength': 'mean', 'Mainreason': 'count'}).rename(columns={'CallLength': 'Mean_CallLength', 'Mainreason': 'Count'}).reset_index()

# Filter the top ten Mainreason categories based on counts
top_ten_mainreason_df = mainreason_grouped_df.nlargest(10, 'Count')

# Sort the filtered DataFrame by 'Mean_CallLength'
sorted_top_ten_mainreason_df = top_ten_mainreason_df.sort_values(by='Mean_CallLength', ascending=False)

# Display the sorted top ten DataFrame
print(sorted_top_ten_mainreason_df)

print(df.columns.tolist())
# callreportnum1 give me the number of unique values for
# callreportnum1
print(df['callreportnum1'].nunique())
print(df.shape)
