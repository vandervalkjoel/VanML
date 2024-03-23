import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged CSV file
# file_path = 'datasets/mergedCalls.csv'
df = pd.read_csv(file_path)

# Check the number of unique values in 'Mainreason'
num_unique_mainreason = df['Mainreason'].nunique()

# Group by 'Mainreason' and calculate the mean 'CallLength' for each group
mainreason_calllength_df = df.groupby('Mainreason')['CallLength'].mean().reset_index().sort_values(by='CallLength', ascending=False)

# Plot the results
plt.figure(figsize=(10, 6))
sns.barplot(x='CallLength', y='Mainreason', data=mainreason_calllength_df)
plt.title('Mean CallLength for each Mainreason')
plt.xlabel('Mean CallLength')
plt.ylabel('Mainreason')
plt.tight_layout()
plt.savefig('Mainreason_CallLength.png')  # Save the plot as a PNG file
plt.show()

# Display the number of unique 'Mainreason' values and the first few rows of the grouped DataFrame
print(f"Number of unique Mainreason values: {num_unique_mainreason}")
print(mainreason_calllength_df.head())
