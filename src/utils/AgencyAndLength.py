import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
calls_df = pd.read_csv('datasets/calls.csv')

# Filter out rows where CallLength is greater than 50
filtered_calls_df = calls_df[calls_df['CallLength'] <= 50]

# Plotting
plt.figure(figsize=(12, 6))

# Plot histogram of the original CallLength values
plt.subplot(1, 2, 1)
sns.histplot(filtered_calls_df['CallLength'], bins=30, kde=True)
plt.title('Histogram of CallLength')
plt.xlabel('CallLength')
plt.ylabel('Frequency')

# Filter out non-positive CallLength values before log transformation
positive_calls_df = filtered_calls_df[filtered_calls_df['CallLength'] > 0]

# Plot histogram of the log-transformed CallLength values
plt.subplot(1, 2, 2)
sns.histplot(np.log(positive_calls_df['CallLength']), bins=30, kde=True)
plt.title('Histogram of log(CallLength)')
plt.xlabel('log(CallLength)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


