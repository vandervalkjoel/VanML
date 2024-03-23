import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.utils import shuffle

# File paths
callsCSV = 'datasets/calls.csv'
reasonsCSV = 'datasets/reasons.csv'
referralsCSV = 'datasets/referrals.csv'
referralCounts = 'datasets/referral_counts.csv'

# Load the dataframes from the CSV files
calls_df = pd.read_csv(callsCSV)
# reasons_df = pd.read_csv(reasonsCSV)
referrals_df = pd.read_csv(referralsCSV)
referral_counts_df = pd.read_csv(referralCounts)

# Merge datasets
referral_counts_df.rename(columns={'Callreportnum2': 'callreportnum'}, inplace=True)
merged_df = pd.merge(calls_df, referral_counts_df, on='callreportnum', how='left')
merged_df['num_referrals'].fillna(0, inplace=True)
print(calls_df.shape)
# print(reasons_df.shape)
print(referral_counts_df.shape)
print(merged_df.shape)
print(merged_df.columns.tolist())
# print(merged_df.head())
# merged_df_final = pd.merge(merged_df, referrals_df, on='CallReportNum', how='left')
# Take merged_df and the num_referrals column and replace all NaN values with 0
merged_df['num_referrals'].fillna(0, inplace=True)
print(merged_df.isna().sum())

# Calculate the correlation between num_referrals and CallLength
correlation = merged_df['num_referrals'].corr(merged_df['CallLength'])
#
print(f"The correlation between num_referrals and CallLength is {correlation:.2f}")


referrals_df.rename(columns={'Callreportnum2': 'callreportnum'}, inplace=True)
merged_df = pd.merge(calls_df, referrals_df, on='callreportnum', how='left')
print(merged_df.shape)

