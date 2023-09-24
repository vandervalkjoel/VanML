# Group by 'CallReportNum' and calculate the count of each group,
import pandas as pd
# then reset the index to make 'CallReportNum' a column again
referralsCSV = 'datasets/referrals.csv'
callsCSV = 'datasets/calls.csv'
calls_df = pd.read_csv(callsCSV)
referrals_df = pd.read_csv(referralsCSV)
referral_counts_df = referrals_df.groupby('Callreportnum2').size().reset_index(name='num_referrals')
#
referral_counts_df.to_csv('datasets/referral_counts.csv', index=False)
# print(referral_counts_df.info())
# print(referral_counts_df.head())
# Give the max average and median number of referrals
# print(referral_counts_df['num_referrals'].max())
# print(referral_counts_df['num_referrals'].mean())
# print(referral_counts_df['num_referrals'].median())

# Define the specific CallReportNum you are interested in
specific_callreportnum = 12345  # Replace with the actual CallReportNum you are looking for

# Filter the DataFrame to find the row where Callreportnum2 is equal to the specific_callreportnum
specific_row = referral_counts_df[referral_counts_df['Callreportnum2'] == 92723061]
print(specific_row)
