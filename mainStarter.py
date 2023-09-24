import pandas as pd
from collections import Counter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def load_data(calls_path, reasons_path, referrals_path):
    calls_df = pd.read_csv(calls_path)
    reasons_df = pd.read_csv(reasons_path)
    referrals_df = pd.read_csv(referrals_path)
    return calls_df, reasons_df, referrals_df


def get_most_frequent_mainreason(mainreason_list):
    counter = Counter(mainreason_list)
    most_common = counter.most_common(1)
    return most_common[0][0]  # Return the most frequent Mainreason


def process_reasons(calls_df, reasons_df):
    # Group the 'reasons' dataset by 'callreportnum1' and aggregate the 'Mainreason' entries
    reasons_grouped_df = reasons_df.groupby('callreportnum1')['Mainreason'].agg(list).reset_index()
    reasons_grouped_df['Mainreason'] = reasons_grouped_df['Mainreason'].apply(get_most_frequent_mainreason)

    # Merge with the 'calls' dataset and drop the redundant column
    merged_df = pd.merge(calls_df, reasons_grouped_df, left_on='callreportnum', right_on='callreportnum1', how='inner')
    merged_df.drop(columns=['callreportnum1'], inplace=True)
    return merged_df


def process_referrals(merged_df, referrals_df):
    # Group the 'referrals' dataset by 'Callreportnum2' and count the number of occurrences
    referrals_count_df = referrals_df.groupby('Callreportnum2').size().reset_index(name='Number_of_Referrals')

    # Merge with the 'merged_df' dataset and drop the redundant column
    merged_df = pd.merge(merged_df, referrals_count_df, left_on='callreportnum', right_on='Callreportnum2', how='left')
    merged_df.drop(columns=['Callreportnum2'], inplace=True)

    # Fill NaN values in the 'Number_of_Referrals' column with 0 and convert to integer
    merged_df['Number_of_Referrals'].fillna(0, inplace=True)
    merged_df['Number_of_Referrals'] = merged_df['Number_of_Referrals'].astype(int)
    return merged_df

def encode_mainreason(df):
    # Identify the top 7 unique values in the 'Mainreason' column
    top_mainreasons1 = df['Mainreason'].value_counts().nlargest(40).index.tolist()
    print(top_mainreasons1)
    top_mainreasons = df['Mainreason'].value_counts().nlargest(7).index.tolist()

    # Create a new column 'EncodedMainreason' and assign the value of 'Mainreason' if it is in the top 7, and 'Other' otherwise
    df['EncodedMainreason'] = df['Mainreason'].apply(lambda x: x if x in top_mainreasons else 'Other')

    # Perform one-hot encoding on the 'EncodedMainreason' column and drop the original 'EncodedMainreason' column
    df_encoded = pd.get_dummies(df, columns=['EncodedMainreason'], prefix='', prefix_sep='')

    return df_encoded


def drop_unwanted_columns(df):
    # List of columns to drop
    columns_to_drop = [
        'CallerDemographicsInterpretationOtherlanguage',
        'CallerDemographicsInterpretationLanguage',
        'CallerTypeAffected3rdParty'
    ]

    # Drop specified columns
    df_dropped = df.drop(columns=columns_to_drop, errors='ignore')  # errors='ignore' will prevent errors if a column is not present
    return df_dropped


def extract_daytime(df):
    def short_daytime(s):
        return int(s.split(" ")[1][0:2])

    def map_daytime(x):
        if 0 < x <= 6:
            return "Midnight"
        elif 6 < x <= 12:
            return "Morning"
        elif 12 < x <= 18:
            return "Afternoon"
        elif 18 < x <= 24:
            return "Evening"

    df['TimeStart'] = df['TimeStart'].apply(short_daytime).apply(map_daytime)
    return df


def bin_daytime(df):
    def bin_time(s):
        if s == "Afternoon" or s == "Morning":
            return "afternoon_morning"
        elif s == "Evening" or s == "Midnight":
            return "evening_midnight"

    df['time_bin'] = df['TimeStart'].apply(bin_time)
    return df


def process_time(df):
    df = extract_daytime(df)
    df = bin_daytime(df)
    return df

def bin_city(df, city_data):
    bins = np.array_split(city_data, 5)  # Split the city_data into 5 bins

    def binning(s):
        if s in bins[0].index:
            return "Very Short Duration"
        elif s in bins[1].index:
            return "Short Duration"
        elif s in bins[2].index:
            return "Moderate Duration"
        elif s in bins[3].index:
            return "Long Duration"
        elif s in bins[4].index:
            return "Very Long Duration"

    df['city_bins'] = df['CityName'].apply(binning)
    return df


def bin_age_group(df):
    def binning(s):
        if s in ["0-12 Child", "13-18 Youth"]:
            return "youth"
        elif s == "19-54 Adult":
            return "adult"
        elif s in ["55-64 Adult", "55-64 Older Adult", "65+ Senior"]:
            return "older adult"
        elif s in ["Prefer not to Disclose", "Unknown"]:
            return "unknown"

    df['age_bins'] = df['CallerDemographicsAgeGroup'].apply(binning)
    return df


def process_city_and_age(df):
    # Binning cities based on mean call length
    city_data = df.groupby("CityName")["CallLength"].agg(["mean", "median", "var"]).sort_values(by="mean")
    df = bin_city(df, city_data)

    # Binning age groups into broader categories
    df = bin_age_group(df)
    return df

def encode_call_type(df):
    # Define a dictionary to map the CallType to a binary value
    call_type_mapping = {
        'Assessment and Referral': 0,
        'Info Only/Ref as Req': 1
    }

    # Map the CallType column using the defined dictionary
    df['CallType'] = df['CallType'].map(call_type_mapping)

    return df
def one_hot_encode_caller_type(df):
    # Perform one-hot encoding on the 'CallerDemographicsCallerType' column and drop the original column
    df_encoded = pd.get_dummies(df, columns=['CallerDemographicsCallerType'], prefix='', prefix_sep='')
    return df_encoded

def encode_time_bin(df):
    # Define a dictionary to map the time_bin to a binary value
    time_bin_mapping = {
        'afternoon_morning': 0,
        'evening_midnight': 1
    }

    # Map the time_bin column using the defined dictionary
    df['time_bin'] = df['time_bin'].map(time_bin_mapping)

    return df
def one_hot_encode_city_age_bins(df):
    # Perform one-hot encoding on the 'city_bins' and 'age_bins' columns and drop the original columns
    df_encoded = pd.get_dummies(df, columns=['city_bins', 'age_bins'], prefix='', prefix_sep='')
    return df_encoded

def create_gender_group_column(df):
    # Define a dictionary to map the CallerDemographicsGender to the corresponding group
    gender_group_mapping = {
        'Prefer not to Disclose': 'Group A',
        'Female': 'Group A',
        'Male': 'Group A',
        'Unknown': 'Group A',
        'Transgender': 'Group B',
        'Trans Male': 'Group B',
        'Trans Female': 'Group B',
        'Non-binary': 'Group D',
        'Gender Fluid': 'Group D',
        '2 Spirited': 'Group C',
        '2 Spirit': 'Group C'
    }

    # Create a new column 'GenderGroup' by mapping the 'CallerDemographicsGender' column using the defined dictionary
    df['GenderGroup'] = df['CallerDemographicsGender'].map(gender_group_mapping)

    return df

def one_hot_encode_gender_group(df):
    # Perform one-hot encoding on the 'GenderGroup' column and drop the original column
    df_encoded = pd.get_dummies(df, columns=['GenderGroup'], prefix='', prefix_sep='')
    return df_encoded

def extract_year_and_season(df):
    # Convert 'DateStart' to datetime format if it is not already
    df['DateStart'] = pd.to_datetime(df['DateStart'])

    # Extract the year from 'DateStart' and create a new column 'Year'
    df['Year'] = (df['DateStart'].dt.year >= 2021).astype(int)

    # Define a dictionary to map the month to the corresponding season
    month_to_season = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }

    # Map the month of 'DateStart' to the corresponding season and create a new column 'Season'
    df['Season'] = df['DateStart'].dt.month.map(month_to_season)

    return df

def one_hot_encode_season(df):
    # Perform one-hot encoding on the 'GenderGroup' column and drop the original column
    df_encoded = pd.get_dummies(df, columns=['Season'], prefix='', prefix_sep='')
    return df_encoded

def one_hot_encode_year(df):
    # Perform one-hot encoding on the 'Year' column and drop the original column
    df_encoded = pd.get_dummies(df, columns=['Year'], prefix='Year', prefix_sep='_')
    return df_encoded

def main():
    calls_path = 'datasets/calls.csv'
    reasons_path = 'datasets/reasons.csv'
    referrals_path = 'datasets/referrals.csv'

    calls_df, reasons_df, referrals_df = load_data(calls_path, reasons_path, referrals_path)
    merged_df = process_reasons(calls_df, reasons_df)
    final_df = process_referrals(merged_df, referrals_df)
    final_df_encoded = encode_mainreason(final_df)
    final_df_cleaned = drop_unwanted_columns(final_df_encoded)
    final_df_time_processed = process_time(final_df_cleaned)
    final_df_city_age_processed = process_city_and_age(final_df_time_processed)
    final_df_call_type_encoded = encode_call_type(final_df_city_age_processed)
    final_df_caller_type_encoded = one_hot_encode_caller_type(final_df_call_type_encoded)

    final_df_time_bin_encoded = encode_time_bin(final_df_caller_type_encoded)

    final_df_city_age_encoded = one_hot_encode_city_age_bins(final_df_time_bin_encoded)

    final_df_with_gender_group = create_gender_group_column(final_df_city_age_encoded)

    final_df_gender_group_encoded = one_hot_encode_gender_group(final_df_with_gender_group)

    final_df_with_year_season = extract_year_and_season(final_df_gender_group_encoded)

    final_df_year_encoded = one_hot_encode_year(final_df_with_year_season)

    final_df_with_year_season = one_hot_encode_season(final_df_year_encoded)
    # Save the final cleaned DataFrame to a CSV file
    final_df_with_year_season.to_csv('datasets/mergedCallsReducedCleaned.csv', index=False)
    df = final_df_with_year_season
    # print(df["Year"].value_counts())






    # Save the final cleaned DataFrame to a CSV file
    # final_df_call_type_encoded.to_csv('datasets/mergedCallsReducedCleaned.csv', index=False)
    # print(final_df_call_type_encoded.groupby('CallerDemographicsCallerType')['CallLength'].mean())


    # Save the final cleaned DataFrame to a CSV file
    # final_df_city_age_processed.to_csv('datasets/mergedCallsReducedCleaned.csv', index=False)

    # return final_df_city_age_processed
    # Display the first few rows of the final cleaned DataFrame
    # print(final_df_time_processed.head())


if __name__ == "__main__":
    # final_df = main()
    # print(final_df['CallType'].value_counts())
    main()





