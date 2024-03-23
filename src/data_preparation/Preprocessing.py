import pandas as pd
from collections import Counter
import numpy as np


class PreprocessingPipeline:
    def __init__(self, calls_path, reasons_path, referrals_path):
        self.calls_df, self.reasons_df, self.referrals_df = self.load_data(calls_path, reasons_path, referrals_path)

    @staticmethod
    def load_data(calls_path, reasons_path, referrals_path):
        calls_df = pd.read_csv(calls_path)
        reasons_df = pd.read_csv(reasons_path)
        referrals_df = pd.read_csv(referrals_path)
        return calls_df, reasons_df, referrals_df

    def run(self):
        self.process_reasons()
        self.process_referrals()
        self.encode_mainreason()
        self.drop_unwanted_columns()
        self.process_time()
        self.process_city_and_age()
        self.encode_call_type()
        self.one_hot_encode_caller_type()
        self.encode_time_bin()
        self.one_hot_encode_city_age_bins()
        self.create_gender_group_column()
        self.one_hot_encode_gender_group()
        self.extract_year_and_season()
        self.one_hot_encode_season()
        self.filter_columns()
        self.convert_boolean_to_binary()

    def convert_boolean_to_binary(self):
        for col in self.calls_df.columns:
            if self.calls_df[col].dtype == 'bool':
                self.calls_df[col] = self.calls_df[col].astype(int)
    def get_most_frequent_mainreason(self, mainreason_list):
        counter = Counter(mainreason_list)
        most_common = counter.most_common(1)
        return most_common[0][0]  # Return the most frequent Mainreason

    def process_reasons(self):
        # Group the 'reasons' dataset by 'callreportnum1' and count the number of occurrences
        reasons_count_df = self.reasons_df.groupby('callreportnum1').size().reset_index(name='reasons_count')

        # Group the 'reasons' dataset by 'callreportnum1' and aggregate the 'Mainreason' entries
        reasons_grouped_df = self.reasons_df.groupby('callreportnum1')['Mainreason'].agg(list).reset_index()
        reasons_grouped_df['Mainreason'] = reasons_grouped_df['Mainreason'].apply(self.get_most_frequent_mainreason)

        # Merge the count DataFrame with the grouped DataFrame on 'callreportnum1'
        reasons_grouped_df = pd.merge(reasons_grouped_df, reasons_count_df, on='callreportnum1', how='left')

        # Merge with the 'calls' dataset and drop the redundant column
        self.calls_df = pd.merge(self.calls_df, reasons_grouped_df, left_on='callreportnum', right_on='callreportnum1', how='inner')
        self.calls_df.drop(columns=['callreportnum1'], inplace=True)


    def process_referrals(self):
        referrals_count_df = self.referrals_df.groupby('Callreportnum2').size().reset_index(name='Number_of_Referrals')
        self.calls_df = pd.merge(self.calls_df, referrals_count_df, left_on='callreportnum', right_on='Callreportnum2', how='left')
        self.calls_df.drop(columns=['Callreportnum2'], inplace=True)
        self.calls_df['Number_of_Referrals'].fillna(0, inplace=True)
        self.calls_df['Number_of_Referrals'] = self.calls_df['Number_of_Referrals'].astype(int)

    def encode_mainreason(self):
        top_mainreasons = self.calls_df['Mainreason'].value_counts().nlargest(7).index.tolist()
        self.calls_df['EncodedMainreason'] = self.calls_df['Mainreason'].apply(lambda x: x if x in top_mainreasons else 'Other')
        self.calls_df = pd.get_dummies(self.calls_df, columns=['EncodedMainreason'], prefix='', prefix_sep='')

    def drop_unwanted_columns(self):
        columns_to_drop = [
            'CallerDemographicsInterpretationOtherlanguage',
            'CallerDemographicsInterpretationLanguage',
            'CallerTypeAffected3rdParty'
        ]
        self.calls_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    def extract_daytime(self):
        self.calls_df['TimeStart'] = self.calls_df['TimeStart'].apply(lambda s: int(s.split(" ")[1][0:2])).apply(lambda x: "Midnight" if 0 < x <= 6 else "Morning" if 6 < x <= 12 else "Afternoon" if 12 < x <= 18 else "Evening")

    def bin_daytime(self):
        self.calls_df['time_bin'] = self.calls_df['TimeStart'].apply(lambda s: "afternoon_morning" if s in ["Afternoon", "Morning"] else "evening_midnight")

    def process_time(self):
        self.extract_daytime()
        self.bin_daytime()

    def bin_city(self, city_data):
        bins = np.array_split(city_data, 5)
        self.calls_df['city_bins'] = self.calls_df['CityName'].apply(lambda s: "Very Short Duration" if s in bins[0].index else "Short Duration" if s in bins[1].index else "Moderate Duration" if s in bins[2].index else "Long Duration" if s in bins[3].index else "Very Long Duration")

    def bin_age_group(self):
        self.calls_df['age_bins'] = self.calls_df['CallerDemographicsAgeGroup'].apply(lambda s: "youth" if s in ["0-12 Child", "13-18 Youth"] else "adult" if s == "19-54 Adult" else "older adult" if s in ["55-64 Adult", "55-64 Older Adult", "65+ Senior"] else "unknown")

    def process_city_and_age(self):
        city_data = self.calls_df.groupby("CityName")["CallLength"].agg(["mean", "median", "var"]).sort_values(by="mean")
        self.bin_city(city_data)
        self.bin_age_group()

    def encode_call_type(self):
        self.calls_df['CallType'] = self.calls_df['CallType'].map({'Assessment and Referral': 0, 'Info Only/Ref as Req': 1})

    def one_hot_encode_caller_type(self):
        self.calls_df = pd.get_dummies(self.calls_df, columns=['CallerDemographicsCallerType'], prefix='', prefix_sep='')

    def encode_time_bin(self):
        self.calls_df['time_bin'] = self.calls_df['time_bin'].map({'afternoon_morning': 0, 'evening_midnight': 1})

    def one_hot_encode_city_age_bins(self):
        self.calls_df = pd.get_dummies(self.calls_df, columns=['city_bins', 'age_bins'], prefix='', prefix_sep='')

    def create_gender_group_column(self):
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
        self.calls_df['GenderGroup'] = self.calls_df['CallerDemographicsGender'].map(gender_group_mapping)

    def one_hot_encode_gender_group(self):
        self.calls_df = pd.get_dummies(self.calls_df, columns=['GenderGroup'], prefix='', prefix_sep='')

    def extract_year_and_season(self):
        self.calls_df['DateStart'] = pd.to_datetime(self.calls_df['DateStart'])
        self.calls_df['Year'] = (self.calls_df['DateStart'].dt.year >= 2020).astype(int)
        month_to_season = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        self.calls_df['Season'] = self.calls_df['DateStart'].dt.month.map(month_to_season)

    def one_hot_encode_season(self):
        self.calls_df = pd.get_dummies(self.calls_df, columns=['Season'], prefix='', prefix_sep='')

    def filter_columns(self):
        columns_to_keep = [
            'CallLength',
            'CallType',
            'Number_of_Referrals',
            'reasons_count',
            'Abuse',
            'Basic Needs',
            'Government Services',
            'Health',
            'Housing and Homelessness',
            'Income & Financial Assistance',
            'Other',
            'Substance Use',
            'time_bin',
            'Affected 3rd Party',
            'Individual',
            'Service Provider',
            'Unknown',
            'youth',
            'adult',
            'older adult',
            'unknown',
            'Group A',
            'Group B',
            'Group C',
            'Group D',
            'Very Short Duration',
            'Short Duration',
            'Moderate Duration',
            'Long Duration',
            'Very Long Duration',
            'Year',
            'Fall',
            'Spring',
            'Summer',
            'Winter'
        ]
        self.calls_df = self.calls_df[columns_to_keep]

    def get_processed_data(self):
        return self.calls_df
