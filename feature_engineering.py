# feature_engineering.py

import pandas as pd

def create_new_features(df):
    df['age_adherence_interaction'] = df['age'] * df['adherence_rate']
    df['dosage_deviation'] = (df['dosage'] - df['standard_dosage']) / df['standard_dosage']
    df['treatment_compliance_score'] = df['adherence_rate'] * df['duration_days'] / df['duration_days'].max()
    return df
