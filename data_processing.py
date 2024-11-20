import pandas as pd
import joblib 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def handling_missing_values(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Impute numeric columns using SimpleImputer
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

    # Impute categorical columns with mode
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    return df

def preprocess_data(df, fit=True):
    """Preprocess the data to match the training data pipeline."""
    categorical_cols = ['drug_name', 'category', 'gender']

    if fit:
        # Create and fit the encoder during the training phase
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(df[categorical_cols])

        # Save the encoder for future use
        joblib.dump(encoder, 'Model_Development/encoder.pkl')
    else:
        # Load the pre-trained encoder
        encoder = joblib.load('Model_Development/encoder.pkl')
        encoded_features = encoder.transform(df[categorical_cols])

    # Convert the encoded features to a DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and add encoded columns
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df
