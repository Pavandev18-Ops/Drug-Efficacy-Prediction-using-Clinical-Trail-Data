import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle  # Use pickle instead of joblib

# Load the dataset
df = pd.read_csv('clinical_trial_dataset.csv')

# Specify the categorical columns
categorical_cols = ['drug_name', 'category', 'gender']

# Create and fit the encoder (use sparse_output instead of sparse)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(df[categorical_cols])

# Save the encoder to a file using pickle
with open('Model_Development/encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

print("Encoder saved successfully!")
