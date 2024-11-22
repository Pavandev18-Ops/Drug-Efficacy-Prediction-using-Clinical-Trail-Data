#app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from data_processing import load_data, handling_missing_values, preprocess_data
from model_save_load import load_model

# Load the dataset and the trained model
df = load_data('clinical_trial_dataset.csv')

# Specify the path to your .pkl file
model_path = 'models/gradient_boosting_best_model.pkl'

# Load model and feature names
model, feature_names = load_model(model_path)

# Extract unique values for dropdowns and value ranges for sliders
unique_drug_names = df['drug_name'].unique()
age_min, age_max = int(df['age'].min()), int(df['age'].max())
bmi_min, bmi_max = round(df['bmi'].min(), 1), round(df['bmi'].max(), 1)
socio_min, socio_max = round(df['socioeconomic_score'].min(), 1), round(df['socioeconomic_score'].max(), 1)

def get_drug_category(selected_drug):
    category = df[df['drug_name'] == selected_drug]['category'].iloc[0]
    return category

st.title("Drug Efficacy Score Prediction")

# Collect user input for new patient data
st.header("Enter New Patient Data:")
drug_name = st.selectbox("Drug Name", unique_drug_names)
drug_category = get_drug_category(drug_name)
st.write(f"Category for selected drug: {drug_category}")

duration_days = st.number_input("Duration Days", min_value=1, max_value=365, value=30)
standard_dosage = st.number_input("Standard Dosage (mg)", min_value=1.0, value=50.0, step=0.1)
dosage = st.number_input("Dosage (mg)", min_value=1.0, value=50.0, step=0.1)
adherence_rate = st.slider("Adherence Rate (%)", min_value=0, max_value=100, value=80)
age = st.slider("Age", min_value=age_min, max_value=age_max, value=30)
gender = st.selectbox("Gender", df['gender'].unique())  # This will give you 'M' or 'F'
bmi = st.slider("BMI", min_value=bmi_min, max_value=bmi_max, value=25.0)
socioeconomic_score = st.slider("Socioeconomic Score", min_value=socio_min, max_value=socio_max, value=50.0)



# Create a DataFrame for new patient data
user_data = pd.DataFrame({
    'drug_name': [drug_name],
    'category': [drug_category],
    'duration_days': [duration_days],
    'standard_dosage': [standard_dosage],
    'dosage': [dosage],
    'adherence_rate': [adherence_rate],
    'age': [age],
    'gender': [gender],
    'bmi': [bmi],
    'socioeconomic_score': [socioeconomic_score]
})

# Remove the columns if they were dropped during training
user_data = user_data.drop(columns=['treatment_id', 'patient_id', 'drug_id', 'insurance_type', 'region'], errors='ignore')

# Categorical Encoding: Encode 'gender' column using LabelEncoder
label_encoder = LabelEncoder()
user_data['gender'] = label_encoder.fit_transform(user_data['gender'])

# Ensure the user input data has the same columns as the model expects
missing_cols = set(feature_names) - set(user_data.columns)

for col in missing_cols:
    if col in df.columns:  # Check if the column exists in df
        if df[col].dtype == 'object':  # Categorical column
            user_data[col] = df[col].mode()[0]  # Default value for categorical
        else:  # Numeric column
            user_data[col] = df[col].mean()  # Default value for numeric
    else:
        # Assign a default placeholder for truly missing columns
        user_data[col] = 0  # Or another reasonable default
    

# Ensure the columns are in the correct order as expected by the model
user_data = user_data[feature_names]

# Add preprocessing here if necessary (e.g., scaling, encoding)

# Add a button for prediction
if st.button("Predict"):
    # Prediction logic
    prediction = model.predict(user_data[feature_names])
    
    # Display the prediction result
    st.write(f"Predicted Drug Efficacy Score: {prediction[0]}")
