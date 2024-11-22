#model_save_load.py
import pickle

# Function to save both the model and feature names together
def save_model(model, feature_names, file_path):
    # Store the model and feature names in a dictionary
    model_info = {
        'model': model,
        'feature_names': feature_names
    }
    # Save the dictionary as a .pkl file
    pickle.dump(model_info, file_path)

# Function to load the model and feature names from a file
def load_model(file_path):
    # Load the saved model and feature names dictionary
    model_info = pickle.load(file_path)
    model = model_info['model']
    feature_names = model_info['feature_names']
    return model, feature_names
