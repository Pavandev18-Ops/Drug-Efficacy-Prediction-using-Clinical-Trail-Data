import pickle

# Function to save both the model and feature names together
def save_model(model, feature_names, file_path):
    # Store the model and feature names in a dictionary
    model_info = {
        'model': model,
        'feature_names': feature_names
    }
    # Save the dictionary as a .pkl file
    with open(file_path, 'wb') as file:  # Open file in write-binary mode
        pickle.dump(model_info, file)
    print("Model and feature names saved successfully!")

# Function to load the model and feature names from a file
def load_model(model_path):
    # Load the saved model and feature names dictionary
    with open(model_path, 'rb') as file:  # Open file in read-binary mode
        model_info = pickle.load(file)
        
    model = model_info['model']
    feature_names = model_info['feature_names']
    return model, feature_names
