#main.py
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from model_training import train_model, hyperparameter_tuning
from model_save_load import save_model

# Generate synthetic regression data for demonstration
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning to find the best model
best_model, best_params, feature_names = hyperparameter_tuning(X_train, y_train)

# Save the best model to a file
model_file_path = 'models/gradient_boosting_best_model.pkl'
save_model(best_model,feature_names, model_file_path)
