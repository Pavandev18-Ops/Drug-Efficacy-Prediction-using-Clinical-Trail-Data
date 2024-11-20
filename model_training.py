# model_training.py

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train):
    gb_model = GradientBoostingRegressor(random_state=43)
    gb_model.fit(X_train, y_train)
    return gb_model

def hyperparameter_tuning(X_train, y_train):
    gb_model = GradientBoostingRegressor(random_state=43)
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 4],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', None]
    }

    grid_search_gb = GridSearchCV(
        estimator=gb_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search_gb.fit(X_train, y_train)
    return grid_search_gb.best_estimator_, grid_search_gb.best_params_