import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
from scipy.stats import randint

def load_and_prep_data(data_path: str):
    """Load and prepare data for training."""
    data = pd.read_csv(data_path)
    X = data.drop(columns=["date", "demand"])
    X = X.astype('float')
    y = data["demand"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Basic setup
    EXPERIMENT_NAME = "Challenge"
    N_TRIALS = 10

    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Enable autologging
    mlflow.sklearn.autolog(
        log_models=True
    )

    # Load data
    X_train, X_val, y_train, y_val = load_and_prep_data("data/fake_data.csv")

    # Define parameter search space
    param_distributions = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4),
    }

    # Create and run RandomizedSearchCV
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=N_TRIALS,
        cv=5,
        scoring='r2',
        random_state=42
    )

    # Fit the model - autolog will automatically create the runs
    search.fit(X_train, y_train)

if __name__ == "__main__":
    main()