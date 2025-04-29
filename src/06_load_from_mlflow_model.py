import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd

# Import Database
data = pd.read_csv("data/fake_data.csv")
X = data.drop(columns=["date", "demand"])
X = X.astype('float')

# Define MLflow Model path
# You can use the UI to get both the experiment ID and run ID
mlflow.set_tracking_uri('http://localhost:8080')
model_path = "runs:/79ea02313f7741d38363bd54fa39cac0/rf_apples"

# Load model with sklearn flavor
model = mlflow.sklearn.load_model(model_path)

# Make predictions
predictions = model.predict(X)

# Calculate the mean prediction
mean_prediction = predictions.mean()
print(mean_prediction)