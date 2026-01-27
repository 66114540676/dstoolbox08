import pandas as pd
from pycaret.classification import *

# Load sample dataset (Iris dataset for classification)
from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Setup PyCaret environment
clf_setup = setup(data, target='target', session_id=123)

# Compare models
best_model = compare_models()

# Create and tune the best model
tuned_model = tune_model(best_model)

# Evaluate the model
evaluate_model(tuned_model)

# Make predictions
predictions = predict_model(tuned_model)

# Save the model
save_model(tuned_model, 'best_iris_model')

print("PyCaret workflow completed. Model saved as 'best_iris_model.pkl'")