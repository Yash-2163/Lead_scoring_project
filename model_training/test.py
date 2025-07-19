import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_loader import load_data

# Constants
SAMPLE_SIZE = 300
RANDOM_STATE = 42

# --- Load full pipeline (preprocessing + model) ---
model_pipeline = joblib.load('../final_model/final_model_pipeline.pkl')

# --- Load raw data ---
X, y = load_data()

# --- Take random subset for testing ---
sample_df = X.sample(n=min(SAMPLE_SIZE, len(X)), random_state=RANDOM_STATE)
y_sample = y.loc[sample_df.index]

# --- Predict using full pipeline (which includes preprocessing) ---
y_pred = model_pipeline.predict(sample_df)
y_proba = model_pipeline.predict_proba(sample_df)[:, 1]

# --- Evaluation ---
print("📊 Test Set Performance on Sample:")
print("Accuracy: ", round(accuracy_score(y_sample, y_pred), 4))
print("Precision:", round(precision_score(y_sample, y_pred), 4))
print("Recall:   ", round(recall_score(y_sample, y_pred), 4))
print("F1 Score: ", round(f1_score(y_sample, y_pred), 4))
print("ROC AUC:  ", round(roc_auc_score(y_sample, y_proba), 4))
