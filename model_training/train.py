# Need to add ML flow

import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from preprocessing import build_full_preprocessing_pipeline 
from data_loader import load_data

# --- Step 1: Load Data ---
X_raw, y = load_data()

# --- Step 2: Load saved preprocessing pipeline ---
preprocessing = joblib.load('../final_model/preprocessing_pipeline.pkl')

# --- Step 3: Define model and hyperparameters ---
lgbm = LGBMClassifier(random_state=42)
param_grid = {
    'model__n_estimators': [100],
    'model__learning_rate': [0.1],
    'model__max_depth': [3]
}

# --- Step 4: Create full pipeline with loaded preprocessing and model ---
pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('model', lgbm)
])

# --- Step 5: Cross-validation and hyperparameter tuning ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

# --- Step 6: Train model ---
grid_search.fit(X_raw, y)

# --- Step 7: Save the best pipeline (preprocessing + model) ---
joblib.dump(grid_search.best_estimator_, '../final_model/final_model_pipeline.pkl')

# --- Step 8: Evaluate on training set ---
y_pred = grid_search.predict(X_raw)
y_proba = grid_search.predict_proba(X_raw)[:, 1]

print("\n✅ Final Model Performance:")
print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall   : {recall_score(y, y_pred):.4f}")
print(f"F1 Score : {f1_score(y, y_pred):.4f}")
print(f"ROC AUC  : {roc_auc_score(y, y_proba):.4f}")
