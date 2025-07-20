import os
import sys
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure model_training imports resolve when running under Airflow
BASE_DIR = "/opt/airflow"
MODEL_TRAINING_DIR = os.path.join(BASE_DIR, "model_training")
sys.path.insert(0, MODEL_TRAINING_DIR)

from data_loader import load_data
from preprocessing import build_full_preprocessing_pipeline

# Paths for final model artifacts
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")
PREPROCESSING_PATH = os.path.join(FINAL_MODEL_DIR, "preprocessing_pipeline.pkl")
OUTPUT_MODEL_PATH = os.path.join(FINAL_MODEL_DIR, "final_model_pipeline.pkl")

# --- Step 1: Load Data ---
X_raw, y = load_data()

# --- Step 2: Load saved preprocessing pipeline ---
preprocessing = joblib.load(PREPROCESSING_PATH)

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

# --- Step 6: MLflow Tracking ---
# Use the mlflow service hostname from docker-compose
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("lead_conversion_retraining")

with mlflow.start_run(run_name="lgbm_retrain_run"):
    # Log hyperparameters
    mlflow.log_params({
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3
    })

    # Train model
    grid_search.fit(X_raw, y)

    # Save the best pipeline
    joblib.dump(grid_search.best_estimator_, OUTPUT_MODEL_PATH)

    # Log model to MLflow Model Registry
    mlflow.sklearn.log_model(
        sk_model=grid_search.best_estimator_,
        artifact_path="model",
        registered_model_name="LeadConversionModel"
    )

    # Evaluate on training set
    y_pred = grid_search.predict(X_raw)
    y_proba = grid_search.predict_proba(X_raw)[:, 1]

    # Log metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba)
    }
    mlflow.log_metrics(metrics)

    # Print performance
    print("\n✅ Final Model Performance:")
    for name, val in metrics.items():
        print(f"{name.capitalize():10}: {val:.4f}")




# import os
# import sys
# import pandas as pd
# import joblib
# import mlflow
# import mlflow.sklearn
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split # Import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import warnings # Import warnings to manage UndefinedMetricWarning

# # Ensure model_training imports resolve when running under Airflow
# BASE_DIR = "/opt/airflow"
# MODEL_TRAINING_DIR = os.path.join(BASE_DIR, "model_training")
# sys.path.insert(0, MODEL_TRAINING_DIR)

# from data_loader import load_data
# from preprocessing import build_full_preprocessing_pipeline

# # Paths for final model artifacts
# FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")
# PREPROCESSING_PATH = os.path.join(FINAL_MODEL_DIR, "preprocessing_pipeline.pkl")
# OUTPUT_MODEL_PATH = os.path.join(FINAL_MODEL_DIR, "final_model_pipeline.pkl")

# # --- Step 1: Load Data ---
# # load_data is expected to return X (features) and y (target)
# X_raw, y = load_data()

# # Ensure the target variable has at least two unique classes for meaningful classification
# # if len(y.unique()) < 2:
# #     raise ValueError(
# #         f"Target variable 'y' has less than two unique classes ({y.unique()}). "
# #         "Cannot perform classification training and evaluation meaningfully."
# #     )

# # --- Step 1.1: Split Data into Training and Validation Sets ---
# # Use stratified split to maintain class distribution in both sets
# X_train, X_val, y_train, y_val = train_test_split(
#     X_raw, y, test_size=0.2, random_state=42, stratify=y
# )

# # print(f"Training set shape: {X_train.shape}, {y_train.shape}")
# # print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
# # print(f"y_train value counts:\n{y_train.value_counts()}")
# # print(f"y_val value counts:\n{y_val.value_counts()}")


# # --- Step 2: Load saved preprocessing pipeline ---
# # Ensure the preprocessing pipeline exists
# if not os.path.exists(PREPROCESSING_PATH):
#     raise FileNotFoundError(f"Preprocessing pipeline not found at: {PREPROCESSING_PATH}. "
#                             "Please ensure it's built and saved before running train.py.")
# preprocessing = joblib.load(PREPROCESSING_PATH)

# # --- Step 3: Define model and hyperparameters ---
# lgbm = LGBMClassifier(random_state=42)
# param_grid = {
#     'model__n_estimators': [100],
#     'model__learning_rate': [0.1],
#     'model__max_depth': [3]
# }

# # --- Step 4: Create full pipeline with loaded preprocessing and model ---
# pipeline = Pipeline([
#     ('preprocessing', preprocessing),
#     ('model', lgbm)
# ])

# # --- Step 5: Cross-validation and hyperparameter tuning ---
# # StratifiedKFold is used within GridSearchCV, which is good for imbalanced data
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# grid_search = GridSearchCV(
#     estimator=pipeline,
#     param_grid=param_grid,
#     scoring='roc_auc', # Keep ROC AUC as primary scoring for grid search
#     cv=cv,
#     verbose=1,
#     n_jobs=-1
# )

# # --- Step 6: MLflow Tracking ---
# # Use the mlflow service hostname from docker-compose
# mlflow.set_tracking_uri("http://mlflow:5000")
# mlflow.set_experiment("lead_conversion_retraining")

# with mlflow.start_run(run_name="lgbm_retrain_run"):
#     # Log hyperparameters
#     mlflow.log_params({
#         'n_estimators': 100,
#         'learning_rate': 0.1,
#         'max_depth': 3
#     })

#     # Train model using GridSearchCV on the TRAINING data
#     print("Starting GridSearchCV training...")
#     grid_search.fit(X_train, y_train)
#     print("GridSearchCV training complete.")

#     # Save the best pipeline
#     best_pipeline = grid_search.best_estimator_
#     joblib.dump(best_pipeline, OUTPUT_MODEL_PATH)
#     print(f"Best model pipeline saved to: {OUTPUT_MODEL_PATH}")

#     # Log model to MLflow Model Registry
#     # It's recommended to log a signature for better MLOps practices
#     # You can infer signature from X_train before preprocessing if needed,
#     # or from preprocessed X_train if the model expects preprocessed input.
#     # For simplicity, logging without input_example for now, but consider adding it.
#     mlflow.sklearn.log_model(
#         sk_model=best_pipeline,
#         artifact_path="model",
#         registered_model_name="LeadConversionModel",
#         # input_example=X_train.head(1) # Consider adding an input example
#     )
#     print("Model logged to MLflow Model Registry.")

#     # --- Step 7: Evaluate on Validation Set ---
#     # Predict on the held-out VALIDATION set
#     y_pred_val = best_pipeline.predict(X_val)
#     y_proba_val = best_pipeline.predict_proba(X_val)[:, 1]

#     # Log metrics
#     metrics = {}

#     # Always calculate accuracy
#     metrics["accuracy"] = accuracy_score(y_val, y_pred_val)

#     # Handle UndefinedMetricWarning for precision, recall, f1_score
#     # Set zero_division=0 to return 0.0 when a division by zero occurs
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=UserWarning) # Catch general UserWarnings if any
#         warnings.simplefilter("ignore", category=warnings.UndefinedMetricWarning) # Specifically ignore UndefinedMetricWarning

#         metrics["precision"] = precision_score(y_val, y_pred_val, zero_division=0)
#         metrics["recall"] = recall_score(y_val, y_pred_val, zero_division=0)
#         metrics["f1_score"] = f1_score(y_val, y_pred_val, zero_division=0)

#     # Only calculate ROC AUC if y_val has more than one class
#     if len(y_val.unique()) > 1:
#         metrics["roc_auc"] = roc_auc_score(y_val, y_proba_val)
#     else:
#         metrics["roc_auc"] = float('nan') # Log NaN if ROC AUC cannot be computed
#         print("Warning: Only one class present in y_val. ROC AUC score is not defined.")

#     mlflow.log_metrics(metrics)
#     print("Metrics logged to MLflow.")

#     # Print final performance
#     print("\n✅ Final Model Performance on Validation Set:")
#     for name, val in metrics.items():
#         print(f"{name.capitalize():10}: {val:.4f}")

# print("\nModel training and evaluation complete.")




# import os
# import sys
# import pandas as pd
# import joblib
# import mlflow
# import mlflow.sklearn
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import warnings

# # --- Setup Paths and Imports ---
# # This ensures that the script can find custom modules when run by Airflow
# BASE_DIR = "/opt/airflow" 
# # Assuming your custom modules are in a 'model_training' directory
# # Adjust if your project structure is different
# MODEL_TRAINING_DIR = os.path.join(BASE_DIR, "model_training") 
# if MODEL_TRAINING_DIR not in sys.path:
#     sys.path.insert(0, MODEL_TRAINING_DIR)

# from data_loader import load_data
# # The preprocessing module is needed because the saved pipeline object
# # requires the definition of custom classes like CategoricalPreprocessor
# from preprocessing import build_full_preprocessing_pipeline, CategoricalPreprocessor

# # Define paths for loading the preprocessing pipeline and saving the final model
# FINAL_MODEL_DIR = os.path.join(BASE_DIR, "working_model")
# PREPROCESSING_PATH = os.path.join(FINAL_MODEL_DIR, "preprocessing_pipeline.pkl")
# OUTPUT_MODEL_PATH = os.path.join(FINAL_MODEL_DIR, "final_model_pipeline.pkl")
# os.makedirs(FINAL_MODEL_DIR, exist_ok=True)


# # --- Step 1: Load and Split Data ---
# print("▶ Step 1: Loading data...")
# X_raw, y = load_data()

# # It's crucial to split the data before training to get an unbiased evaluation
# print("▶ Step 1.1: Splitting data into training and validation sets...")
# X_train, X_val, y_train, y_val = train_test_split(
#     X_raw, y, test_size=0.2, random_state=42, stratify=y
# )
# print(f"Training set size: {len(X_train)} rows")
# print(f"Validation set size: {len(X_val)} rows")


# # --- Step 2: Load Preprocessing Pipeline ---
# print(f"▶ Step 2: Loading preprocessing pipeline from {PREPROCESSING_PATH}...")
# if not os.path.exists(PREPROCESSING_PATH):
#     raise FileNotFoundError(f"Preprocessing pipeline not found at: {PREPROCESSING_PATH}. Please run the preprocessing build script first.")
# preprocessing_pipeline = joblib.load(PREPROCESSING_PATH)


# # --- Step 3: Define Model and Search Grid ---
# print("▶ Step 3: Defining model and hyperparameter grid...")
# lgbm = LGBMClassifier(random_state=42)
# param_grid = {
#     'model__n_estimators': [100, 200],
#     'model__learning_rate': [0.05, 0.1],
#     'model__max_depth': [3, 5]
# }


# # --- Step 4: Create Full Training Pipeline ---
# # This combines the loaded preprocessing steps with the model
# full_pipeline = Pipeline([
#     ('preprocessing', preprocessing_pipeline),
#     ('model', lgbm)
# ])


# # --- Step 5: Hyperparameter Tuning with GridSearchCV ---
# print("▶ Step 5: Setting up GridSearchCV for hyperparameter tuning...")
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# grid_search = GridSearchCV(
#     estimator=full_pipeline,
#     param_grid=param_grid,
#     scoring='roc_auc',
#     cv=cv,
#     verbose=1,
#     n_jobs=-1
# )


# # --- Step 6: MLflow Experiment Tracking ---
# print("▶ Step 6: Starting MLflow run...")
# mlflow.set_tracking_uri("http://mlflow:5000")
# mlflow.set_experiment("lead_conversion_retraining")

# with mlflow.start_run(run_name="lgbm_gridsearch_retrain"):
#     # Train the model on the training data
#     grid_search.fit(X_train, y_train)

#     # --- Save the Best Model ---
#     best_pipeline = grid_search.best_estimator_
#     print(f"\n▶ Saving best model pipeline to: {OUTPUT_MODEL_PATH}")
#     joblib.dump(best_pipeline, OUTPUT_MODEL_PATH)
    
#     # Log best hyperparameters to MLflow
#     mlflow.log_params(grid_search.best_params_)

#     # Log the best model to MLflow Model Registry
#     print("▶ Logging model to MLflow Model Registry...")
#     mlflow.sklearn.log_model(
#         sk_model=best_pipeline,
#         artifact_path="model",
#         registered_model_name="LeadConversionModel",
#         input_example=X_train.head(5) # Add an input example for schema inference
#     )

#     # --- Step 7: Evaluate on Held-Out Validation Set ---
#     print("▶ Step 7: Evaluating model on the validation set...")
#     y_pred_val = best_pipeline.predict(X_val)
#     y_proba_val = best_pipeline.predict_proba(X_val)[:, 1]

#     # Calculate and log metrics, handling potential warnings
#     metrics = {
#         "val_accuracy": accuracy_score(y_val, y_pred_val),
#         "val_precision": precision_score(y_val, y_pred_val, zero_division=0),
#         "val_recall": recall_score(y_val, y_pred_val, zero_division=0),
#         "val_f1_score": f1_score(y_val, y_pred_val, zero_division=0)
#     }
#     if len(y_val.unique()) > 1:
#         metrics["val_roc_auc"] = roc_auc_score(y_val, y_proba_val)
#     else:
#         metrics["val_roc_auc"] = float('nan')

#     mlflow.log_metrics(metrics)
#     print("▶ Metrics logged to MLflow.")

#     # Print final performance for the logs
#     print("\n✅ Final Model Performance on Validation Set:")
#     for name, val in metrics.items():
#         print(f"  {name.replace('_', ' ').capitalize():15}: {val:.4f}")

# print("\n🎉 Model training and evaluation complete.")




