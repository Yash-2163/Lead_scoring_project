import pandas as pd
import os
import json
import sys  # Import sys module

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from evidently import Report
from evidently.presets import DataDriftPreset
from model_training.data_loader import load_data

BASE_DIR = "/opt/airflow"
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")

# ─── Step 1: Load reference and new data ────────────────────────────────
reference_data_path = os.path.join(FINAL_MODEL_DIR, "reference_data.csv")
reference_data = pd.read_csv(reference_data_path)
new_data, y = load_data()  # From Postgres

# Normalize columns to lowercase for consistent matching
reference_data.columns = reference_data.columns.str.lower()
new_data.columns = new_data.columns.str.lower()

# ─── Step 2: Align schemas (drop ID cols) ────────────────────────────────
ID_COLS = ["prospect_id", "lead_number"]
reference_data = reference_data.drop(columns=ID_COLS, errors="ignore")
new_data = new_data.drop(columns=ID_COLS, errors="ignore")

new_data['converted']=y

# Optional: keep only common columns
common_cols = reference_data.columns.intersection(new_data.columns)
reference_data = reference_data[common_cols]
new_data = new_data[common_cols]

# ─── Step 2.1: Convert 'converted' column to int (or numeric) ────────────
for df_name, df in [("reference_data", reference_data), ("new_data", new_data)]:
    if 'converted' in df.columns:
        # Convert to numeric safely, fill NAs with 0, then convert to int
        df['converted'] = pd.to_numeric(df['converted'], errors='coerce').fillna(0).astype(int)
    else:
        raise KeyError(f"'converted' column missing in {df_name}")

# ─── Step 3: Create and run drift report ─────────────────────────────────
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(
    reference_data=reference_data,
    current_data=new_data
)

# ─── Step 4: Extract results dict ────────────────────────────────────────
result = snapshot.dict()
print("=== EVIDENTLY OUTPUT ===")
print(json.dumps(result, indent=2))

# ─── Step 5: Determine drift based on DriftedColumnsCount ───────────────
drifted_cols_metric = result.get("metrics", [])[0]
count = drifted_cols_metric.get("value", {}).get("count", 0)
drift_detected = count > 0

# ─── Step 6: Write flag ─────────────────────────────────────────────────
flag_path = os.path.join(FINAL_MODEL_DIR, "drift_detected.txt")
with open(flag_path, "w") as f:
    f.write("true" if drift_detected else "false")

print(f"Data drift detected? {drift_detected}")



# import pandas as pd
# import os
# import json
# import sys

# # Add the project root to the Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.insert(0, project_root)

# from evidently import Report
# from evidently.presets import DataDriftPreset
# # Make sure to import the correct DataDriftPreset if using a newer Evidently version
# # from evidently.metric_preset import DataDriftPreset

# from model_training.data_loader import load_data

# BASE_DIR = "/opt/airflow"
# FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")

# # ─── Step 1: Load reference and new data ────────────────────────────────
# reference_data_path = os.path.join(FINAL_MODEL_DIR, "reference_data.csv")
# # Add a check for reference_data.csv existence
# if not os.path.exists(reference_data_path):
#     raise FileNotFoundError(f"Reference data file not found at: {reference_data_path}")
# reference_data = pd.read_csv(reference_data_path)
# new_data_X, y = load_data() # Assuming load_data returns X, y (features, target)
# new_data = new_data_X # Use only features for drift detection, as previously discussed
# new_data['converted']=y

# # Normalize columns to lowercase for consistent matching
# # Apply this immediately after loading to ensure consistency for all subsequent operations
# reference_data.columns = reference_data.columns.str.lower()
# new_data.columns = new_data.columns.str.lower()

# # ─── Step 2: Align schemas (drop ID cols) ────────────────────────────────
# ID_COLS = ["prospect_id", "lead_number"]
# # Drop ID columns from both DataFrames
# reference_data = reference_data.drop(columns=ID_COLS, errors="ignore")
# new_data = new_data.drop(columns=ID_COLS, errors="ignore")


# # ─── Step 2.1: Convert 'converted' column to int (or numeric) ────────────
# # Apply conversion *before* filtering common columns IF 'converted' needs to be kept
# # in the dataset for drift analysis (e.g., as a feature, not just target).
# # Given your problem is lead conversion, 'converted' is likely the target.
# # If 'converted' is a target and should NOT be part of drift *features*, then:
# # 1. Ensure `load_data` returns X and y.
# # 2. Use X for drift detection.
# # 3. Handle 'converted' (y) separately if needed for target drift, but not in DataDriftPreset (which is for features).

# # Assuming 'converted' is still present in the dataframes for some form of analysis
# # (even if it's the target, it might be included in the reference_data.csv that Evidently processes)
# # If 'converted' is intended to be excluded from *feature* drift, adjust `new_data` and `reference_data`
# # BEFORE passing them to the Evidently report.

# # Let's assume 'converted' *should* be in the dataframes for drift analysis.
# # If it's your *target*, DataDriftPreset can still analyze it, but it might be better
# # to have a separate target drift report. For now, we'll keep it in.
# for df_name, df in [("reference_data", reference_data), ("new_data", new_data)]:
#     if 'converted' in df.columns:
#         # Convert to numeric safely, fill NAs with 0, then convert to int
#         df['converted'] = pd.to_numeric(df['converted'], errors='coerce').fillna(0).astype(int)
#     else:
#         # This KeyError is still valid if 'converted' is truly expected but missing
#         raise KeyError(f"'converted' column missing in {df_name}. Check data source or preprocessing.")


# # ─── Crucial Fix: Revisit "keep only common columns" logic ────────────────
# # If 'converted' is intended to be part of the drift detection, this filtering step
# # should ensure it's present in both *before* the filtering, and thus in `common_cols`.
# # Alternatively, if 'converted' is the target and not a feature for drift,
# # then remove it from the DataFrames *before* this common_cols intersection.

# # For drift detection on FEATURES, it's common to remove the target.
# # Let's adjust based on the assumption that `load_data()` already provides X (features)
# # and `reference_data.csv` might contain target.
# # It's best to explicitly decide what columns go into drift detection.

# # If `converted` is supposed to be present in `reference_data` for drift analysis,
# # the `common_cols` logic should NOT remove it implicitly.
# # The error means that after `reference_data.columns.intersection(new_data.columns)`,
# # `converted` was *not* in `common_cols`. This implies `new_data` did not have 'converted'.

# # Let's ensure 'converted' is always considered if it's in the data:
# # First, extract features for drift detection, excluding the target if it's not a feature.
# # Assuming 'converted' is the TARGET and should NOT be part of feature drift detection:

# # Remove 'converted' from reference_data if it's present, before intersection
# if 'converted' in reference_data.columns:
#     reference_data_for_drift = reference_data.drop(columns=['converted'], errors='ignore')
# else:
#     reference_data_for_drift = reference_data.copy()

# # `new_data` is already `new_data_X` from `load_data()`, so it should *not* have 'converted' if `load_data` correctly drops target.
# # If `new_data` *still* has 'converted', then your `load_data` needs a fix.
# if 'converted' in new_data.columns:
#     print("Warning: 'converted' column found in `new_data` (features). It will be dropped for drift detection.")
#     new_data_for_drift = new_data.drop(columns=['converted'], errors='ignore')
# else:
#     new_data_for_drift = new_data.copy()


# # Now, keep only common columns on the *features* intended for drift
# common_feature_cols = reference_data_for_drift.columns.intersection(new_data_for_drift.columns)
# reference_data_for_drift = reference_data_for_drift[common_feature_cols]
# new_data_for_drift = new_data_for_drift[common_feature_cols]

# # ─── Step 3: Create and run drift report ─────────────────────────────────
# # Pass the dataframes specifically prepared for drift detection
# report = Report(metrics=[DataDriftPreset()])
# snapshot = report.run(
#     reference_data=reference_data_for_drift,
#     current_data=new_data_for_drift
# )

# # ─── Step 4: Extract results dict ────────────────────────────────────────
# result = snapshot.dict()
# print("=== EVIDENTLY OUTPUT ===")
# print(json.dumps(result, indent=2))

# # ─── Step 5: Determine drift based on DriftedColumnsCount ───────────────
# # Check if metrics are present before accessing
# if result.get("metrics") and len(result["metrics"]) > 0:
#     drifted_cols_metric = result["metrics"][0]
#     count = drifted_cols_metric.get("result", {}).get("drift_detected", False) # DataDriftPreset has 'drift_detected' directly
#     drift_detected = count # Use directly if it's a boolean
# else:
#     print("Warning: No metrics found in Evidently report. Assuming no drift detected.")
#     drift_detected = False

# # The previous logic `count = drifted_cols_metric.get("value", {}).get("count", 0)`
# # is for older Evidently versions or different presets.
# # For DataDriftPreset, the boolean `drift_detected` is usually under `result` directly.
# # Let's refine based on typical DataDriftPreset output:
# # result['metrics'][0]['result']['drift_detected'] will be True/False
# # result['metrics'][0]['result']['number_of_drifted_columns'] for count

# # Re-evaluating based on DataDriftPreset output structure
# # Try to get the specific 'drift_detected' flag from DataDriftPreset
# try:
#     drift_detected_flag_from_evidently = result["metrics"][0]["result"]["drift_detected"]
#     print(f"Evidently's primary drift_detected flag: {drift_detected_flag_from_evidently}")
#     drift_detected = drift_detected_flag_from_evidently
# except (KeyError, IndexError):
#     print("Could not find 'drift_detected' flag in expected Evidently report structure. Fallback to count > 0.")
#     # Fallback to checking drifted column count if the flag isn't directly available
#     num_drifted_columns = result.get("metrics", [])[0].get("result", {}).get("number_of_drifted_columns", 0)
#     drift_detected = num_drifted_columns > 0


# # ─── Step 6: Write flag ─────────────────────────────────────────────────
# flag_path = os.path.join(FINAL_MODEL_DIR, "drift_detected.txt")
# with open(flag_path, "w") as f:
#     f.write("true" if drift_detected else "false")

# print(f"Data drift detected? {drift_detected}")