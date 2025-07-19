# nnsk
# xnjnsk
# xsnmkxnsknks
# nxsnjnsj
# njsanlk
# jnjxansj
# kjncdanjns





import pandas as pd
import os
import json
import sys # Import sys module

# Add the project root to the Python path
# Assuming check_drift.py is in lead_conversion_project/drift_detection/
# We need to go up two levels to reach lead_conversion_project/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from model_training.data_loader import load_data

BASE_DIR = "/opt/airflow"
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")

# Load reference and new data
reference_data_path = os.path.join(FINAL_MODEL_DIR, "reference_data.csv")
reference_data = pd.read_csv(reference_data_path)

new_data, _ = load_data()  # From Postgres

# Create and run drift report
report = Report(metrics=[DataDriftPreset()])
my_report = Report(metrics=[DataDriftPreset()]) \
    .run(
      current_data=new_data, 
      reference_data=reference_data
    )
# Extract result
result = my_report.dict()

print("=== EVIDENTLY OUTPUT ===")
print(json.dumps(result, indent=2))
# Grab the first metric (DriftedColumnsCount)
drifted_cols_metric = result["metrics"][0]

# It looks like:
# {
#    "metric_id": "DriftedColumnsCount(drift_share=0.5)",
#    "value": {"count": X, "share": Y}
# }
count = drifted_cols_metric["value"].get("count", 0)

# Decide: if any columns drifted, we say "drift detected"
drift_detected = count > 0


# Write drift flag
flag_path = os.path.join(FINAL_MODEL_DIR, "drift_detected.txt")
with open(flag_path, "w") as f:
    f.write("true" if drift_detected else "false")

print(f"Data drift detected? {drift_detected}")