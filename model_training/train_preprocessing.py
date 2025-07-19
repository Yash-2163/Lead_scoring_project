from data_loader import load_data
from preprocessing import build_full_preprocessing_pipeline
import joblib

X, y = load_data()

pipeline = build_full_preprocessing_pipeline()
pipeline.fit(X)

joblib.dump(pipeline, '../final_model/preprocessing_pipeline.pkl')
print("✅ Preprocessing pipeline saved.")
