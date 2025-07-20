# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import pickle
# import os
# import logging

# # --- Setup Logging ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # --- Flask App Setup ---
# app = Flask(__name__)
# CORS(app)

# # --- Load Model Pipeline ---
# MODEL_PATH = os.path.join("..", "working_model", "final_model_pipeline.pkl")

# try:
#     with open(MODEL_PATH, 'rb') as f:
#         model = pickle.load(f)
#     logger.info("✅ Model loaded successfully from %s", MODEL_PATH)
# except FileNotFoundError:
#     logger.error("❌ Model file not found at %s", MODEL_PATH)
#     model = None
# except Exception as e:
#     logger.error("❌ Failed to load model: %s", e)
#     model = None

# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None:
#         return jsonify({"error": "Model not loaded"}), 500

#     if 'file' not in request.files:
#         return jsonify({"error": "No file part in the request"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     try:
#         # Try reading CSV (fallback to semi-colon if needed)
#         try:
#             input_df = pd.read_csv(file)
#         except pd.errors.ParserError:
#             file.seek(0)
#             input_df = pd.read_csv(file, delimiter=';')

#         if input_df.empty:
#             return jsonify({"error": "Uploaded file is empty"}), 400

#         # Predict
#         predictions = model.predict(input_df)
#         prediction_proba = model.predict_proba(input_df)[:, 1]

#         results_df = input_df.copy()
#         results_df['Predicted_Conversion'] = predictions
#         results_df['Prediction_Probability'] = prediction_proba

#         return jsonify({"predictions": results_df.to_dict(orient='records')})

#     except Exception as e:
#         logger.exception("Prediction failed")
#         return jsonify({'error': f"Prediction failed: {e}"}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({"status": "OK", "model_loaded": model is not None}), 200

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5001, debug=True)



import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- IMPORTANT ---
# This backend requires your custom preprocessing module to load the model.
# Ensure 'preprocessing.py' is in the same directory as this 'app.py' file.
try:
    from preprocessing import CategoricalPreprocessor
    print("[INFO] Custom 'preprocessing' module loaded.")
except ImportError:
    print("[ERROR] 'preprocessing.py' not found. Model loading will fail.")
    # In a real app, you might exit here if the module is critical
    # sys.exit(1) 

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Load the Trained Model ---
# Assumes the 'final_model' directory is at the project root, one level above this 'backend' directory.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, "final_model", "final_model_pipeline.pkl")

model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"[ERROR] Model file not found at {MODEL_PATH}. Please ensure the training script has been run and the model is saved.")
except Exception as e:
    print(f"[ERROR] An error occurred while loading the model: {e}")

# --- API Endpoints ---
@app.route("/predict", methods=["POST"])
def predict():
    """Receives a file upload, runs prediction, and returns results."""
    if model is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        input_df = pd.read_csv(file)
        
        # The loaded model is a full pipeline, handling all preprocessing internally.
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1] # Probability of class 1

        # Add results back to a copy of the input data for context
        results_df = input_df.copy()
        results_df['Predicted_Conversion'] = predictions
        results_df['Prediction_Probability'] = [f"{p:.2%}" for p in probabilities]

        # Return the results in a JSON format
        return jsonify({"predictions": results_df.to_dict(orient='records')})

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """A simple endpoint to check if the service is running and the model is loaded."""
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    # Runs the Flask app. Port 5000 is standard for backend services.
    app.run(host="0.0.0.0", port=5001, debug=True)
