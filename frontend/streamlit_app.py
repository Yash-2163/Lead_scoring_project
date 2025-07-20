# This is the code used when we spin up intances locally



import streamlit as st
import pandas as pd
import requests
import io

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Lead Conversion Predictor",
    page_icon="🚀",
    layout="centered"
)

# --- App Title and Description ---
st.title("🚀 Lead Conversion Predictor (POC)")
st.write(
    "Upload a CSV file with lead data to predict which leads are most likely to convert. "
    "This is a proof-of-concept using a trained machine learning model."
)

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Display a preview of the uploaded data
    try:
        df_preview = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df_preview.head())

        # Reset file pointer to be read again for the request
        uploaded_file.seek(0)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # --- Prediction Button ---
    if st.button("Get Predictions"):
        with st.spinner("Sending data to the model... Please wait."):
            # Define the backend URL. In Docker, 'backend' is the service name.
            # For local testing, you might change 'backend' to 'localhost'.
            backend_url = "http://localhost:5001/predict"
            
            try:
                # Prepare the file for the POST request
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                
                # Send the request to the Flask backend
                response = requests.post(backend_url, files=files, timeout=30)

                # --- Handle the Response ---
                if response.status_code == 200:
                    st.success("✅ Predictions received successfully!")
                    
                    # Parse the JSON response and display the results
                    results_data = response.json()
                    results_df = pd.DataFrame(results_data["predictions"])
                    
                    st.write("Prediction Results:")
                    st.dataframe(results_df)

                else:
                    st.error(f"Error from backend: {response.status_code}")
                    try:
                        st.json(response.json())
                    except requests.exceptions.JSONDecodeError:
                        st.text(response.text)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Connection Error: Could not connect to the backend service. "
                    "Please ensure the backend container is running and accessible."
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin.")



# code used to run when we use docker to spin up instances

# import streamlit as st
# import pandas as pd
# import requests
# import io
# import os

# # --- Streamlit Page Configuration ---
# st.set_page_config(
#     page_title="Lead Conversion Predictor",
#     page_icon="🚀",
#     layout="centered"
# )

# # --- App Title and Description ---
# st.title("🚀 Lead Conversion Predictor (POC)")
# st.write(
#     "Upload a CSV file with lead data to predict which leads are most likely to convert. "
#     "This is a proof-of-concept using a trained machine learning model."
# )

# # --- File Uploader ---
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     # Display a preview of the uploaded data
#     try:
#         df_preview = pd.read_csv(uploaded_file)
#         st.write("Data Preview:")
#         st.dataframe(df_preview.head())

#         # Reset file pointer to be read again for the request
#         uploaded_file.seek(0)
#     except Exception as e:
#         st.error(f"Error reading CSV file: {e}")
#         st.stop()

#     # --- Prediction Button ---
#     if st.button("Get Predictions"):
#         with st.spinner("Sending data to the model... Please wait."):
#             # --- FIX for Docker ---
#             # When running in Docker Compose, services communicate using their service names.
#             # The backend service is named 'backend' in docker-compose.yml and runs on port 5000.
#             backend_url = "http://backend:5001/predict"
            
#             try:
#                 # Prepare the file for the POST request
#                 files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                
#                 # Send the request to the Flask backend
#                 response = requests.post(backend_url, files=files, timeout=30)

#                 # --- Handle the Response ---
#                 if response.status_code == 200:
#                     st.success("✅ Predictions received successfully!")
                    
#                     # Parse the JSON response and display the results
#                     results_data = response.json()
#                     results_df = pd.DataFrame(results_data["predictions"])
                    
#                     st.write("Prediction Results:")
#                     st.dataframe(results_df)

#                 else:
#                     st.error(f"Error from backend: {response.status_code}")
#                     try:
#                         st.json(response.json())
#                     except requests.exceptions.JSONDecodeError:
#                         st.text(response.text)

#             except requests.exceptions.ConnectionError:
#                 st.error(
#                     f"Connection Error: Could not connect to the backend service at {backend_url}. "
#                     "Please ensure the backend container is running and accessible."
#                 )
#             except Exception as e:
#                 st.error(f"An unexpected error occurred: {e}")
# else:
#     st.info("Please upload a CSV file to begin.")
