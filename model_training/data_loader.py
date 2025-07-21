# Loading data from local 

import pandas as pd
import os

def load_data(target_col='Converted'):
    """
    Loads raw data and splits into features and target.
    No preprocessing here.
    """
    # Define the base directory for your project within the Docker container
    BASE_DIR = "/opt/airflow"

    # Construct the absolute path to the data file
    # Assuming your data file is at /opt/airflow/data/Lead_Scoring.csv
    file_path = os.path.join(BASE_DIR, 'data', 'sample_data1.csv')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}. Please ensure it's in the 'data' directory under the project root (/opt/airflow/data/).")

    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y # Typically, load_data should return X and y, not the full df and y


