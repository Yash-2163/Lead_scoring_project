# Loading data from local 

# import pandas as pd
# import os

# def load_data(target_col='Converted'):
#     """
#     Loads raw data and splits into features and target.
#     No preprocessing here.
#     """
#     # Define the base directory for your project within the Docker container
#     BASE_DIR = "/opt/airflow"

#     # Construct the absolute path to the data file
#     # Assuming your data file is at /opt/airflow/data/Lead_Scoring.csv
#     file_path = os.path.join(BASE_DIR, 'data', 'sample_dataset.csv')

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Data file not found at: {file_path}. Please ensure it's in the 'data' directory under the project root (/opt/airflow/data/).")

#     df = pd.read_csv(file_path)
#     X = df.drop(columns=[target_col])
#     y = df[target_col]
#     return X, y # Typically, load_data should return X and y, not the full df and y



# Loading data from postgres

# model_training/data_loader.py

import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

def load_data(target_col='Converted'):
    """
    Step A: Load raw data from Postgres instead of a CSV file.
    
    Returns:
        X (pd.DataFrame): features
        y (pd.Series): target column
    """
    # ─── Step 1: Build the DB connection string ─────────────────────────────
    # We expect an environment variable POSTGRES_CONN of the form:
    #   postgresql://<user>:<pass>@<host>:<port>/<db>
    # In your Docker‑Compose you already set:
    #   AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/lead_db
    # We can reuse that:
    conn_uri = os.getenv(
        "POSTGRES_CONN",
        os.getenv("AIRFLOW__CORE__SQL_ALCHEMY_CONN").replace("postgresql+", "")
    )
    
    # ─── Step 2: Connect and query ──────────────────────────────────────────
    # Using psycopg2 + RealDictCursor so pandas can consume it directly
    conn = psycopg2.connect(conn_uri, cursor_factory=RealDictCursor)
    
    # Adjust the table name and schema as needed:
    sql = "SELECT * FROM lead_scoring_table;"  # ← your actual table name
    
    # Read into a DataFrame
    df = pd.read_sql(sql, conn)
    
    # Close the connection
    conn.close()
    
    # ─── Step 3: Split into X and y ─────────────────────────────────────────
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DB table columns: {df.columns.tolist()}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y
