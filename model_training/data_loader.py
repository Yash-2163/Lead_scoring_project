import pandas as pd

def load_data(file_path='../data/Lead_Scoring.csv', target_col='Converted'):
    """
    Loads raw data and splits into features and target.
    No preprocessing here.
    """
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
