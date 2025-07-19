import pandas as pd

# Load full dataset
df = pd.read_csv("./data/Lead_Scoring.csv")

# Take a small random sample of 50 rows
sample_df = df.sample(n=20, random_state=42)

# # Save to a new file
# sample_df.to_csv("./data/sample_dataset.csv", index=False)

# print("Sample saved to 'sample_dataset.csv'")


print(sample_df.dtypes)