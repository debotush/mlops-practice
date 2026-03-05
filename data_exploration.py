import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_parquet('green_tripdata_2021-01.parquet')

# Basic information
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

# Save this info to a file
with open('data_summary.txt', 'w') as f:
    f.write(f"Dataset Shape: {df.shape}\n")
    f.write(f"\nColumns: {df.columns.tolist()}\n")
    f.write(f"\nMissing Values:\n{df.isnull().sum()}\n")
