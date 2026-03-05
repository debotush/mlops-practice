import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_parquet('green_tripdata_2021-01.parquet')

# Create plots directory
import os
os.makedirs('plots', exist_ok=True)

# Clean data - remove missing values and outliers
df_clean = df.dropna(subset=['passenger_count', 'trip_distance', 'fare_amount', 'total_amount'])
df_clean = df_clean[(df_clean['trip_distance'] > 0) & (df_clean['trip_distance'] < 100)]
df_clean = df_clean[(df_clean['fare_amount'] > 0) & (df_clean['fare_amount'] < 200)]
df_clean = df_clean[(df_clean['passenger_count'] > 0) & (df_clean['passenger_count'] <= 6)]

print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_clean.shape}")

# Plot 1: Trip Distance Distribution
plt.figure(figsize=(10, 6))
plt.hist(df_clean['trip_distance'], bins=50, edgecolor='black')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Frequency')
plt.title('Distribution of Trip Distances')
plt.savefig('plots/trip_distance_distribution.png')
plt.close()

# Plot 2: Fare Amount vs Trip Distance
plt.figure(figsize=(10, 6))
plt.scatter(df_clean['trip_distance'], df_clean['fare_amount'], alpha=0.3)
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Fare Amount ($)')
plt.title('Fare Amount vs Trip Distance')
plt.savefig('plots/fare_vs_distance.png')
plt.close()

# Plot 3: Correlation heatmap
numeric_cols = ['trip_distance', 'fare_amount', 'passenger_count', 'total_amount', 'tip_amount']
plt.figure(figsize=(10, 8))
sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# Plot 4: Passenger count distribution
plt.figure(figsize=(10, 6))
df_clean['passenger_count'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Number of Passengers')
plt.ylabel('Frequency')
plt.title('Distribution of Passenger Count')
plt.savefig('plots/passenger_count_distribution.png')
plt.close()

print("\nPlots saved in 'plots/' directory!")
print(f"\nCorrelation with fare_amount:")
print(df_clean[numeric_cols].corr()['fare_amount'].sort_values(ascending=False))
