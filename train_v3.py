import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Load both datasets
df_jan = pd.read_parquet('data/green_tripdata_2021-01.parquet')
df_feb = pd.read_parquet('data/green_tripdata_2021-02.parquet')

# Combine
df = pd.concat([df_jan, df_feb], ignore_index=True)

# Clean
df_clean = df.dropna(subset=['passenger_count', 'trip_distance', 'fare_amount', 'total_amount', 'PULocationID'])
df_clean = df_clean[(df_clean['trip_distance'] > 0) & (df_clean['trip_distance'] < 100)]
df_clean = df_clean[(df_clean['fare_amount'] > 0) & (df_clean['fare_amount'] < 200)]
df_clean = df_clean[(df_clean['passenger_count'] > 0) & (df_clean['passenger_count'] <= 6)]

print(f"Combined dataset shape: {df_clean.shape}")

features = ['trip_distance', 'passenger_count', 'PULocationID']
target = 'fare_amount'

X = df_clean[features]
y = df_clean[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train new model
print("\nTraining Linear Regression Model on combined data...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nV3 Model Performance:")
print(f"RMSE: ${rmse:.2f}")
print(f"MAE: ${mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Save new model
with open('data/linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved as 'data/linear_regression_model.pkl'")

# Save results
os.makedirs('results', exist_ok=True)
with open('results/train_v3_results.txt', 'w') as f:
    f.write("V3 TRAINING RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write("Model: Linear Regression retrained on Jan + Feb 2021\n\n")
    f.write(f"Combined dataset shape: {df_clean.shape}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"  RMSE: ${rmse:.2f}\n")
    f.write(f"  MAE: ${mae:.2f}\n")
    f.write(f"  R² Score: {r2:.4f}\n")

print("Results saved to results/train_v3_results.txt")
