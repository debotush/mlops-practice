import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
df = pd.read_parquet('green_tripdata_2021-01.parquet')
df_clean = df.dropna(subset=['passenger_count', 'trip_distance', 'fare_amount', 'total_amount', 'PULocationID'])
df_clean = df_clean[(df_clean['trip_distance'] > 0) & (df_clean['trip_distance'] < 100)]
df_clean = df_clean[(df_clean['fare_amount'] > 0) & (df_clean['fare_amount'] < 200)]
df_clean = df_clean[(df_clean['passenger_count'] > 0) & (df_clean['passenger_count'] <= 6)]

print(f"Cleaned dataset shape: {df_clean.shape}")

# Select features for regression (predicting fare_amount)
# Using 3 features: trip_distance, passenger_count, PULocationID
features = ['trip_distance', 'passenger_count', 'PULocationID']
target = 'fare_amount'

X = df_clean[features]
y = df_clean[target]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train Linear Regression model
print("\n" + "="*50)
print("Training Linear Regression Model...")
print("="*50)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluation metrics
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"\nLinear Regression Performance:")
print(f"RMSE: ${rmse_lr:.2f}")
print(f"MAE: ${mae_lr:.2f}")
print(f"R² Score: {r2_lr:.4f}")

# Save model
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("\nModel saved as 'linear_regression_model.pkl'")

# Create results directory
import os
os.makedirs('results', exist_ok=True)

# Plot: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Fare Amount ($)')
plt.ylabel('Predicted Fare Amount ($)')
plt.title('Linear Regression: Actual vs Predicted Fare Amount')
plt.savefig('results/regression_actual_vs_predicted.png')
plt.close()

# Plot: Residuals
residuals = y_test - y_pred_lr
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lr, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Fare Amount ($)')
plt.ylabel('Residuals ($)')
plt.title('Residual Plot')
plt.savefig('results/regression_residuals.png')
plt.close()

# Save results to file
with open('results/regression_results.txt', 'w') as f:
    f.write("REGRESSION MODEL RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Model: Linear Regression\n")
    f.write(f"Features used: {features}\n")
    f.write(f"Target: {target}\n\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"  RMSE: ${rmse_lr:.2f}\n")
    f.write(f"  MAE: ${mae_lr:.2f}\n")
    f.write(f"  R² Score: {r2_lr:.4f}\n\n")
    f.write("Feature Coefficients:\n")
    for feature, coef in zip(features, lr_model.coef_):
        f.write(f"  {feature}: {coef:.4f}\n")
    f.write(f"  Intercept: {lr_model.intercept_:.4f}\n")

print("\nResults saved in 'results/' directory!")
