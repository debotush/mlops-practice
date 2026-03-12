"""
MLOps Practice 2 - Model Training Script
This script handles data loading, preprocessing, model training, and evaluation
for NYC Green Taxi trip duration prediction (regression task)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import json
from datetime import datetime

def load_data(data_path):
    """Load parquet data file"""
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} records")
    return df

def preprocess_data(df):
    """
    Preprocess the NYC taxi data
    - Create target variable (trip_duration in minutes)
    - Engineer features
    - Handle missing values
    """
    print("Preprocessing data...")
    
    # Create a copy to avoid warnings
    df = df.copy()
    
    # Convert datetime columns
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    
    # Calculate trip duration in minutes (target variable)
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Filter out invalid trips (negative or extremely long durations)
    df = df[(df['trip_duration'] > 0) & (df['trip_duration'] < 120)]  # 0-120 minutes
    
    # Extract time-based features
    df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['lpep_pickup_datetime'].dt.dayofweek
    df['pickup_month'] = df['lpep_pickup_datetime'].dt.month
    
    # Select features for modeling
    feature_columns = [
        'passenger_count',
        'trip_distance',
        'PULocationID',
        'DOLocationID',
        'fare_amount',
        'pickup_hour',
        'pickup_day',
        'pickup_month'
    ]
    
    # Handle missing values
    df = df.dropna(subset=feature_columns + ['trip_duration'])
    
    # Additional filtering for data quality
    df = df[df['trip_distance'] > 0]
    df = df[df['passenger_count'] > 0]
    df = df[df['fare_amount'] > 0]
    
    print(f"After preprocessing: {len(df)} records")
    
    return df, feature_columns

def split_data(df, feature_columns, target_col='trip_duration', test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X = df[feature_columns]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type='random_forest'):
    """Train the model"""
    print(f"Training {model_type} model...")
    
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    print("Model training completed")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred)
    }
    
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    
    return metrics, y_pred

def save_model(model, model_path):
    """Save trained model"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def save_metrics(metrics, metrics_path, version, additional_info=None):
    """Save evaluation metrics"""
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    metrics_data = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    if additional_info:
        metrics_data.update(additional_info)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

def save_data(X_train, X_test, y_train, y_test, data_dir):
    """Save processed train/test data"""
    os.makedirs(data_dir, exist_ok=True)
    
    train_df = X_train.copy()
    train_df['trip_duration'] = y_train
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    
    test_df = X_test.copy()
    test_df['trip_duration'] = y_test
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    
    print(f"Processed data saved to {data_dir}")

if __name__ == "__main__":
    # Configuration
    RAW_DATA_PATH = "data/raw/green_tripdata_2021-01.parquet"
    MODEL_PATH = "models/model.pkl"
    METRICS_PATH = "models/metrics.json"
    PROCESSED_DATA_DIR = "data/processed"
    VERSION = "v1"
    
    # Load and preprocess data
    df = load_data(RAW_DATA_PATH)
    df, feature_columns = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, feature_columns)
    
    # Save processed data
    save_data(X_train, X_test, y_train, y_test, PROCESSED_DATA_DIR)
    
    # Train model
    model = train_model(X_train, y_train, model_type='random_forest')
    
    # Evaluate model
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    # Save model and metrics
    save_model(model, MODEL_PATH)
    save_metrics(metrics, METRICS_PATH, VERSION, {
        'data_source': RAW_DATA_PATH,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'features': feature_columns
    })
    
    print("\n=== Training pipeline completed successfully ===")
