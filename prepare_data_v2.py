"""
MLOps Practice 2 - Data Preparation Script
This script combines January and February data and creates new train/test splits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data(data_path):
    """Load parquet data file"""
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} records")
    return df

def preprocess_data(df):
    """
    Preprocess the NYC taxi data
    """
    print("Preprocessing data...")
    
    # Create a copy
    df = df.copy()
    
    # Convert datetime columns
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    
    # Calculate trip duration in minutes
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Filter out invalid trips
    df = df[(df['trip_duration'] > 0) & (df['trip_duration'] < 120)]
    
    # Extract time features
    df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['lpep_pickup_datetime'].dt.dayofweek
    df['pickup_month'] = df['lpep_pickup_datetime'].dt.month
    
    # Select features
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
    
    # Filter for data quality
    df = df[df['trip_distance'] > 0]
    df = df[df['passenger_count'] > 0]
    df = df[df['fare_amount'] > 0]
    
    print(f"After preprocessing: {len(df)} records")
    
    return df, feature_columns

def combine_datasets(jan_path, feb_path):
    """Combine January and February datasets"""
    print("\n=== Combining Datasets ===")
    
    # Load both datasets
    df_jan = load_data(jan_path)
    df_feb = load_data(feb_path)
    
    print(f"\nJanuary data: {len(df_jan)} records")
    print(f"February data: {len(df_feb)} records")
    
    # Preprocess each
    df_jan_processed, feature_columns = preprocess_data(df_jan)
    df_feb_processed, _ = preprocess_data(df_feb)
    
    # Combine
    df_combined = pd.concat([df_jan_processed, df_feb_processed], ignore_index=True)
    print(f"\nCombined dataset: {len(df_combined)} records")
    
    return df_combined, feature_columns

def split_and_save_data(df, feature_columns, output_dir, test_size=0.2, random_state=42):
    """Split data and save to CSV"""
    print(f"\nSplitting data (test_size={test_size})...")
    
    X = df[feature_columns]
    y = df['trip_duration']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train data
    train_df = X_train.copy()
    train_df['trip_duration'] = y_train
    train_path = os.path.join(output_dir, 'train.csv')
    train_df.to_csv(train_path, index=False)
    print(f"Training data saved to {train_path}")
    
    # Save test data
    test_df = X_test.copy()
    test_df['trip_duration'] = y_test
    test_path = os.path.join(output_dir, 'test.csv')
    test_df.to_csv(test_path, index=False)
    print(f"Test data saved to {test_path}")
    
    return X_train, X_test, y_train, y_test

def analyze_data_distributions(df_jan, df_feb, df_combined):
    """Analyze and compare data distributions"""
    print("\n=== Data Distribution Analysis ===")
    
    # Compare basic statistics
    print("\nTrip Duration Statistics (minutes):")
    print(f"{'':20} {'January':>12} {'February':>12} {'Combined':>12}")
    print("-" * 60)
    
    for stat_name, stat_func in [
        ('Mean', lambda x: x['trip_duration'].mean()),
        ('Median', lambda x: x['trip_duration'].median()),
        ('Std Dev', lambda x: x['trip_duration'].std()),
        ('Min', lambda x: x['trip_duration'].min()),
        ('Max', lambda x: x['trip_duration'].max())
    ]:
        jan_val = stat_func(df_jan)
        feb_val = stat_func(df_feb)
        comb_val = stat_func(df_combined)
        print(f"{stat_name:20} {jan_val:12.2f} {feb_val:12.2f} {comb_val:12.2f}")
    
    print("\nTrip Distance Statistics (miles):")
    print(f"{'':20} {'January':>12} {'February':>12} {'Combined':>12}")
    print("-" * 60)
    
    for stat_name, stat_func in [
        ('Mean', lambda x: x['trip_distance'].mean()),
        ('Median', lambda x: x['trip_distance'].median()),
        ('Std Dev', lambda x: x['trip_distance'].std())
    ]:
        jan_val = stat_func(df_jan)
        feb_val = stat_func(df_feb)
        comb_val = stat_func(df_combined)
        print(f"{stat_name:20} {jan_val:12.2f} {feb_val:12.2f} {comb_val:12.2f}")

if __name__ == "__main__":
    # Configuration
    JAN_DATA_PATH = "data/raw/green_tripdata_2021-01.parquet"
    FEB_DATA_PATH = "data/raw/green_tripdata_2021-02.parquet"
    OUTPUT_DIR = "data/processed"
    
    # Combine datasets
    df_combined, feature_columns = combine_datasets(JAN_DATA_PATH, FEB_DATA_PATH)
    
    # Load original data for analysis
    df_jan = load_data(JAN_DATA_PATH)
    df_feb = load_data(FEB_DATA_PATH)
    df_jan_proc, _ = preprocess_data(df_jan)
    df_feb_proc, _ = preprocess_data(df_feb)
    
    # Analyze distributions
    analyze_data_distributions(df_jan_proc, df_feb_proc, df_combined)
    
    # Split and save
    X_train, X_test, y_train, y_test = split_and_save_data(
        df_combined, feature_columns, OUTPUT_DIR
    )
    
    print("\n=== Data preparation completed successfully ===")
