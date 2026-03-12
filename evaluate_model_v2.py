"""
MLOps Practice 2 - Evaluation Script for Version 2
This script evaluates the existing model on new data
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import os

def load_model(model_path):
    """Load trained model"""
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_data(data_path):
    """Load test data"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    return df

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("Evaluating model on new data...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred)
    }
    
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    
    # Calculate error statistics
    errors = y_test - y_pred
    error_stats = {
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'median_error': float(np.median(errors)),
        'percentile_95_error': float(np.percentile(np.abs(errors), 95))
    }
    
    return metrics, error_stats, y_pred

def compare_metrics(old_metrics_path, new_metrics):
    """Compare new metrics with previous version"""
    print("\n=== Metrics Comparison ===")
    
    if os.path.exists(old_metrics_path):
        with open(old_metrics_path, 'r') as f:
            old_data = json.load(f)
            old_metrics = old_data['metrics']
        
        print(f"Version {old_data.get('version', 'unknown')} vs New Data:")
        print(f"RMSE: {old_metrics['rmse']:.2f} -> {new_metrics['rmse']:.2f} "
              f"(change: {((new_metrics['rmse']/old_metrics['rmse']-1)*100):+.2f}%)")
        print(f"MAE: {old_metrics['mae']:.2f} -> {new_metrics['mae']:.2f} "
              f"(change: {((new_metrics['mae']/old_metrics['mae']-1)*100):+.2f}%)")
        print(f"R²: {old_metrics['r2_score']:.4f} -> {new_metrics['r2_score']:.4f} "
              f"(change: {((new_metrics['r2_score']/old_metrics['r2_score']-1)*100):+.2f}%)")
    else:
        print("No previous metrics found for comparison")

def save_metrics(metrics, error_stats, metrics_path, version, data_info):
    """Save evaluation metrics"""
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    metrics_data = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'error_statistics': error_stats,
        'data_info': data_info
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

def generate_evaluation_report(metrics, error_stats, old_metrics_path):
    """Generate textual evaluation report"""
    report = []
    report.append("="*60)
    report.append("MODEL EVALUATION REPORT - VERSION 2")
    report.append("="*60)
    report.append("")
    report.append("PERFORMANCE METRICS:")
    report.append(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.2f} minutes")
    report.append(f"  MAE (Mean Absolute Error): {metrics['mae']:.2f} minutes")
    report.append(f"  R² Score: {metrics['r2_score']:.4f}")
    report.append("")
    report.append("ERROR STATISTICS:")
    report.append(f"  Mean Error: {error_stats['mean_error']:.2f} minutes")
    report.append(f"  Std Error: {error_stats['std_error']:.2f} minutes")
    report.append(f"  Median Error: {error_stats['median_error']:.2f} minutes")
    report.append(f"  95th Percentile Error: {error_stats['percentile_95_error']:.2f} minutes")
    report.append("")
    
    # Load old metrics for comparison
    if os.path.exists(old_metrics_path):
        with open(old_metrics_path, 'r') as f:
            old_data = json.load(f)
            old_metrics = old_data['metrics']
        
        report.append("COMPARISON WITH PREVIOUS VERSION:")
        rmse_change = ((metrics['rmse']/old_metrics['rmse']-1)*100)
        mae_change = ((metrics['mae']/old_metrics['mae']-1)*100)
        r2_change = ((metrics['r2_score']/old_metrics['r2_score']-1)*100)
        
        report.append(f"  RMSE Change: {rmse_change:+.2f}%")
        report.append(f"  MAE Change: {mae_change:+.2f}%")
        report.append(f"  R² Change: {r2_change:+.2f}%")
        report.append("")
        
        # Analysis
        report.append("ANALYSIS:")
        if abs(rmse_change) > 10 or abs(mae_change) > 10:
            report.append("  ⚠ SIGNIFICANT PERFORMANCE DEGRADATION DETECTED!")
            report.append("  The model's error metrics have increased by more than 10%.")
            report.append("  This indicates potential data drift or distribution shift.")
            report.append("")
            report.append("  POSSIBLE CAUSES:")
            report.append("  - Temporal drift: February data has different patterns than January")
            report.append("  - Seasonal effects: Weather, holidays, or behavioral changes")
            report.append("  - Data quality issues in the new dataset")
            report.append("")
            report.append("  RECOMMENDED ACTIONS:")
            report.append("  1. Investigate feature distributions in new data")
            report.append("  2. Check for missing or anomalous values")
            report.append("  3. Consider retraining with combined data")
            report.append("  4. Implement monitoring for continued drift")
        elif abs(rmse_change) > 5 or abs(mae_change) > 5:
            report.append("  ⚡ MODERATE PERFORMANCE CHANGE DETECTED")
            report.append("  The model shows some degradation on new data.")
            report.append("  This is expected with temporal data drift.")
            report.append("  Consider retraining if degradation continues.")
        else:
            report.append("  ✓ Model performance is stable on new data")
            report.append("  Minimal degradation detected, within acceptable limits.")
    
    report.append("="*60)
    
    return "\n".join(report)

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/model.pkl"
    TEST_DATA_PATH = "data/processed/test.csv"
    OLD_METRICS_PATH = "models/metrics.json"
    NEW_METRICS_PATH = "models/metrics_v2.json"
    REPORT_PATH = "reports/evaluation_v2.txt"
    VERSION = "v2"
    
    # Load model and data
    model = load_model(MODEL_PATH)
    test_df = load_data(TEST_DATA_PATH)
    
    # Separate features and target
    feature_columns = [col for col in test_df.columns if col != 'trip_duration']
    X_test = test_df[feature_columns]
    y_test = test_df['trip_duration']
    
    # Evaluate
    metrics, error_stats, y_pred = evaluate_model(model, X_test, y_test)
    
    # Compare with old metrics
    compare_metrics(OLD_METRICS_PATH, metrics)
    
    # Save new metrics
    data_info = {
        'n_samples': len(X_test),
        'features': feature_columns
    }
    save_metrics(metrics, error_stats, NEW_METRICS_PATH, VERSION, data_info)
    
    # Generate and save report
    report = generate_evaluation_report(metrics, error_stats, OLD_METRICS_PATH)
    print("\n" + report)
    
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_PATH}")
