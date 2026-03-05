import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle

# Load and clean data
df = pd.read_parquet('green_tripdata_2021-01.parquet')
df_clean = df.dropna(subset=['passenger_count', 'trip_distance', 'fare_amount', 'payment_type', 'tip_amount'])
df_clean = df_clean[(df_clean['trip_distance'] > 0) & (df_clean['trip_distance'] < 100)]
df_clean = df_clean[(df_clean['fare_amount'] > 0) & (df_clean['fare_amount'] < 200)]
df_clean = df_clean[(df_clean['passenger_count'] > 0) & (df_clean['passenger_count'] <= 6)]

# Filter payment types
df_clean = df_clean[df_clean['payment_type'].isin([1.0, 2.0])]
df_clean['payment_type'] = df_clean['payment_type'].astype(int)

# Calculate trip duration as new feature
df_clean['trip_duration_minutes'] = (
    df_clean['lpep_dropoff_datetime'] - df_clean['lpep_pickup_datetime']
).dt.total_seconds() / 60

df_clean = df_clean[(df_clean['trip_duration_minutes'] > 0) & (df_clean['trip_duration_minutes'] < 120)]

print(f"Cleaned dataset shape: {df_clean.shape}")

# OLD MODEL: 3 features
features_old = ['trip_distance', 'fare_amount', 'tip_amount']
# NEW MODEL: 4 features (added trip_duration_minutes)
features_new = ['trip_distance', 'fare_amount', 'tip_amount', 'trip_duration_minutes']
target = 'payment_type'

print("\n" + "="*60)
print("COMPARING OLD (3 features) vs NEW (4 features) MODEL")
print("="*60)

# Function to evaluate model multiple times
def evaluate_model(X, y, features_name, n_runs=5):
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + run, stratify=y
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results['accuracy'].append(accuracy)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
    
    return results

# Evaluate OLD model
print("\n--- OLD MODEL (3 features) ---")
X_old = df_clean[features_old]
y = df_clean[target]
results_old = evaluate_model(X_old, y, "Old Model", n_runs=5)

print(f"Accuracy:  {np.mean(results_old['accuracy']):.4f} ± {np.std(results_old['accuracy']):.4f}")
print(f"Precision: {np.mean(results_old['precision']):.4f} ± {np.std(results_old['precision']):.4f}")
print(f"Recall:    {np.mean(results_old['recall']):.4f} ± {np.std(results_old['recall']):.4f}")
print(f"F1-Score:  {np.mean(results_old['f1']):.4f} ± {np.std(results_old['f1']):.4f}")

# Evaluate NEW model
print("\n--- NEW MODEL (4 features) ---")
X_new = df_clean[features_new]
results_new = evaluate_model(X_new, y, "New Model", n_runs=5)

print(f"Accuracy:  {np.mean(results_new['accuracy']):.4f} ± {np.std(results_new['accuracy']):.4f}")
print(f"Precision: {np.mean(results_new['precision']):.4f} ± {np.std(results_new['precision']):.4f}")
print(f"Recall:    {np.mean(results_new['recall']):.4f} ± {np.std(results_new['recall']):.4f}")
print(f"F1-Score:  {np.mean(results_new['f1']):.4f} ± {np.std(results_new['f1']):.4f}")

# Calculate improvement
accuracy_improvement = np.mean(results_new['accuracy']) - np.mean(results_old['accuracy'])
f1_improvement = np.mean(results_new['f1']) - np.mean(results_old['f1'])

print("\n" + "="*60)
print("IMPROVEMENT ANALYSIS")
print("="*60)
print(f"Accuracy Improvement: {accuracy_improvement:.4f} ({accuracy_improvement*100:.2f}%)")
print(f"F1-Score Improvement: {f1_improvement:.4f}")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_ind(results_old['accuracy'], results_new['accuracy'])
print(f"\nStatistical Significance Test (t-test):")
print(f"  p-value: {p_value:.6f}")
if p_value < 0.05:
    print(f"  ✓ Improvement is STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print(f"  ✗ Improvement is NOT statistically significant (p >= 0.05)")

# Train and save final model
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42, stratify=y)
final_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
final_model.fit(X_train, y_train)

with open('random_forest_classifier_improved.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("\nImproved model saved as 'random_forest_classifier_improved.pkl'")

# Save report
import os
os.makedirs('results', exist_ok=True)

with open('results/classification_improvement_report.txt', 'w') as f:
    f.write("CLASSIFICATION MODEL IMPROVEMENT ANALYSIS\n")
    f.write("="*60 + "\n\n")
    f.write("NEW FEATURE ADDED: trip_duration_minutes\n\n")
    
    f.write("OLD MODEL (3 features):\n")
    f.write(f"  Features: {features_old}\n")
    f.write(f"  Accuracy: {np.mean(results_old['accuracy']):.4f} ± {np.std(results_old['accuracy']):.4f}\n")
    f.write(f"  F1-Score: {np.mean(results_old['f1']):.4f} ± {np.std(results_old['f1']):.4f}\n\n")
    
    f.write("NEW MODEL (4 features):\n")
    f.write(f"  Features: {features_new}\n")
    f.write(f"  Accuracy: {np.mean(results_new['accuracy']):.4f} ± {np.std(results_new['accuracy']):.4f}\n")
    f.write(f"  F1-Score: {np.mean(results_new['f1']):.4f} ± {np.std(results_new['f1']):.4f}\n\n")
    
    f.write("="*60 + "\n")
    f.write("SAME RISKS AS REGRESSION MODEL:\n")
    f.write("="*60 + "\n")
    f.write("See regression_improvement_report.txt for detailed analysis\n")
    f.write("of risks related to trip_duration_minutes feature.\n")

print("\nReport saved in 'results/classification_improvement_report.txt'")
