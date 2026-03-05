import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
df = pd.read_parquet('green_tripdata_2021-01.parquet')
df_clean = df.dropna(subset=['passenger_count', 'trip_distance', 'fare_amount', 'payment_type', 'tip_amount'])
df_clean = df_clean[(df_clean['trip_distance'] > 0) & (df_clean['trip_distance'] < 100)]
df_clean = df_clean[(df_clean['fare_amount'] > 0) & (df_clean['fare_amount'] < 200)]
df_clean = df_clean[(df_clean['passenger_count'] > 0) & (df_clean['passenger_count'] <= 6)]

# Filter payment types (1 = Credit Card, 2 = Cash)
df_clean = df_clean[df_clean['payment_type'].isin([1.0, 2.0])]
df_clean['payment_type'] = df_clean['payment_type'].astype(int)

print(f"Cleaned dataset shape: {df_clean.shape}")
print(f"\nPayment type distribution:")
print(df_clean['payment_type'].value_counts())
print(f"\n1 = Credit Card, 2 = Cash")

# Select features for classification (predicting payment_type)
# Using 3 features: trip_distance, fare_amount, tip_amount
features = ['trip_distance', 'fare_amount', 'tip_amount']
target = 'payment_type'

X = df_clean[features]
y = df_clean[target]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train Random Forest Classifier
print("\n" + "="*50)
print("Training Random Forest Classifier...")
print("="*50)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf, average='weighted')
recall = recall_score(y_test, y_pred_rf, average='weighted')
f1 = f1_score(y_test, y_pred_rf, average='weighted')

print(f"\nRandom Forest Classifier Performance:")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\n" + "="*50)
print("Classification Report:")
print("="*50)
print(classification_report(y_test, y_pred_rf, target_names=['Credit Card', 'Cash']))

# Save model
with open('random_forest_classifier.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("\nModel saved as 'random_forest_classifier.pkl'")

# Create results directory
import os
os.makedirs('results', exist_ok=True)

# Plot: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Credit Card', 'Cash'], yticklabels=['Credit Card', 'Cash'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Payment Type Classification')
plt.savefig('results/classification_confusion_matrix.png')
plt.close()

# Plot: Feature Importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance for Payment Type Classification')
plt.gca().invert_yaxis()
plt.savefig('results/classification_feature_importance.png')
plt.close()

# Save results to file
with open('results/classification_results.txt', 'w') as f:
    f.write("CLASSIFICATION MODEL RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Model: Random Forest Classifier\n")
    f.write(f"Features used: {features}\n")
    f.write(f"Target: {target} (1=Credit Card, 2=Cash)\n\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall: {recall:.4f}\n")
    f.write(f"  F1-Score: {f1:.4f}\n\n")
    f.write("Feature Importance:\n")
    for idx, row in feature_importance.iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
    f.write("\n" + "="*50 + "\n")
    f.write("Classification Report:\n")
    f.write("="*50 + "\n")
    f.write(classification_report(y_test, y_pred_rf, target_names=['Credit Card', 'Cash']))

print("\nResults saved in 'results/' directory!")
