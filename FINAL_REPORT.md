# MLOps Fundamentals - Practice Session Report

**Student Name:** Debotush Dam
**Date:** March 5, 2026  
**GitHub Repository:** https://github.com/debotush/mlops-practice

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Data Analysis](#data-analysis)
4. [Model 1: Regression Model](#model-1-regression-model)
5. [Model 2: Classification Model](#model-2-classification-model)
6. [Feature Addition Analysis](#feature-addition-analysis)
7. [Conclusion](#conclusion)
8. [Repository Structure](#repository-structure)

---

## 1. Project Overview

This project implements two machine learning models on the NYC Green Taxi dataset (January 2021):
- **Regression Model**: Predicts taxi fare amount
- **Classification Model**: Predicts payment type (Cash vs Credit Card)

Both models were trained with 3 features initially, then improved by adding a 4th feature with statistical validation.

---

## 2. Dataset Description

**Dataset:** NYC Green Taxi Trip Data (January 2021)
- **Source:** green_tripdata_2021-01.parquet
- **Original Size:** 76,518 rows × 20 columns
- **After Cleaning:** 38,191 rows (removed missing values and outliers)

**Key Features:**
- `trip_distance`: Distance traveled in miles
- `fare_amount`: Base fare amount in USD
- `passenger_count`: Number of passengers
- `tip_amount`: Tip amount in USD
- `payment_type`: 1 = Credit Card, 2 = Cash
- `PULocationID`: Pickup location ID
- `lpep_pickup_datetime`: Pickup timestamp
- `lpep_dropoff_datetime`: Dropoff timestamp

---

## 3. Data Analysis

### 3.1 Data Cleaning
- Removed rows with missing values in key columns
- Filtered unrealistic values:
  - Trip distance: 0 < distance < 100 miles
  - Fare amount: 0 < fare < $200
  - Passenger count: 0 < passengers ≤ 6

### 3.2 Exploratory Data Analysis

**Correlation Analysis:**
```
fare_amount      1.000000
total_amount     0.983898  (very high - expected)
trip_distance    0.944537  (strong positive correlation)
tip_amount       0.204210  (weak correlation)
passenger_count -0.013816  (no correlation)
```

**Key Insights:**
- Strong linear relationship between trip distance and fare
- Tip amount shows weak correlation with fare (cash payers often don't tip)
- Passenger count has minimal impact on fare

**Visualizations Created:**
- Trip distance distribution
- Fare vs Distance scatter plot
- Correlation heatmap
- Passenger count distribution

All plots saved in `plots/` directory.

---

## 4. Model 1: Regression Model

### 4.1 Objective
Predict `fare_amount` based on trip characteristics.

### 4.2 Initial Model (3 Features)

**Features Selected:**
1. `trip_distance` - Primary predictor (correlation: 0.94)
2. `passenger_count` - Slight impact on fare
3. `PULocationID` - Location-based pricing variation

**Algorithm:** Linear Regression
- Simple, interpretable model
- Works well for linear relationships
- Fast training and prediction

**Model Performance:**
```
RMSE: $3.98
MAE:  $1.85
R² Score: 0.9149 (91.49%)
```

**Interpretation:**
- Model explains 91.49% of fare variance
- Average prediction error: ~$4
- Strong performance for initial 3-feature model

**Model Formula:**
```
fare = (2.80 × distance) + (0.50 × passengers) + (0.02 × location) + 5.20
```

### 4.3 Evaluation Metrics Explained

**RMSE (Root Mean Squared Error):**
- Measures average prediction error in dollars
- Penalizes large errors more heavily
- Lower is better

**MAE (Mean Absolute Error):**
- Average absolute difference between predicted and actual
- More intuitive than RMSE
- Less sensitive to outliers

**R² Score (Coefficient of Determination):**
- Proportion of variance explained by the model
- Range: 0 to 1 (1 = perfect fit)
- 0.91 means 91% of fare variation is explained

### 4.4 Model Files
- **Code:** `regression_model.py`
- **Saved Model:** `linear_regression_model.pkl`
- **Results:** `results/regression_results.txt`
- **Visualizations:** `results/regression_actual_vs_predicted.png`

---

## 5. Model 2: Classification Model

### 5.1 Objective
Predict `payment_type` (1 = Credit Card, 2 = Cash).

### 5.2 Initial Model (3 Features)

**Features Selected:**
1. `trip_distance` - Longer trips may prefer cards
2. `fare_amount` - Higher fares may use cards
3. `tip_amount` - **CRITICAL FEATURE**: Cash payers rarely tip

**Algorithm:** Random Forest Classifier
- Ensemble of 100 decision trees
- Handles non-linear relationships
- Provides feature importance scores
- Robust to outliers

**Model Performance:**
```
Accuracy:  91.99%
Precision: 0.93
Recall:    0.92
F1-Score:  0.92
```

**Per-Class Performance:**
```
Credit Card:
  Precision: 99% (rarely misclassifies as Credit Card)
  Recall: 88% (catches 88% of Credit Card payments)

Cash:
  Precision: 84%
  Recall: 98% (excellent at detecting Cash payments)
```

### 5.3 Evaluation Metrics Explained

**Accuracy:**
- Percentage of correct predictions
- 91.99% = correctly predicted 7,000 out of 7,605 payments

**Precision:**
- "When I predict Credit Card, how often am I right?"
- High precision = few false alarms

**Recall:**
- "Out of all actual Credit Cards, how many did I catch?"
- High recall = don't miss many

**F1-Score:**
- Harmonic mean of precision and recall
- Balances both metrics

**Confusion Matrix:**
```
                Predicted
              Credit  Cash
Actual Credit  4046    549
       Cash     61    2949
```

### 5.4 Feature Importance
```
tip_amount:      65%  (Most important!)
fare_amount:     25%
trip_distance:   10%
```

**Why tip_amount dominates:**
- Cash payers often don't tip (or tips aren't recorded)
- Credit card payers usually tip
- Strong signal for payment method

### 5.5 Model Files
- **Code:** `classification_model.py`
- **Saved Model:** `random_forest_classifier.pkl`
- **Results:** `results/classification_results.txt`
- **Confusion Matrix:** `results/classification_confusion_matrix.png`

---

## 6. Feature Addition Analysis

### 6.1 New Feature: trip_duration_minutes

**Definition:**
```python
trip_duration_minutes = (dropoff_datetime - pickup_datetime) / 60
```

**Rationale:**
- Longer trips may have higher fares (traffic, time-based pricing)
- Could improve regression predictions
- May help classify payment types (longer trips → prefer cards?)

### 6.2 Regression Model Improvement

**OLD MODEL (3 features):**
```
Features: [trip_distance, passenger_count, PULocationID]
RMSE: $3.83 ± $0.17
R²: 0.9198 ± 0.0064
```

**NEW MODEL (4 features):**
```
Features: [trip_distance, passenger_count, PULocationID, trip_duration_minutes]
RMSE: $2.85 ± $0.22
R²: 0.9556 ± 0.0067
```

**Improvement:**
- ✅ RMSE improved by $0.99 (25.8% reduction in error!)
- ✅ R² improved by 0.0358 (3.58 percentage points)
- ✅ **p-value: 0.000093** (HIGHLY SIGNIFICANT)

**Statistical Validation:**
- Ran model 5 times with different random splits
- Consistently better performance across all runs
- p-value << 0.05 proves improvement is real, not random

### 6.3 Classification Model Improvement

**OLD MODEL (3 features):**
```
Features: [trip_distance, fare_amount, tip_amount]
Accuracy: 91.74% ± 0.26%
```

**NEW MODEL (4 features):**
```
Features: [trip_distance, fare_amount, tip_amount, trip_duration_minutes]
Accuracy: 91.82% ± 0.27%
```

**Improvement:**
- ❌ Only 0.08% improvement (negligible)
- ❌ **p-value: 0.69** (NOT SIGNIFICANT)
- Trip duration doesn't help predict payment type

**Why No Improvement?**
- Payment type is determined by customer preference, not trip characteristics
- `tip_amount` already captures the signal (cash = no tip)
- Duration adds no new information for this task

---

## 7. Addressing Assignment Questions

### Q1: How can you ensure that the improvement is not due to randomness or data leakage?

**Answer:**

**A. Cross-Validation (Testing for Randomness)**
1. **Multiple Runs:** Trained each model 5 times with different data splits
   - Old model RMSE: $3.83 ± $0.17
   - New model RMSE: $2.85 ± $0.22
   - Low variance = consistent performance

2. **Statistical Testing:** t-test with p-value = 0.000093
   - p < 0.05 = statistically significant
   - Only 0.009% chance improvement is random
   - Strong evidence of real improvement

3. **Multiple Metrics:** All metrics improved (RMSE, MAE, R²)
   - If only one metric improved, could be random
   - All improving = robust evidence

**B. Avoiding Data Leakage**
1. **Train-Test Split:** Used 80-20 split
   - Model never sees test data during training
   - Prevents memorization

2. **Feature Engineering Done Before Split:** 
   - Created trip_duration_minutes before splitting
   - No information leakage from test to train

3. **No Target Leakage:**
   - trip_duration_minutes doesn't contain information about fare_amount
   - It's an independent variable calculated from timestamps
   
4. **Temporal Validation:**
   - Could further validate by training on early dates, testing on later dates
   - Ensures model works on future data

**Potential Leakage Risk:**
- If fare calculation uses duration in production, there could be circular dependency
- Mitigated by using duration as predictor, not target

---

### Q2: What risks exist if this feature cannot be reliably generated in production?

**Answer:**

**FEATURE: trip_duration_minutes**

**A. Real-Time Prediction Problem**

**Risk:**
- To predict fare BEFORE trip ends, we don't know duration yet!
- Feature only available AFTER trip completes
- Cannot use this model for pre-trip fare estimates

**Impact:**
- Model useless for ride-hailing apps (Uber, Lyft) that need upfront pricing
- Only works for post-trip fare validation
- Severely limits deployment scenarios

**Mitigation:**
- Use estimated duration based on:
  - Historical data for route
  - Current traffic conditions
  - Time of day patterns
  - Google Maps/Waze API predictions

---

**B. Data Availability Issues**

**Risk:**
- Requires accurate timestamp data from GPS/system
- GPS failures → missing/incorrect timestamps
- System clock errors → wrong duration calculations
- Network issues → delayed timestamp recording

**Impact:**
- Missing data = model cannot make predictions
- Incorrect timestamps = wrong predictions
- Need fallback for ~5-10% of trips

**Mitigation:**
- Validate timestamps before using:
```python
  if pickup_time > dropoff_time:
      # Invalid data, use fallback model
```
- Monitor feature availability in production
- Have backup model without duration feature
- Alert system for data quality issues

---

**C. Data Leakage in Production**

**Risk:**
- If fare is calculated using duration, model "cheats" by using target information
- Creates circular dependency: fare depends on duration, we predict fare from duration
- Model performance in training won't match production

**Example:**
```
# If taxi meter calculates fare as:
fare = base_fare + (rate_per_minute × duration)

# Then using duration to predict fare is circular!
```

**Impact:**
- Inflated performance metrics during development
- Poor real-world performance
- Trust issues with stakeholders

**Mitigation:**
- Understand fare calculation logic in production
- If duration is used in fare, DON'T use it as predictor
- Document dependencies clearly
- Use only truly independent features

---

**D. Production Deployment Challenges**

**Risk:**
- Real-time timestamp processing required
- Time zone handling complexity (trips across zones)
- Clock synchronization across distributed systems
- Edge cases: very long trips, overnight trips, system downtime

**Impact:**
- Complex infrastructure requirements
- Higher maintenance costs
- More potential failure points
- Debugging difficulties

**Example Edge Cases:**
```
- Trip crosses midnight: duration calculation wrong
- Daylight saving time change during trip
- GPS outage mid-trip: partial data
- Trip paused (driver break): inflates duration
```

**Mitigation:**
- Robust timestamp handling:
```python
  # Handle timezone-aware timestamps
  duration = (dropoff - pickup).total_seconds() / 60
  
  # Cap unrealistic values
  if duration > 180:  # 3 hours max
      duration = median_duration_for_distance
```
- Comprehensive testing for edge cases
- Monitoring and alerting for anomalies
- Feature validation pipeline

---

**E. Model Maintenance & Monitoring**

**Risk:**
- Feature distribution may change over time
- Traffic patterns evolve (new roads, construction)
- Model performance degrades silently

**Impact:**
- Predictions become less accurate
- User complaints increase
- Revenue loss from incorrect fares

**Mitigation:**
- Monitor feature statistics in production:
  - Track mean, median, percentiles of duration
  - Alert if distribution shifts significantly
- Regular model retraining (monthly/quarterly)
- A/B testing for model updates
- Fallback to simpler model if issues detected

---

**F. Recommended Production Strategy**

1. **Primary Model:** Use estimated duration from route/traffic data
   - Available before trip starts
   - Reliable and validated
   - No dependency on actual trip completion

2. **Fallback Model:** 3-feature model without duration
   - When duration unavailable/unreliable
   - Simpler, more robust
   - Proven 91% R² performance

3. **Monitoring Dashboard:**
   - Feature availability rate
   - Prediction error trends
   - Duration distribution shifts
   - Model switching frequency

4. **Validation Pipeline:**
```python
   def predict_fare(trip_data):
       if has_valid_duration(trip_data):
           return duration_model.predict(trip_data)
       else:
           return fallback_model.predict(trip_data)
```

---

## 8. Conclusion

### 8.1 Summary of Results

**Regression Model:**
- ✅ Initial 3-feature model: R² = 0.9149
- ✅ Improved 4-feature model: R² = 0.9556
- ✅ Statistically significant improvement (p < 0.001)
- ✅ Added feature (trip_duration) is highly valuable

**Classification Model:**
- ✅ Initial 3-feature model: 91.99% accuracy
- ❌ Improved 4-feature model: 91.82% accuracy (no significant change)
- ❌ Added feature doesn't help for payment type prediction
- ✅ tip_amount is the dominant feature (65% importance)

### 8.2 Key Learnings

1. **Feature Engineering Impact:**
   - Same feature (duration) can be highly effective for one task (regression)
   - But ineffective for another task (classification)
   - Importance depends on problem domain

2. **Statistical Validation is Critical:**
   - Always use cross-validation
   - Perform statistical tests (t-test, p-values)
   - Don't trust single-run results

3. **Production Considerations:**
   - Feature availability in real-time is crucial
   - Data leakage risks must be identified early
   - Always have fallback strategies

4. **Model Selection:**
   - Linear Regression: Fast, interpretable, works well for linear relationships
   - Random Forest: Handles non-linearity, provides feature importance
   - Choose based on problem requirements

### 8.3 Future Work

1. **Try More Features:**
   - Hour of day (rush hour pricing)
   - Day of week (weekend vs weekday)
   - Weather conditions
   - Traffic density

2. **Advanced Models:**
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks for complex patterns
   - Ensemble methods combining multiple models

3. **Production Deployment:**
   - Build REST API for model serving
   - Implement monitoring dashboard
   - A/B testing framework
   - Automated retraining pipeline

4. **Feature Store:**
   - Centralized feature management
   - Ensure train-serve consistency
   - Track feature lineage

---

## 9. Repository Structure
```
mlops-practice/
│
├── README.md                              # Project overview
├── requirements.txt                       # Python dependencies
├── .gitignore                            # Git ignore rules
│
├── green_tripdata_2021-01.parquet        # Dataset (not in repo)
│
├── data_exploration.py                   # Initial data analysis
├── data_visualization.py                 # Create plots
├── data_summary.txt                      # Data statistics
│
├── regression_model.py                   # Initial regression model (3 features)
├── regression_model_improved.py          # Improved regression (4 features)
├── linear_regression_model.pkl           # Saved model (3 features)
├── linear_regression_model_improved.pkl  # Saved model (4 features)
│
├── classification_model.py               # Initial classification (3 features)
├── classification_model_improved.py      # Improved classification (4 features)
├── random_forest_classifier.pkl          # Saved model (3 features)
├── random_forest_classifier_improved.pkl # Saved model (4 features)
│
├── plots/                                # Data visualizations
│   ├── trip_distance_distribution.png
│   ├── fare_vs_distance.png
│   ├── correlation_heatmap.png
│   └── passenger_count_distribution.png
│
└── results/                              # Model results and analysis
    ├── regression_results.txt
    ├── regression_actual_vs_predicted.png
    ├── regression_residuals.png
    ├── regression_improvement_report.txt
    ├── regression_improvement_analysis.png
    ├── classification_results.txt
    ├── classification_confusion_matrix.png
    ├── classification_feature_importance.png
    └── classification_improvement_report.txt
```

### Git Branches
- `main`: Complete project with all models
- `regression-model`: Regression model code
- `classification-model`: Classification model code

---

## 10. How to Run This Project

### 10.1 Setup Environment
```bash
# Clone repository
git clone https://github.com/debotush/mlops-practice.git
cd mlops-practice

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 10.2 Download Data

Download `green_tripdata_2021-01.parquet` from course page and place in project root.

### 10.3 Run Analysis
```bash
# Data exploration
python data_exploration.py
python data_visualization.py

# Train models
python regression_model.py
python classification_model.py

# Improved models with 4th feature
python regression_model_improved.py
python classification_model_improved.py
```

### 10.4 View Results

- Plots: `plots/` directory
- Model results: `results/` directory
- Saved models: `*.pkl` files

---

## 11. References

- Dataset: NYC Taxi and Limousine Commission (TLC) Trip Record Data
- Scikit-learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/
- MLOps Best Practices: https://ml-ops.org/

---

**End of Report**
