# MLOps Fundamentals - Practice Session

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Machine Learning Operations practice project implementing regression and classification models on NYC Green Taxi dataset.

## 🎯 Project Overview

This project demonstrates end-to-end ML model development including:
- Data exploration and visualization
- Training regression and classification models
- Model evaluation with multiple metrics
- Feature engineering and improvement analysis
- Statistical validation of model improvements
- Production deployment considerations

## 📊 Models Implemented

### 1. Regression Model
- **Task:** Predict taxi fare amount
- **Algorithm:** Linear Regression
- **Performance:** R² = 0.9556 (95.56%)
- **Features:** trip_distance, passenger_count, PULocationID, trip_duration_minutes

### 2. Classification Model
- **Task:** Predict payment type (Cash vs Credit Card)
- **Algorithm:** Random Forest Classifier
- **Performance:** Accuracy = 91.99%
- **Features:** trip_distance, fare_amount, tip_amount

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda

### Installation
```bash
# Clone the repository
git clone https://github.com/debotush/mlops-practice.git
cd mlops-practice

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# 1. Data exploration
python data_exploration.py
python data_visualization.py

# 2. Train initial models (3 features)
python regression_model.py
python classification_model.py

# 3. Train improved models (4 features)
python regression_model_improved.py
python classification_model_improved.py
```

## 📁 Project Structure
```
mlops-practice/
├── README.md                              # This file
├── FINAL_REPORT.md                        # Comprehensive project report
├── requirements.txt                       # Python dependencies
│
├── data_exploration.py                    # Data analysis
├── data_visualization.py                  # Visualization generation
│
├── regression_model.py                    # Regression (3 features)
├── regression_model_improved.py           # Regression (4 features)
├── classification_model.py                # Classification (3 features)
├── classification_model_improved.py       # Classification (4 features)
│
├── plots/                                 # Data visualizations
└── results/                               # Model results and reports
```

## 📈 Key Results

### Regression Model Improvement
| Metric | 3 Features | 4 Features | Improvement |
|--------|-----------|-----------|-------------|
| RMSE   | $3.83     | $2.85     | **$0.99** ✅ |
| R²     | 0.9198    | 0.9556    | **+3.58%** ✅ |
| p-value| -         | 0.000093  | **Significant** ✅ |

### Classification Model
| Metric | Value |
|--------|-------|
| Accuracy | 91.99% |
| Precision | 0.93 |
| Recall | 0.92 |
| F1-Score | 0.92 |

## 🔍 Dataset

**Source:** NYC Taxi and Limousine Commission (TLC)
- **File:** green_tripdata_2021-01.parquet
- **Period:** January 2021
- **Records:** 76,518 (38,191 after cleaning)
- **Features:** 20 columns including trip distance, fare, payment type, timestamps

## 📝 Features Used

### Regression Model (Predicting Fare)
1. `trip_distance` - Distance in miles
2. `passenger_count` - Number of passengers
3. `PULocationID` - Pickup location
4. `trip_duration_minutes` - Trip duration (NEW)

### Classification Model (Predicting Payment Type)
1. `trip_distance` - Distance in miles
2. `fare_amount` - Fare in USD
3. `tip_amount` - Tip in USD (strongest predictor!)
4. `trip_duration_minutes` - Trip duration (minimal impact)

## 🔬 Key Findings

1. **trip_duration_minutes significantly improves regression** (p < 0.001)
   - RMSE reduced by 25.8%
   - R² increased from 91.98% to 95.56%

2. **trip_duration_minutes does NOT improve classification** (p = 0.69)
   - Only 0.08% accuracy increase (not significant)
   - tip_amount is the dominant feature (65% importance)

3. **Production Risks Identified:**
   - Duration unavailable for pre-trip predictions
   - Requires reliable timestamp data
   - Potential data leakage if fare calculation uses duration

## 📊 Visualizations

All visualizations are saved in the `plots/` directory:
- Trip distance distribution
- Fare vs Distance scatter plot
- Correlation heatmap
- Passenger count distribution
- Confusion matrix
- Feature importance
- Model comparison plots

## 🛠️ Technologies Used

- **Python 3.12**
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning models
- **matplotlib & seaborn** - Data visualization
- **numpy** - Numerical computing
- **scipy** - Statistical testing

## 📚 Documentation

For detailed analysis, methodology, and results, see [FINAL_REPORT.md](FINAL_REPORT.md).

## 🌿 Git Branches

- `main` - Complete project
- `regression-model` - Regression model development
- `classification-model` - Classification model development

## 🎓 Assignment Requirements

- [x] Create Git repository
- [x] Train regression model
- [x] Train classification model
- [x] Use 3-4 features
- [x] Evaluate with appropriate metrics
- [x] Save models to separate branches
- [x] Add 4th feature and analyze improvement
- [x] Address randomness and data leakage
- [x] Discuss production risks
- [x] Create comprehensive report

## 👤 Author

**Debotush**
- GitHub: [@debotush](https://github.com/debotush)

## 📄 License

This project is created for educational purposes as part of MLOps Fundamentals course.

## 🙏 Acknowledgments

- NYC Taxi and Limousine Commission for the dataset
- MLOps Fundamentals course instructors
- Scikit-learn and pandas communities

---

**Note:** The dataset file (`green_tripdata_2021-01.parquet`) is not included in the repository due to size. Download it from the course page.
