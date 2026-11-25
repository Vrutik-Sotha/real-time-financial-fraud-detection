# Real-Time Financial Fraud Detection System

A Machine Learning-based financial fraud detection application built using Python and Streamlit. The system analyzes transaction attributes, predicts potentially fraudulent transactions, monitors incoming transactions through batch processing and simulation, and maintains prediction and model-training records using SQLite.

## Project Overview

Financial fraud detection is an important challenge in digital banking and online payment systems. This project demonstrates how machine learning can help identify suspicious financial transactions by analyzing transaction-related features.

The application provides an interactive dashboard where users can:

- Predict whether an individual transaction is potentially fraudulent.
- Upload CSV files to analyze multiple transactions.
- Simulate incoming transactions for continuous monitoring.
- Compare the performance of multiple machine learning algorithms.
- Track transaction history and model-training activity.
- Register datasets for reuse during model retraining.
- Switch between available trained models.

**Note:** This is an academic machine learning project. The live monitoring functionality uses simulated transactions and CSV uploads. It is not directly connected to real banking systems, payment gateways, or financial institutions.

## Key Features

### 1. Dashboard
- Displays transaction statistics and prediction summaries.
- Shows fraudulent and legitimate transaction distributions.
- Displays recent transaction activity.
- Provides an overview of system activity.

### 2. Model Performance
- Compares multiple machine learning algorithms.
- Displays accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Provides confusion matrices and performance visualizations.
- Helps evaluate the strengths and weaknesses of different models.

### 3. Transaction Prediction
- Accepts individual transaction details through a manual input form.
- Processes transaction attributes using the trained preprocessing pipeline.
- Predicts whether a transaction is potentially fraudulent or legitimate.
- Displays prediction probability and risk classification.
- Saves prediction records in the SQLite database.

### 4. Live Transaction Monitoring
- Supports bulk transaction analysis through CSV uploads.
- Processes multiple transactions using the trained model.
- Provides a simulated live transaction feed.
- Records predictions for monitoring and analysis.

### 5. Dataset Management and Retraining
- Maintains a persistent dataset registry.
- Allows registered datasets to be reused after application restarts.
- Supports model training and comparison using available labelled datasets.
- Supports switching between trained machine learning models.
- Includes model backup and rollback functionality.

Predictions are stored separately from training datasets. New predictions do not automatically become training data.

## Machine Learning Algorithms

The project implements and compares three classification algorithms:

| Algorithm | Description |
|---|---|
| Logistic Regression | Classification model used as a baseline, with class weighting to address class imbalance. |
| Random Forest | Ensemble learning algorithm that combines multiple decision trees. |
| XGBoost | Gradient boosting algorithm used for classification. |

The application uses model evaluation results to identify a candidate model for prediction.

## Dataset Information

The project is designed to support the transaction schema associated with the [Kartik2112 Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

### Dataset authenticity and current benchmark

The current repository version uses a locally generated benchmark dataset containing:

- `fraudTrain.csv` – 4,000 generated training records.
- `fraudTest.csv` – 1,000 generated test records.

**These files are generated benchmark data, not the original full Kaggle dataset.**

The application is designed around compatible transaction fields and can be used with the original dataset after obtaining and validating the actual files.

### Features used for prediction

The model uses eight engineered or selected features:

| Feature | Description |
|---|---|
| `amt` | Transaction amount |
| `hour` | Transaction hour |
| `day_of_week` | Day of the week |
| `is_weekend` | Indicates whether the transaction occurred on a weekend |
| `distance_km` | Distance between customer and merchant locations |
| `city_pop` | Population of the customer's city |
| `age` | Customer age |
| `category` | Transaction category |

Personally identifying fields and transaction identifiers are excluded from model training.

### Data preprocessing

The preprocessing pipeline includes:

- Feature engineering from transaction attributes.
- Calculation of geographical distance using the Haversine formula.
- Numerical feature scaling using `StandardScaler`.
- Categorical feature encoding using `OneHotEncoder`.
- A consistent preprocessing pipeline for training and prediction.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning and preprocessing |
| XGBoost | Gradient boosting classification |
| SQLite | Persistent database storage |
| Joblib | Model serialization |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |

## Project Structure

```text
Real-Time Financial Fraud Detection System/
│
├── app.py
├── model.py
├── preprocessing.py
├── database.py
├── train.py
├── test_pipeline.py
│
├── utils/
│   └── visualization.py
│
├── data/
│   └── Generated or downloaded datasets
│
├── models/
│   └── Saved model artifacts
│
├── PROJECT_ARCHITECTURE.md
├── README.md
└── requirements.txt
```

**Note:** Dataset files, local database files, trained model artifacts, and virtual environments are excluded from the GitHub repository through `.gitignore`.

## Installation and Setup

### Prerequisites

- Python 3.10 or later (use a version compatible with the dependencies).
- Git.
- pip package manager.

### Step 1: Clone the repository

```bash
git clone https://github.com/Vrutik-Sotha/real-time-financial-fraud-detection.git
```

### Step 2: Navigate to the project directory

```bash
cd real-time-financial-fraud-detection
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the application

```bash
streamlit run app.py
```

The application will open in your browser, generally at:

```text
http://localhost:8501
```

Follow the application's dataset initialization instructions if it needs benchmark data or a downloaded dataset.

## Database and Persistence

The project uses SQLite to maintain application records, including:

- Transaction prediction history.
- Model training history.
- Registered dataset information.

This allows the application to retain relevant records across application restarts.

## Testing

The project includes an automated test suite in `test_pipeline.py`.

Run the tests using:

```bash
pytest test_pipeline.py -v
```

The tests are intended to validate application components and the machine learning workflow. Passing automated tests does not independently establish real-world model accuracy or production readiness.

## Important Limitations

- The currently available benchmark dataset is synthetically generated.
- Performance on generated data should not be interpreted as real-world fraud detection performance.
- Real banking or payment gateway integration is not implemented.
- Live monitoring currently relies on simulation and uploaded CSV data.
- Model performance depends on dataset quality, class distribution, and validation methodology.
- The application is an academic demonstration and is not intended for production financial decisions.

## Future Enhancements

- Integration with real-time transaction data streams.
- Evaluation using the complete original Kaggle dataset.
- Independent validation and testing on larger datasets.
- Advanced anomaly detection techniques.
- Improved model explainability using SHAP.
- Role-based authentication and access control.
- Cloud deployment and scalable database integration.
- Automated monitoring of model drift.

## Project Objective

The primary objective of this project is to demonstrate how machine learning can be applied to financial transaction classification through an interactive application that combines prediction, model evaluation, transaction monitoring, dataset management, and persistent record keeping.

## Author

**Vrutik Sotha**

Computer Science and Engineering

