# Real-Time Financial Fraud Detection Platform 🛡️💳

An enterprise-grade, explainable Machine Learning application designed to detect fraudulent financial transactions using **human-readable transaction attributes** (Amount, Merchant Category, Geographic Distance, Timestamp, Cardholder Age, and Population).

Built with **Python**, **Scikit-learn**, **XGBoost**, **Pandas**, **SQLite**, and **Streamlit**.

---

## 📌 Project Architecture & Structure

For a full technical deep dive and Mermaid sequence/dataflow diagrams, see [PROJECT_ARCHITECTURE.md](file:///c:/Users/vrutik/OneDrive/Desktop/web%20development/Real-Time%20Financial%20Fraud%20Detection%20System/PROJECT_ARCHITECTURE.md).

```text
Real-Time Financial Fraud Detection System/
│
├── app.py                     # Streamlit 4-page FinTech web application
├── model.py                   # Multi-model benchmarking (LR, RF, XGBoost), switching & backups
├── preprocessing.py           # Feature engineering, Haversine formula, and ColumnTransformer pipeline
├── database.py                # SQLite persistence (transaction_history, training_history, dataset_registry)
├── train.py                   # Model training and comparison CLI script
├── test_pipeline.py           # Comprehensive 15-module automated test suite
├── requirements.txt           # Project dependencies (Scikit-learn, XGBoost, Streamlit, etc.)
├── PROJECT_ARCHITECTURE.md    # Detailed architecture specifications and Mermaid diagrams
├── README.md                  # Complete documentation and interview preparation guide
│
├── data/
│   ├── fraudTrain.csv         # Baseline training dataset (Kartik2112 schema)
│   ├── fraudTest.csv          # Held-out evaluation dataset
│   ├── registered_datasets/   # Persistent storage for newly registered datasets
│   └── monitoring.db          # Persistent SQLite audit database
│
└── models/
    ├── fraud_model.pkl        # Active champion model & candidate bundle
    └── fraud_model_backup.pkl # Safety rollback copy of previous working model
```

---

## 🛠️ Technology Stack

- **Language:** Python 3.9+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & Ensembles:** Scikit-learn (Logistic Regression, Random Forest), XGBoost
- **Persistent Storage:** SQLite (`sqlite3`) with zero-configuration persistence
- **Data Visualization:** Matplotlib, Seaborn
- **Interactive UI:** Streamlit with custom FinTech CSS styling
- **Model Serialization:** Joblib

---

## 📥 Dataset Setup: Kartik2112 Credit Card Fraud Detection

This application is built for the **Kaggle: Credit Card Fraud Detection by Kartik2112** dataset:
🔗 [Kaggle Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

### Raw Dataset Attributes:
`trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`, `first`, `last`, `gender`, `street`, `city`, `state`, `zip`, `lat`, `long`, `city_pop`, `job`, `dob`, `trans_num`, `unix_time`, `merch_lat`, `merch_long`, `is_fraud`.

### Setup Instructions:
1. Download `fraudTrain.csv` and `fraudTest.csv` from Kaggle.
2. Place both files inside the `data/` directory:
   ```text
   data/fraudTrain.csv
   data/fraudTest.csv
   ```
3. *Note:* If the files are not yet downloaded, the system automatically initializes a benchmark sample dataset with the exact Kartik2112 schema so the application and test suite run immediately out-of-the-box.
4. *Privacy Notice:* The dataset is generated using the Sparkov synthetic generator. This ensures realistic consumer behavior while adhering to data privacy standards (no real bank customer PII is used).

---

## 🔍 Feature Engineering & Preprocessing Pipeline

To eliminate black-box PCA coordinates and ensure genuine manual transaction input, raw transaction records are transformed into 8 human-interpretable features:

| Feature Name | Type | Description | Extraction / Engineering Logic |
| :--- | :--- | :--- | :--- |
| **`amt`** | Numerical | Transaction dollar amount | Direct input |
| **`category`** | Categorical | Merchant business category | One-hot encoded across 14 standard categories |
| **`hour`** | Numerical | Hour of day (0–23) | Extracted from `trans_date_trans_time` |
| **`day_of_week`** | Numerical | Day of week (0=Mon, 6=Sun) | Extracted from `trans_date_trans_time` |
| **`is_weekend`** | Binary | Weekend indicator (0 or 1) | 1 if day of week is Saturday or Sunday |
| **`distance_km`** | Numerical | Customer-to-Merchant distance | Computed using vectorized **Haversine formula** from `(lat, long)` to `(merch_lat, merch_long)` |
| **`city_pop`** | Numerical | Customer city population | Direct demographic indicator |
| **`age`** | Numerical | Cardholder age | Transaction year minus birth year (`dob`) |

### Data Leakage Prevention:
`ColumnTransformer` standardizes numerical features with `StandardScaler` and encodes `category` with `OneHotEncoder(handle_unknown='ignore')`. The transformer is **fitted strictly on the training set** (`fraudTrain.csv`) and only transforms test or incoming manual records.

---

## ⚔️ Multi-Model Benchmarking & Dynamic Switching

The training pipeline trains and compares three distinct classifiers:
1. **Logistic Regression:** Linear decision boundary with balanced class weighting:
   $$w_j = \frac{N}{2 \times N_j}$$
2. **Random Forest:** Bagging ensemble of 100 decision trees with balanced sub-sampling (`class_weight='balanced'`).
3. **XGBoost:** Extreme Gradient Boosted Trees with scale-position weighting (`scale_pos_weight`):
   $$\text{scale\_pos\_weight} = \frac{N_{\text{legitimate}}}{N_{\text{fraud}}}$$

### 🔄 Dynamic Champion Model Switching:
The candidate models are all stored within the serialized bundle. In the sidebar, users can switch between **Logistic Regression**, **Random Forest**, and **XGBoost** via a dropdown. The active model immediately performs subsequent inferences **without requiring retraining or server restart**.

---

## 🚀 Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Run Initial Training CLI
```bash
python train.py
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`.

### 4. Run the Full Test Suite
```bash
python test_pipeline.py
```
Runs the 15-module end-to-end verification suite.

---

## 🖥️ Four-Page Application Walkthrough

### Page 1: 📊 Dashboard
- **KPI Metrics:** Total Training Records, Dataset Fraud Count, Fraud Percentage, Lifetime Monitored Transactions (queried live from SQLite).
- **Visuals:** Class distribution donut chart, Kartik2112 architecture overview, administrative system status card, and recent persistent audit snapshot.

### Page 2: 📈 Model Performance & Comparison
- **Benchmark Table:** Side-by-side comparison of Logistic Regression, Random Forest, and XGBoost across Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
- **Diagnostic Plots:** Confusion Matrix heatmap, ROC Curve, and Precision-Recall Curve on the held-out test dataset (`fraudTest.csv`).
- **Interview Explainer:** Concise educational breakdown on why PR-AUC and F1-score outperform Accuracy on imbalanced datasets.

### Page 3: 🔍 Transaction Prediction (Genuine Manual Input)
- Clean, human-readable input form:
  - Transaction Amount ($ USD)
  - Transaction Date & Time (Date & Time pickers)
  - Merchant Category (Dropdown of 14 categories)
  - Customer-to-Merchant Distance (km)
  - Customer City Population
  - Cardholder Age
- Quick demo scenario buttons (*Everyday Grocery Store ($34.20)* vs. *High-Value Online Anomaly ($1,850.00)*).
- Instant prediction with FinTech alert banners (`FRAUDULENT` / `LEGITIMATE`), probability progress gauge, and tiered risk classification.
- Automatically logged to SQLite audit trail.

### Page 4: 📡 Live Transaction Monitoring (Bulk & Continuous Ingestion)
- **Two Flexible Ingestion Modes:**
  - **Mode A: Bulk CSV Batch Processing:**
    - Downloadable sample CSV template (`sample_transactions_template.csv`).
    - Multi-record CSV batch upload with schema validation (`amt`, `category`, `trans_date_trans_time`, `distance_km`, `city_pop`, `age`).
    - Valid vs. invalid row detection and 5-row preview.
    - Vectorized model inference with live progress indicator.
    - Detailed prediction results table and one-click batch CSV export.
  - **Mode B: Continuous Live Stream Simulator:**
    - Configurable transaction count (5 to 100) and streaming pacing interval (0.05s to 1.5s).
    - Scenario stream profiles (*Normal Consumer Stream*, *High-Risk Anomaly Surge*, *Balanced Mixed Stream*).
    - Instant live streaming inference, real-time alert cards, and immediate individual SQLite persistence.
- **Common Unified Monitoring Dashboard:**
  - Real-time lifetime KPIs from SQLite: Total Analyzed, Flagged Fraudulent, Legitimate Cleared, High-Risk Alerts, Fraud Detection Rate, and Average Fraud Risk.
  - 4 diagnostic charts: Monitoring outcomes bar chart, cumulative activity time-series (`plot_transactions_over_time`), fraud probability distribution histogram, and ingestion source breakdown.
  - Filterable SQLite audit table (filter by Prediction, Risk Tier, Source, or Search Query).
  - One-click export of filtered audit logs to CSV and database reset controls.

### Sidebar: Persistent Dataset Management & Model Controls
- **Active Model Switching:** Dynamic dropdown to select champion algorithm.
- **Persistent Dataset Registry:** Displays the currently registered training dataset, total records, fraud rate, and registration timestamp.
- **Retrain on Registered Dataset:** Retrain directly on the registered dataset without re-uploading every session.
- **Replace / Register Dataset:** Upload new CSV, automatically validated, stored in `data/registered_datasets/`, and registered in SQLite.
- **Model Backup & Rollback:** Automatically backs up working models to `models/fraud_model_backup.pkl` with a **"Restore Previous Model Backup"** rollback button.

---

## 🎓 Technical Interview & Placement Guide

### 1. Why replace anonymized PCA features with human-readable attributes?
Real-world fraud investigators and banking interfaces require interpretable transaction variables (e.g. "Amount was $2,500 at 3:00 AM from a merchant 800 km away") to comply with Adverse Action Notice requirements. The Kartik2112 dataset enables genuine manual transaction input without losing statistical anomaly patterns.

### 2. How is Customer-to-Merchant Distance calculated?
We apply the **Haversine formula**, which computes great-circle distance over the Earth's spherical surface using customer `(lat, long)` and merchant `(merch_lat, merch_long)` coordinates:
$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right), \quad d = 2 R \arcsin(\sqrt{a})$$
In credit card fraud, extreme distances (e.g. cross-border or out-of-state Card-Not-Present purchases) correlate strongly with compromise.

### 3. Why are PR-AUC and F1-Score superior to Accuracy and ROC-AUC here?
In datasets where fraud is ~1%, a model predicting 100% legitimate transactions achieves 99% accuracy but zero fraud detection. ROC-AUC can remain overly optimistic when negative instances dominate. **Precision-Recall AUC (PR-AUC)** focuses specifically on the minority positive class.

### 4. How is data leakage prevented during retraining?
The `ColumnTransformer` is fitted solely on the training partition. Feature statistics (mean, variance, category dictionaries) are computed from training data and serialized into the model bundle with Joblib. Test data and live incoming transactions are only transformed using these frozen statistics.

### 5. Why use SQLite for monitoring history?
SQLite provides a zero-configuration, lightweight, serverless relational database built into Python's standard library. It ensures the audit trail persists across app restarts without requiring heavy infrastructure (e.g. PostgreSQL, Redis, or Kafka) in an academic or interview setting.
