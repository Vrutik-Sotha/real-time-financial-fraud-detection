"""
Data Preprocessing and Feature Engineering Module for Financial Fraud Detection
Designed for the Kartik2112 Credit Card Fraud Detection Dataset.
Transforms raw transaction metadata into human-interpretable engineered features.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Benchmark Kartik2112 merchant categories
ALL_CATEGORIES = [
    'entertainment',
    'food_dining',
    'gas_transport',
    'grocery_net',
    'grocery_pos',
    'health_fitness',
    'home',
    'kids_pets',
    'misc_net',
    'misc_pos',
    'personal_care',
    'shopping_net',
    'shopping_pos',
    'travel'
]

# Engineered model features
NUMERICAL_FEATURES = ['amt', 'hour', 'day_of_week', 'is_weekend', 'distance_km', 'city_pop', 'age']
CATEGORICAL_FEATURES = ['category']
MODEL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth using the Haversine formula. Vectorized with NumPy.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    km = 6371.0 * c
    return km


def load_dataset(file_path_or_buffer):
    """
    Loads a transaction CSV dataset.
    Returns: pandas.DataFrame
    """
    df = pd.read_csv(file_path_or_buffer)
    # Strip unnecessary index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df


def validate_dataset_schema(df, require_target=True):
    """
    Validates that the uploaded DataFrame contains the essential columns
    expected from the Kartik2112 schema.
    Returns: (is_valid: bool, error_message: str)
    """
    required_cols = ['amt', 'category', 'trans_date_trans_time']
    if require_target:
        required_cols.append('is_fraud')

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, "Schema validated successfully."


def engineer_features(df):
    """
    Engineers human-readable model features from raw transaction records:
    - Extracts 'hour', 'day_of_week', and 'is_weekend' from 'trans_date_trans_time'
    - Calculates 'distance_km' from customer (lat, long) to merchant (merch_lat, merch_long)
    - Calculates 'age' from 'dob' and transaction date
    - Normalizes 'category' to lowercase standard tokens
    """
    clean_df = df.copy()

    # 1. Parse timestamps
    if 'trans_date_trans_time' in clean_df.columns:
        dt_series = pd.to_datetime(clean_df['trans_date_trans_time'], errors='coerce')
        clean_df['hour'] = dt_series.dt.hour.fillna(12).astype(int)
        clean_df['day_of_week'] = dt_series.dt.dayofweek.fillna(2).astype(int)
        clean_df['is_weekend'] = clean_df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
    else:
        clean_df['hour'] = clean_df.get('hour', 12)
        clean_df['day_of_week'] = clean_df.get('day_of_week', 2)
        clean_df['is_weekend'] = clean_df.get('is_weekend', 0)

    # 2. Calculate Distance
    if all(col in clean_df.columns for col in ['lat', 'long', 'merch_lat', 'merch_long']):
        clean_df['distance_km'] = haversine_distance(
            clean_df['lat'].values,
            clean_df['long'].values,
            clean_df['merch_lat'].values,
            clean_df['merch_long'].values
        )
    elif 'distance_km' not in clean_df.columns:
        clean_df['distance_km'] = 10.0  # default typical distance in km

    # 3. Calculate Age
    if 'dob' in clean_df.columns and 'trans_date_trans_time' in clean_df.columns:
        dob_dt = pd.to_datetime(clean_df['dob'], errors='coerce')
        tx_dt = pd.to_datetime(clean_df['trans_date_trans_time'], errors='coerce')
        clean_df['age'] = (tx_dt.dt.year - dob_dt.dt.year).fillna(40).astype(int)
        clean_df['age'] = clean_df['age'].clip(18, 100)
    elif 'age' not in clean_df.columns:
        clean_df['age'] = 40

    # 4. Fallbacks for Amount and City Population
    if 'amt' not in clean_df.columns:
        clean_df['amt'] = 50.0
    if 'city_pop' not in clean_df.columns:
        clean_df['city_pop'] = 50000

    # 5. Clean category
    clean_df['category'] = clean_df['category'].astype(str).str.lower().str.strip()

    return clean_df


def build_preprocessing_pipeline():
    """
    Constructs a Scikit-Learn ColumnTransformer pipeline:
    - StandardScaler on numerical features (amt, hour, day_of_week, is_weekend, distance_km, city_pop, age)
    - OneHotEncoder on categorical feature (category) with fixed categories and handle_unknown='ignore'
    """
    pipeline = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(categories=[ALL_CATEGORIES], handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return pipeline


def prepare_data_for_modeling(df):
    """
    Extracts engineered features (X) and target variable (y) from a DataFrame.
    Returns: X (DataFrame), y (Series or None)
    """
    engineered_df = engineer_features(df)
    X = engineered_df[MODEL_FEATURE_COLUMNS].copy()

    y = None
    if 'is_fraud' in engineered_df.columns:
        y = engineered_df['is_fraud'].astype(int)

    return X, y


def format_manual_transaction(input_dict):
    """
    Converts a user-entered transaction dictionary into a single-row DataFrame
    compatible with the model's preprocessing pipeline.
    
    Expected keys:
    - amt: float
    - category: str
    - date_time: datetime or str
    - distance_km: float
    - city_pop: int
    - age: int
    """
    dt_val = input_dict.get('date_time', datetime.now())
    if isinstance(dt_val, str):
        dt_val = pd.to_datetime(dt_val, errors='coerce') or datetime.now()

    hour = dt_val.hour
    day_of_week = dt_val.weekday()
    is_weekend = 1 if day_of_week in [5, 6] else 0

    cat_cleaned = str(input_dict.get('category', 'grocery_pos')).lower().strip()
    if cat_cleaned not in ALL_CATEGORIES:
        cat_cleaned = 'misc_pos'

    row = {
        'amt': float(input_dict.get('amt', 50.0)),
        'hour': int(hour),
        'day_of_week': int(day_of_week),
        'is_weekend': int(is_weekend),
        'distance_km': float(input_dict.get('distance_km', 10.0)),
        'city_pop': int(input_dict.get('city_pop', 50000)),
        'age': int(input_dict.get('age', 40)),
        'category': cat_cleaned
    }
    return pd.DataFrame([row])


def generate_benchmark_dataset_files(data_dir="data", n_train=4000, n_test=1000):
    """
    Generates realistic benchmark training and testing datasets mirroring the exact
    Kartik2112 dataset schema if fraudTrain.csv / fraudTest.csv are not yet downloaded.
    Ensures the application works out-of-the-box.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "fraudTrain.csv")
    test_path = os.path.join(data_dir, "fraudTest.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        return train_path, test_path

    np.random.seed(42)

    def _create_sample_df(n_samples):
        # Generate timestamps
        start_date = pd.Timestamp("2024-01-01")
        random_seconds = np.random.randint(0, 180 * 86400, size=n_samples)
        timestamps = [start_date + pd.Timedelta(seconds=int(s)) for s in random_seconds]

        # Categories
        categories = np.random.choice(ALL_CATEGORIES, size=n_samples)

        # Baseline: 98.8% Legit, 1.2% Fraud
        is_fraud = np.random.choice([0, 1], size=n_samples, p=[0.988, 0.012])

        # Amounts: legit median ~$50, fraud median ~$500 with higher variance
        amt = []
        for f, cat in zip(is_fraud, categories):
            if f == 1:
                val = np.random.exponential(scale=350.0) + 150.0
            else:
                val = np.random.exponential(scale=45.0) + 5.0
            amt.append(round(min(val, 5000.0), 2))

        # Coordinates around US
        base_lat = 37.0 + np.random.normal(0, 4.0, size=n_samples)
        base_long = -95.0 + np.random.normal(0, 10.0, size=n_samples)

        # Merchant coordinates (fraud has higher distances)
        merch_lat = []
        merch_long = []
        for f, lat, lon in zip(is_fraud, base_lat, base_long):
            delta = np.random.normal(0, 2.5 if f == 1 else 0.15)
            merch_lat.append(lat + delta)
            merch_long.append(lon + delta)

        city_pop = np.random.randint(1000, 1200000, size=n_samples)
        dob = [f"{np.random.randint(1950, 2005)}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 28):02d}" for _ in range(n_samples)]

        df_out = pd.DataFrame({
            "trans_date_trans_time": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
            "cc_num": [f"400012345678{i % 1000:04d}" for i in range(n_samples)],
            "merchant": [f"fraud_merchant_{i % 50}" for i in range(n_samples)],
            "category": categories,
            "amt": amt,
            "gender": np.random.choice(["M", "F"], size=n_samples),
            "city": [f"City_{i % 20}" for i in range(n_samples)],
            "state": ["CA", "NY", "TX", "FL", "IL"][0:1] * n_samples,
            "zip": [90001 + (i % 50) for i in range(n_samples)],
            "lat": base_lat,
            "long": base_long,
            "city_pop": city_pop,
            "job": ["Engineer", "Teacher", "Manager", "Analyst", "Clerk"][0:1] * n_samples,
            "dob": dob,
            "trans_num": [f"tx_{i:06d}" for i in range(n_samples)],
            "unix_time": [int(t.timestamp()) for t in timestamps],
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "is_fraud": is_fraud
        })
        return df_out

    if not os.path.exists(train_path):
        train_df = _create_sample_df(n_train)
        train_df.to_csv(train_path, index=False)

    if not os.path.exists(test_path):
        test_df = _create_sample_df(n_test)
        test_df.to_csv(test_path, index=False)

    return train_path, test_path


def generate_sample_csv_template():
    """
    Generates a downloadable sample CSV template for Bulk CSV Upload.
    Contains realistic valid and mixed transactions matching the required schema.
    """
    sample_rows = [
        {"amt": 28.50, "category": "grocery_pos", "trans_date_trans_time": "2024-05-10 14:15:00", "distance_km": 4.2, "city_pop": 55000, "age": 35},
        {"amt": 15.75, "category": "food_dining", "trans_date_trans_time": "2024-05-10 18:30:00", "distance_km": 6.8, "city_pop": 55000, "age": 42},
        {"amt": 45.00, "category": "gas_transport", "trans_date_trans_time": "2024-05-11 08:20:00", "distance_km": 11.5, "city_pop": 55000, "age": 50},
        {"amt": 1850.00, "category": "shopping_net", "trans_date_trans_time": "2024-05-11 03:45:00", "distance_km": 820.0, "city_pop": 350000, "age": 28},
        {"amt": 89.90, "category": "entertainment", "trans_date_trans_time": "2024-05-11 20:10:00", "distance_km": 15.0, "city_pop": 55000, "age": 22},
        {"amt": 2400.00, "category": "misc_net", "trans_date_trans_time": "2024-05-12 02:15:00", "distance_km": 1450.0, "city_pop": 1200000, "age": 31},
        {"amt": 62.40, "category": "health_fitness", "trans_date_trans_time": "2024-05-12 11:00:00", "distance_km": 8.0, "city_pop": 55000, "age": 45},
        {"amt": 12.50, "category": "personal_care", "trans_date_trans_time": "2024-05-12 15:45:00", "distance_km": 3.1, "city_pop": 55000, "age": 29},
        {"amt": 1420.00, "category": "travel", "trans_date_trans_time": "2024-05-13 01:30:00", "distance_km": 2100.0, "city_pop": 850000, "age": 26},
        {"amt": 38.00, "category": "home", "trans_date_trans_time": "2024-05-13 16:20:00", "distance_km": 7.5, "city_pop": 55000, "age": 58}
    ]
    df_tmpl = pd.DataFrame(sample_rows)
    return df_tmpl.to_csv(index=False)


def persist_and_register_dataset(file_or_df, filename, target_dir=os.path.join("data", "registered_datasets"), db_path=os.path.join("data", "monitoring.db")):
    """
    Saves an uploaded or newly provided dataset into data/registered_datasets/
    and registers it in the SQLite database registry as the active dataset.
    Returns: (success: bool, info_or_error: dict|str)
    """
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)

    try:
        if isinstance(file_or_df, pd.DataFrame):
            df = file_or_df.copy()
            df.to_csv(target_path, index=False)
        else:
            if hasattr(file_or_df, "seek"):
                file_or_df.seek(0)
            df = pd.read_csv(file_or_df)
            df.to_csv(target_path, index=False)

        is_valid, msg = validate_dataset_schema(df, require_target=True)
        if not is_valid:
            if os.path.exists(target_path):
                os.remove(target_path)
            return False, msg

        total = len(df)
        fraud = int(df['is_fraud'].sum()) if 'is_fraud' in df.columns else 0
        legit = total - fraud
        size_kb = os.path.getsize(target_path) / 1024.0

        from database import register_dataset, get_active_dataset
        register_dataset(
            filename=filename,
            filepath=os.path.abspath(target_path),
            total_records=total,
            fraud_records=fraud,
            legit_records=legit,
            file_size_kb=size_kb,
            is_active=True,
            db_path=db_path
        )
        active_meta = get_active_dataset(db_path=db_path)
        return True, active_meta
    except Exception as ex:
        return False, str(ex)


