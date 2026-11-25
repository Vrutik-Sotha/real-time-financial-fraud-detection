"""
Machine Learning Model Training, Comparison, and Inference Module
Supports Logistic Regression, Random Forest, and XGBoost with class imbalance handling,
comprehensive metrics (Precision, Recall, F1, ROC-AUC, PR-AUC), model serialization, and backup.
"""

import os
import shutil
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)

from preprocessing import build_preprocessing_pipeline, format_manual_transaction

DEFAULT_MODEL_PATH = os.path.join("models", "fraud_model.pkl")
BACKUP_MODEL_PATH = os.path.join("models", "fraud_model_backup.pkl")


def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Computes all standard classification and imbalanced metrics:
    Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    roc_auc_val = None
    pr_auc_val = None

    if y_probs is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc_val = roc_auc_score(y_true, y_probs)
        except Exception:
            roc_auc_val = None

        try:
            p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
            pr_auc_val = auc(r_curve, p_curve)
        except Exception:
            pr_auc_val = None

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc_val) if roc_auc_val is not None else None,
        "pr_auc": float(pr_auc_val) if pr_auc_val is not None else None,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_probs": y_probs
    }


def train_and_compare_models(X_train, y_train, X_test, y_test, random_state=42):
    """
    Trains and compares 3 diverse classifiers:
    1. Logistic Regression (Linear baseline with class weights)
    2. Random Forest (Ensemble bagging trees with balanced subsamples)
    3. XGBoost (Extreme Gradient Boosted Trees with positive class scale weighting)
    
    Fits the preprocessing pipeline strictly on X_train to prevent data leakage.
    Returns:
        best_bundle: dict (model, pipeline, metrics, model_name, etc.)
        comparison_results: dict mapping model_name -> metrics
    """
    # 1. Fit Preprocessing Pipeline strictly on Training Data
    pipeline = build_preprocessing_pipeline()
    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans = pipeline.transform(X_test)

    # Calculate class imbalance weighting factor
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos = (n_neg / n_pos) if n_pos > 0 else 1.0

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=min(scale_pos, 100.0),
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1
        )
    }

    comparison_results = {}
    best_name = None
    best_f1 = -1.0
    best_model = None

    for name, model in candidate_models.items():
        # Train
        model.fit(X_train_trans, y_train)

        # Inference on Test set
        y_pred = model.predict(X_test_trans)
        try:
            y_probs = model.predict_proba(X_test_trans)[:, 1]
        except Exception:
            y_probs = None

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_probs)
        comparison_results[name] = metrics

        # Select model prioritizing F1-score (harmonic balance of Precision and Recall)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_name = name
            best_model = model

    best_bundle = {
        "model": best_model,
        "pipeline": pipeline,
        "model_name": best_name,
        "metrics": comparison_results[best_name],
        "all_comparisons": comparison_results,
        "candidate_models": candidate_models,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

    return best_bundle, comparison_results


def save_model_bundle(bundle, file_path=DEFAULT_MODEL_PATH):
    """Saves the trained model and preprocessing pipeline bundle using Joblib."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(bundle, file_path)
    return file_path


def load_model_bundle(file_path=DEFAULT_MODEL_PATH):
    """Loads the trained model and pipeline bundle from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model bundle not found at {file_path}")
    bundle = joblib.load(file_path)
    # Ensure candidate_models dictionary is populated for seamless switching
    if "candidate_models" not in bundle or not bundle["candidate_models"]:
        model_name = bundle.get("model_name", "Trained Classifier")
        bundle["candidate_models"] = {model_name: bundle.get("model")}
    return bundle


def switch_active_model(bundle, target_model_name):
    """
    Switches the active model in the bundle to one of the trained candidate models
    without retraining or overwriting.
    """
    if bundle is None:
        return bundle
    candidates = bundle.get("candidate_models", {})
    if target_model_name in candidates and candidates[target_model_name] is not None:
        bundle["model_name"] = target_model_name
        bundle["model"] = candidates[target_model_name]
        if "all_comparisons" in bundle and target_model_name in bundle["all_comparisons"]:
            bundle["metrics"] = bundle["all_comparisons"][target_model_name]
    return bundle


def backup_model_bundle(source_path=DEFAULT_MODEL_PATH, backup_path=BACKUP_MODEL_PATH):
    """Creates a safety backup of the current working model bundle."""
    if os.path.exists(source_path):
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(source_path, backup_path)
        return True
    return False


def restore_model_backup(backup_path=BACKUP_MODEL_PATH, target_path=DEFAULT_MODEL_PATH):
    """Restores the previous working model bundle from backup."""
    if os.path.exists(backup_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(backup_path, target_path)
        return True
    return False


def predict_manual_transaction(bundle, manual_input_dict, risk_threshold_low=0.30, risk_threshold_high=0.70):
    """
    Transforms a manual human-readable transaction dictionary,
    applies the fitted pipeline, and runs model inference.
    
    Returns:
        dict containing prediction (int & str), fraud_probability (float),
        risk_category (str), and formatted attributes.
    """
    model = bundle["model"]
    pipeline = bundle["pipeline"]

    # 1. Format into DataFrame
    sample_df = format_manual_transaction(manual_input_dict)

    # 2. Transform using fitted preprocessing pipeline
    sample_trans = pipeline.transform(sample_df)

    # 3. Predict class and probability
    pred = int(model.predict(sample_trans)[0])
    try:
        prob = float(model.predict_proba(sample_trans)[0][1])
    except Exception:
        prob = 1.0 if pred == 1 else 0.0

    # 4. Determine Risk Category
    if prob < risk_threshold_low:
        risk_category = "Low Risk"
    elif prob <= risk_threshold_high:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"

    return {
        "prediction_code": pred,
        "prediction_label": "FRAUDULENT" if pred == 1 else "LEGITIMATE",
        "fraud_probability": prob,
        "risk_category": risk_category,
        "model_name": bundle.get("model_name", "Trained Classifier"),
        "formatted_features": sample_df.to_dict(orient="records")[0]
    }


def predict_bulk_transactions(bundle, df, risk_threshold_low=0.30, risk_threshold_high=0.70, source="Bulk CSV Upload", start_tx_id=6000):
    """
    Vectorized bulk prediction on a DataFrame of transactions.
    Applies the fitted preprocessing pipeline, generates predictions and probabilities,
    assigns risk categories, and prepares formatted records ready for SQLite and UI display.
    """
    from preprocessing import engineer_features, MODEL_FEATURE_COLUMNS

    model = bundle["model"]
    pipeline = bundle["pipeline"]

    # 1. Feature engineer the uploaded batch
    engineered_df = engineer_features(df)
    X_batch = engineered_df[MODEL_FEATURE_COLUMNS].copy()

    # 2. Transform with fitted ColumnTransformer
    X_trans = pipeline.transform(X_batch)

    # 3. Vectorized model inference
    preds = model.predict(X_trans).astype(int)
    try:
        probs = model.predict_proba(X_trans)[:, 1].astype(float)
    except Exception:
        probs = np.where(preds == 1, 1.0, 0.0)

    # 4. Construct result records
    records = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        row = df.iloc[i]
        tx_id = f"TX-B{start_tx_id + i:05d}"
        prob_val = float(prob)

        if prob_val < risk_threshold_low:
            risk_cat = "Low Risk"
        elif prob_val <= risk_threshold_high:
            risk_cat = "Medium Risk"
        else:
            risk_cat = "High Risk"

        pred_label = "FRAUDULENT" if pred == 1 else "LEGITIMATE"
        ts = str(row.get('trans_date_trans_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        records.append({
            "transaction_id": tx_id,
            "timestamp": ts,
            "amount": float(row.get('amt', 50.0)),
            "category": str(row.get('category', 'misc_pos')),
            "distance_km": float(engineered_df.iloc[i].get('distance_km', 10.0)),
            "city_pop": int(row.get('city_pop', 50000)),
            "hour": int(engineered_df.iloc[i].get('hour', 12)),
            "day_of_week": int(engineered_df.iloc[i].get('day_of_week', 2)),
            "is_weekend": int(engineered_df.iloc[i].get('is_weekend', 0)),
            "age": int(engineered_df.iloc[i].get('age', 40)),
            "prediction": pred_label,
            "fraud_probability": prob_val,
            "risk_category": risk_cat,
            "source": source
        })

    # Convert to display DataFrame
    results_df = pd.DataFrame([
        {
            "Transaction ID": r["transaction_id"],
            "Timestamp": r["timestamp"],
            "Amount ($)": r["amount"],
            "Category": r["category"],
            "Distance (km)": r["distance_km"],
            "Prediction": r["prediction"],
            "Fraud Probability": f"{r['fraud_probability'] * 100:.2f}%",
            "Risk Tier": r["risk_category"],
            "raw_pred": 1 if r["prediction"] == "FRAUDULENT" else 0,
            "raw_prob": r["fraud_probability"]
        }
        for r in records
    ])

    return results_df, records
