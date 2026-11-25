"""
Model Training and Comparison CLI Script
Trains and evaluates Logistic Regression, Random Forest, and XGBoost on the Kartik2112 dataset,
selects the top-performing model, saves the bundle to models/fraud_model.pkl, and logs metrics to SQLite.
"""

import os
import pandas as pd
from preprocessing import load_dataset, prepare_data_for_modeling, generate_benchmark_dataset_files
from model import train_and_compare_models, save_model_bundle, backup_model_bundle, DEFAULT_MODEL_PATH
from database import init_db, log_training_run


def run_training_pipeline(train_csv=None, test_csv=None):
    print("=" * 65)
    print("🚀 FINANCIAL FRAUD DETECTION SYSTEM: MODEL TRAINING PIPELINE")
    print("=" * 65)

    # 1. Dataset availability check
    if train_csv is None or not os.path.exists(train_csv):
        train_path, test_path = generate_benchmark_dataset_files(data_dir="data")
    else:
        train_path = train_csv
        test_path = test_csv if (test_csv and os.path.exists(test_csv)) else train_csv

    print(f"📁 Training Dataset: {train_path}")
    print(f"📁 Evaluation Dataset: {test_path}")

    # 2. Load Datasets
    print("\n⏳ Ingesting and validating transaction records...")
    raw_train_df = load_dataset(train_path)
    raw_test_df = load_dataset(test_path)

    print(f"   -> Train records: {len(raw_train_df):,}")
    print(f"   -> Test records: {len(raw_test_df):,}")

    # 3. Feature Engineering & Preprocessing Preparation
    print("\n⏳ Engineering human-readable features (Distance, Time, Demographics)...")
    X_train, y_train = prepare_data_for_modeling(raw_train_df)
    X_test, y_test = prepare_data_for_modeling(raw_test_df)

    print(f"   -> Engineered Feature Matrix Shape: {X_train.shape}")
    print(f"   -> Train Fraud Count: {int(y_train.sum()):,} ({y_train.sum() / len(y_train) * 100:.2f}%)")
    print(f"   -> Test Fraud Count:  {int(y_test.sum()):,} ({y_test.sum() / len(y_test) * 100:.2f}%)")

    # 4. Train and Compare Models
    print("\n⚔️ Training and benchmarking candidate algorithms...")
    print("   1. Logistic Regression (Linear with Balanced Class Weights)")
    print("   2. Random Forest (Bagging Ensemble with Balanced Subsamples)")
    print("   3. XGBoost (Gradient Boosted Decision Trees with Scale Weighting)")

    best_bundle, comparison = train_and_compare_models(X_train, y_train, X_test, y_test)

    # 5. Display Benchmark Summary Table
    print("\n" + "=" * 65)
    print(f"{'MODEL':<22} | {'ACC':<6} | {'PREC':<6} | {'REC':<6} | {'F1':<6} | {'ROC-AUC':<7}")
    print("-" * 65)
    for name, m in comparison.items():
        roc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] else "N/A"
        print(f"{name:<22} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {roc_str:<7}")
    print("=" * 65)

    print(f"\n🏆 Selected Champion Model: {best_bundle['model_name']} (F1-Score: {best_bundle['metrics']['f1_score']:.4f})")

    # 6. Safety Backup & Save Model
    if os.path.exists(DEFAULT_MODEL_PATH):
        backup_model_bundle(DEFAULT_MODEL_PATH)
        print("💾 Previous working model backed up to models/fraud_model_backup.pkl")

    saved_file = save_model_bundle(best_bundle, DEFAULT_MODEL_PATH)
    print(f"💾 Champion model and pipeline saved to: {saved_file}")

    # 7. Log to SQLite
    init_db()
    log_training_run({
        "model_name": best_bundle["model_name"],
        "train_records": len(X_train),
        "test_records": len(X_test),
        "accuracy": best_bundle["metrics"]["accuracy"],
        "precision": best_bundle["metrics"]["precision"],
        "recall": best_bundle["metrics"]["recall"],
        "f1_score": best_bundle["metrics"]["f1_score"],
        "roc_auc": best_bundle["metrics"]["roc_auc"],
        "pr_auc": best_bundle["metrics"]["pr_auc"],
        "notes": f"Trained with {best_bundle['model_name']} on Kartik2112 dataset"
    })
    print("📊 Training metadata and metrics logged to SQLite audit database.")
    print("=" * 65)
    print("✅ Training Pipeline Complete! Ready for Streamlit Inference.\n")
    return best_bundle


if __name__ == "__main__":
    run_training_pipeline()
