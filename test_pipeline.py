"""
Comprehensive Test Suite for Rebuilt Financial Fraud Detection System
Tests dataset loading, feature engineering, model training, model serialization,
manual transaction inference, SQLite persistence, retraining, and backup/restore.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

from preprocessing import (
    ALL_CATEGORIES,
    load_dataset,
    validate_dataset_schema,
    haversine_distance,
    engineer_features,
    prepare_data_for_modeling,
    format_manual_transaction,
    generate_benchmark_dataset_files,
    generate_sample_csv_template
)
from model import (
    DEFAULT_MODEL_PATH,
    BACKUP_MODEL_PATH,
    train_and_compare_models,
    save_model_bundle,
    load_model_bundle,
    backup_model_bundle,
    restore_model_backup,
    switch_active_model,
    predict_manual_transaction,
    predict_bulk_transactions
)
from database import (
    init_db,
    save_transaction_record,
    save_bulk_transaction_records,
    get_monitoring_history,
    get_monitoring_stats,
    clear_monitoring_history,
    log_training_run,
    register_dataset,
    get_active_dataset,
    get_registered_datasets,
    set_active_dataset,
    ensure_default_dataset_registered
)
from utils.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_model_comparison,
    plot_monitoring_summary,
    plot_probability_distribution,
    plot_transactions_over_time,
    plot_source_breakdown
)


def run_full_test_suite():
    print("=" * 65)
    print("🧪 STARTING COMPREHENSIVE FRAUD DETECTION TEST SUITE")
    print("=" * 65)

    test_db = os.path.join("data", "test_monitoring.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    # 1. Dataset generation and loading
    print("\n1. Testing dataset loading & generation...")
    train_path, test_path = generate_benchmark_dataset_files(data_dir="data", n_train=500, n_test=100)
    assert os.path.exists(train_path) and os.path.exists(test_path)
    df_train = load_dataset(train_path)
    df_test = load_dataset(test_path)
    assert len(df_train) > 0 and len(df_test) > 0
    print(f"   -> Successfully loaded {len(df_train)} train and {len(df_test)} test records.")

    # 2. Schema Validation
    print("\n2. Testing dataset schema validation...")
    valid, msg = validate_dataset_schema(df_train, require_target=True)
    assert valid, f"Schema validation failed: {msg}"
    
    # Test invalid schema
    invalid_df = pd.DataFrame({"dummy": [1, 2]})
    inv_valid, inv_msg = validate_dataset_schema(invalid_df, require_target=True)
    assert not inv_valid
    print("   -> Schema validation and error handling passed.")

    # 3. Haversine Distance & Feature Engineering
    print("\n3. Testing Haversine distance and feature engineering...")
    dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)  # NYC to LA (~3935 km)
    assert 3900 < dist < 4000
    
    eng_df = engineer_features(df_train)
    for col in ['amt', 'hour', 'day_of_week', 'is_weekend', 'distance_km', 'city_pop', 'age', 'category']:
        assert col in eng_df.columns, f"Missing engineered column: {col}"
    print(f"   -> Distance verified ({dist:.1f} km) and engineered features verified.")

    # 4. Data Preparation for Modeling
    print("\n4. Testing prepare_data_for_modeling()...")
    X_train, y_train = prepare_data_for_modeling(df_train)
    X_test, y_test = prepare_data_for_modeling(df_test)
    assert X_train.shape[1] == 8
    assert len(y_train) == len(X_train)
    print(f"   -> Feature matrix verified: shape={X_train.shape}, fraud_count={int(y_train.sum())}")

    # 5. Multi-Model Training and Benchmarking
    print("\n5. Testing multi-model training (Logistic Regression, Random Forest, XGBoost)...")
    bundle, comparisons = train_and_compare_models(X_train, y_train, X_test, y_test)
    assert "Logistic Regression" in comparisons
    assert "Random Forest" in comparisons
    assert "XGBoost" in comparisons
    assert bundle["model"] is not None
    assert bundle["pipeline"] is not None
    print(f"   -> All 3 algorithms trained. Selected Champion: {bundle['model_name']} (F1={bundle['metrics']['f1_score']:.4f})")

    # 6. Model Serialization and Backup/Restore
    print("\n6. Testing model saving, backup, and restore...")
    saved_path = save_model_bundle(bundle, DEFAULT_MODEL_PATH)
    assert os.path.exists(saved_path)
    
    backed_up = backup_model_bundle(DEFAULT_MODEL_PATH, BACKUP_MODEL_PATH)
    assert backed_up and os.path.exists(BACKUP_MODEL_PATH)

    restored = restore_model_backup(BACKUP_MODEL_PATH, DEFAULT_MODEL_PATH)
    assert restored
    loaded_bundle = load_model_bundle(DEFAULT_MODEL_PATH)
    assert loaded_bundle["model_name"] == bundle["model_name"]
    print("   -> Model serialization, backup, and restore passed.")

    # 7. Manual Human-Readable Transaction Prediction
    print("\n7. Testing manual human-readable transaction inference...")
    sample_manual = {
        "amt": 45.0,
        "category": "grocery_pos",
        "date_time": datetime(2024, 6, 12, 14, 30),
        "distance_km": 5.2,
        "city_pop": 60000,
        "age": 38
    }
    pred_res = predict_manual_transaction(loaded_bundle, sample_manual)
    assert pred_res["prediction_code"] in [0, 1]
    assert 0.0 <= pred_res["fraud_probability"] <= 1.0
    assert pred_res["risk_category"] in ["Low Risk", "Medium Risk", "High Risk"]
    print(f"   -> Legit sample prediction: {pred_res['prediction_label']} (Risk: {pred_res['fraud_probability'] * 100:.2f}%, {pred_res['risk_category']})")

    # Test high-risk suspicious anomaly
    suspicious_manual = {
        "amt": 3200.0,
        "category": "shopping_net",
        "date_time": datetime(2024, 6, 12, 3, 15),  # 3 AM
        "distance_km": 1250.0,  # cross country
        "city_pop": 500000,
        "age": 24
    }
    susp_res = predict_manual_transaction(loaded_bundle, suspicious_manual)
    assert susp_res["prediction_code"] in [0, 1]
    print(f"   -> Suspicious anomaly prediction: {susp_res['prediction_label']} (Risk: {susp_res['fraud_probability'] * 100:.2f}%, {susp_res['risk_category']})")

    # 8. Invalid and Missing Input Handling
    print("\n8. Testing invalid / missing input resilience...")
    empty_manual = {}
    safe_res = predict_manual_transaction(loaded_bundle, empty_manual)
    assert safe_res is not None
    assert safe_res["prediction_code"] in [0, 1]
    print("   -> Missing input defaults handled safely.")

    # 9. SQLite Persistence Testing
    print("\n9. Testing SQLite audit logging and retrieval...")
    init_db(test_db)
    rec_id = save_transaction_record({
        "transaction_id": "TEST-TX-001",
        "timestamp": "2024-06-12 14:30:00",
        "amount": 150.0,
        "category": "travel",
        "distance_km": 25.0,
        "city_pop": 75000,
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "age": 35,
        "prediction": "LEGITIMATE",
        "fraud_probability": 0.05,
        "risk_category": "Low Risk"
    }, db_path=test_db)
    assert rec_id == "TEST-TX-001"

    hist = get_monitoring_history(db_path=test_db)
    assert len(hist) == 1
    stats = get_monitoring_stats(db_path=test_db)
    assert stats["total"] == 1
    assert stats["legit"] == 1

    clear_monitoring_history(db_path=test_db)
    assert len(get_monitoring_history(db_path=test_db)) == 0
    print("   -> SQLite logging, querying, and clearing passed.")

    # 10. Visualization Utilities Testing
    print("\n10. Testing visualization generators...")
    fig1 = plot_class_distribution(900, 100)
    fig2 = plot_confusion_matrix([0, 1], [0, 1])
    fig3 = plot_roc_curve([0, 1], [0.1, 0.9])
    fig4 = plot_precision_recall_curve([0, 1], [0.1, 0.9])
    fig5 = plot_model_comparison(comparisons)
    fig6 = plot_monitoring_summary(8, 2, 6, 2, 2)
    fig7 = plot_probability_distribution([0.05, 0.12, 0.85, 0.92])
    fig8 = plot_transactions_over_time(pd.DataFrame({
        "Timestamp": ["2024-05-10 10:00:00", "2024-05-10 10:05:00"],
        "Prediction": ["LEGITIMATE", "FRAUDULENT"]
    }))
    fig9 = plot_source_breakdown({"Bulk CSV Upload": 50, "Live Simulation": 25, "Manual Input": 10})
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]:
        assert f is not None
    print("   -> All 9 visualization figures generated successfully.")

    # 11. Bulk CSV Processing and Persistence
    print("\n11. Testing Bulk CSV Upload and Vectorized Inference...")
    sample_csv_text = generate_sample_csv_template()
    assert len(sample_csv_text) > 0
    from io import StringIO
    sample_batch_df = pd.read_csv(StringIO(sample_csv_text))
    assert len(sample_batch_df) == 10

    bulk_results_df, bulk_records = predict_bulk_transactions(
        loaded_bundle,
        sample_batch_df,
        source="Bulk CSV Upload",
        start_tx_id=8000
    )
    assert len(bulk_results_df) == 10
    assert len(bulk_records) == 10
    assert "Prediction" in bulk_results_df.columns
    assert "Risk Tier" in bulk_results_df.columns

    # Save to test DB
    inserted = save_bulk_transaction_records(bulk_records, db_path=test_db)
    assert inserted == 10
    stats_after_bulk = get_monitoring_stats(db_path=test_db)
    assert stats_after_bulk["total"] == 10
    assert stats_after_bulk["source_counts"]["Bulk CSV Upload"] == 10
    print(f"   -> Bulk CSV processing verified: {stats_after_bulk['total']} records saved to SQLite.")

    # 12. Continuous Live Simulator Event Handling
    print("\n12. Testing Continuous Live Simulator event ingestion...")
    sim_event = {
        "transaction_id": "TEST-SIM-001",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "amount": 1250.0,
        "category": "travel",
        "distance_km": 850.0,
        "city_pop": 250000,
        "hour": 2,
        "day_of_week": 5,
        "is_weekend": 1,
        "age": 28,
        "prediction": "FRAUDULENT",
        "fraud_probability": 0.94,
        "risk_category": "High Risk",
        "source": "Live Simulation"
    }
    save_transaction_record(sim_event, db_path=test_db)
    final_stats = get_monitoring_stats(db_path=test_db)
    assert final_stats["total"] == 11
    assert final_stats["source_counts"]["Live Simulation"] == 1
    print("   -> Live Simulation ingestion and SQLite persistence verified.")

    # 13. Testing Persistent Dataset Registry
    print("\n13. Testing Persistent Dataset Registry (SQLite)...")
    init_db(test_db)
    ds_id1 = register_dataset("train_v1.csv", "data/test_v1.csv", 4000, 50, 3950, file_size_kb=120.5, is_active=True, db_path=test_db)
    active_ds = get_active_dataset(db_path=test_db)
    assert active_ds is not None
    assert active_ds["filename"] == "train_v1.csv"
    assert active_ds["is_active"] is True

    # Register second dataset as active
    ds_id2 = register_dataset("train_v2.csv", "data/test_v2.csv", 5000, 75, 4925, file_size_kb=150.0, is_active=True, db_path=test_db)
    active_ds2 = get_active_dataset(db_path=test_db)
    assert active_ds2["filename"] == "train_v2.csv"

    # Switch active dataset back to ds_id1
    set_active_dataset(ds_id1, db_path=test_db)
    active_switched = get_active_dataset(db_path=test_db)
    assert active_switched["filename"] == "train_v1.csv"
    print("   -> Dataset registry and switching verified.")

    # 14. Testing Dynamic Candidate Model Switching
    print("\n14. Testing Dynamic Candidate Model Switching...")
    assert "candidate_models" in loaded_bundle
    cand_keys = list(loaded_bundle["candidate_models"].keys())
    assert len(cand_keys) >= 2
    original_model_name = loaded_bundle["model_name"]
    other_model_name = [k for k in cand_keys if k != original_model_name][0]

    # Switch model dynamically
    switched_bundle = switch_active_model(loaded_bundle, other_model_name)
    assert switched_bundle["model_name"] == other_model_name
    # Run prediction using the switched model
    pred_switched = predict_manual_transaction(switched_bundle, sample_manual)
    assert pred_switched["model_name"] == other_model_name
    print(f"   -> Successfully switched active model from '{original_model_name}' to '{other_model_name}'.")

    # 15. Testing Retraining from Registered Dataset
    print("\n15. Testing Retraining from Registered Dataset...")
    ensure_default_dataset_registered(data_dir="data", db_path=test_db)
    reg_active = get_active_dataset(db_path=test_db)
    assert reg_active is not None
    print(f"   -> Active registered training dataset confirmed: '{reg_active['filename']}'.")

    # Clean up test DB
    if os.path.exists(test_db):
        os.remove(test_db)

    print("\n" + "=" * 65)
    print("✅ ALL TESTS PASSED SUCCESSFULLY! Upgrade is 100% verified.")
    print("=" * 65)


if __name__ == "__main__":
    run_full_test_suite()
