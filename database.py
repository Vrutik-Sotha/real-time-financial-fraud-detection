"""
SQLite Persistence Module for Financial Fraud Detection System
Provides persistent audit logging for analyzed transactions, live monitoring history,
bulk CSV ingestion, training run history, and persistent training dataset registry.
"""

import os
import sqlite3
from datetime import datetime
import pandas as pd

DEFAULT_DB_PATH = os.path.join("data", "monitoring.db")


def get_db_connection(db_path=DEFAULT_DB_PATH):
    """Establish a connection to the SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DEFAULT_DB_PATH):
    """
    Initializes the SQLite database tables:
    1. transaction_history: Stores audited transactions, predictions, probabilities, and sources.
    2. training_history: Stores model training and retraining benchmark logs.
    3. dataset_registry: Stores persistent metadata for registered training datasets.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transaction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            distance_km REAL NOT NULL,
            city_pop INTEGER NOT NULL,
            hour INTEGER NOT NULL,
            day_of_week INTEGER NOT NULL,
            is_weekend INTEGER NOT NULL,
            age INTEGER NOT NULL,
            prediction TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            risk_category TEXT NOT NULL,
            source TEXT DEFAULT 'Manual Input',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Safe schema migration for 'source' column
    cursor.execute("PRAGMA table_info(transaction_history)")
    columns = [col[1] for col in cursor.fetchall()]
    if "source" not in columns and len(columns) > 0:
        cursor.execute("ALTER TABLE transaction_history ADD COLUMN source TEXT DEFAULT 'Manual Input'")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT NOT NULL,
            train_records INTEGER NOT NULL,
            test_records INTEGER NOT NULL,
            accuracy REAL NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1_score REAL NOT NULL,
            roc_auc REAL,
            pr_auc REAL,
            notes TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL UNIQUE,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_records INTEGER NOT NULL,
            fraud_records INTEGER NOT NULL,
            legit_records INTEGER NOT NULL,
            file_size_kb REAL DEFAULT 0.0,
            is_active INTEGER DEFAULT 1
        )
    """)

    conn.commit()
    conn.close()


def save_transaction_record(record, db_path=DEFAULT_DB_PATH):
    """
    Saves an analyzed transaction record to SQLite.
    Returns the generated transaction_id.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO transaction_history (
            transaction_id, timestamp, amount, category, distance_km,
            city_pop, hour, day_of_week, is_weekend, age,
            prediction, fraud_probability, risk_category, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["transaction_id"],
        str(record.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
        float(record["amount"]),
        str(record["category"]),
        float(record["distance_km"]),
        int(record.get("city_pop", 50000)),
        int(record.get("hour", 12)),
        int(record.get("day_of_week", 2)),
        int(record.get("is_weekend", 0)),
        int(record.get("age", 40)),
        str(record["prediction"]),
        float(record["fraud_probability"]),
        str(record["risk_category"]),
        str(record.get("source", "Manual Input"))
    ))

    conn.commit()
    conn.close()
    return record["transaction_id"]


def save_bulk_transaction_records(records_list, db_path=DEFAULT_DB_PATH):
    """
    High-performance bulk insert of multiple analyzed transaction records.
    Returns count of records inserted.
    """
    if not records_list:
        return 0

    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    rows_to_insert = [
        (
            r["transaction_id"],
            str(r.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
            float(r["amount"]),
            str(r["category"]),
            float(r["distance_km"]),
            int(r.get("city_pop", 50000)),
            int(r.get("hour", 12)),
            int(r.get("day_of_week", 2)),
            int(r.get("is_weekend", 0)),
            int(r.get("age", 40)),
            str(r["prediction"]),
            float(r["fraud_probability"]),
            str(r["risk_category"]),
            str(r.get("source", "Bulk CSV Upload"))
        )
        for r in records_list
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO transaction_history (
            transaction_id, timestamp, amount, category, distance_km,
            city_pop, hour, day_of_week, is_weekend, age,
            prediction, fraud_probability, risk_category, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows_to_insert)

    conn.commit()
    conn.close()
    return len(records_list)


def get_monitoring_history(limit=2000, db_path=DEFAULT_DB_PATH):
    """
    Fetches monitored transactions ordered by newest first.
    Returns: pandas.DataFrame
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    query = """
        SELECT 
            transaction_id AS 'Transaction ID',
            timestamp AS 'Timestamp',
            amount AS 'Amount ($)',
            category AS 'Category',
            distance_km AS 'Distance (km)',
            prediction AS 'Prediction',
            fraud_probability AS 'Fraud Probability',
            risk_category AS 'Risk Tier',
            source AS 'Source',
            created_at AS 'Logged At'
        FROM transaction_history
        ORDER BY id DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def get_monitoring_stats(db_path=DEFAULT_DB_PATH):
    """
    Calculates summary KPIs from the SQLite monitoring database.
    Returns: dict
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM transaction_history")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transaction_history WHERE prediction = 'FRAUDULENT'")
    fraud = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transaction_history WHERE prediction = 'LEGITIMATE'")
    legit = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transaction_history WHERE risk_category = 'High Risk'")
    high_risk = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transaction_history WHERE risk_category = 'Medium Risk'")
    medium_risk = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transaction_history WHERE risk_category = 'Low Risk'")
    low_risk = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(fraud_probability) FROM transaction_history")
    avg_prob_row = cursor.fetchone()[0]
    avg_prob = float(avg_prob_row) if avg_prob_row is not None else 0.0

    # Source breakdown
    cursor.execute("SELECT COUNT(*) FROM transaction_history WHERE source LIKE '%CSV%'")
    csv_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transaction_history WHERE source LIKE '%Simulation%'")
    sim_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transaction_history WHERE source LIKE '%Manual%'")
    manual_count = cursor.fetchone()[0]

    conn.close()
    return {
        "total": total,
        "fraud": fraud,
        "legit": legit,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
        "avg_fraud_prob": avg_prob,
        "fraud_rate": (fraud / total * 100) if total > 0 else 0.0,
        "source_counts": {
            "Bulk CSV Upload": csv_count,
            "Live Simulation": sim_count,
            "Manual Input": manual_count
        }
    }


def clear_monitoring_history(db_path=DEFAULT_DB_PATH):
    """Clears all records in the transaction_history table."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transaction_history")
    conn.commit()
    conn.close()


def log_training_run(run_data, db_path=DEFAULT_DB_PATH):
    """Logs model training run metadata into SQLite."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO training_history (
            model_name, train_records, test_records, accuracy,
            precision, recall, f1_score, roc_auc, pr_auc, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_data["model_name"],
        run_data["train_records"],
        run_data["test_records"],
        run_data["accuracy"],
        run_data["precision"],
        run_data["recall"],
        run_data["f1_score"],
        run_data.get("roc_auc"),
        run_data.get("pr_auc"),
        run_data.get("notes", "")
    ))
    conn.commit()
    conn.close()


def register_dataset(filename, filepath, total_records, fraud_records, legit_records, file_size_kb=0.0, is_active=True, db_path=DEFAULT_DB_PATH):
    """
    Registers or updates a dataset in the dataset_registry table.
    If is_active is True, marks all previous entries as inactive.
    Returns the registered dataset ID.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    if is_active:
        cursor.execute("UPDATE dataset_registry SET is_active = 0")

    cursor.execute("""
        INSERT INTO dataset_registry (
            filename, filepath, total_records, fraud_records, legit_records, file_size_kb, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(filepath) DO UPDATE SET
            filename=excluded.filename,
            total_records=excluded.total_records,
            fraud_records=excluded.fraud_records,
            legit_records=excluded.legit_records,
            file_size_kb=excluded.file_size_kb,
            is_active=excluded.is_active,
            upload_timestamp=CURRENT_TIMESTAMP
    """, (
        str(filename),
        str(filepath),
        int(total_records),
        int(fraud_records),
        int(legit_records),
        float(file_size_kb),
        1 if is_active else 0
    ))

    conn.commit()
    cursor.execute("SELECT id FROM dataset_registry WHERE filepath = ?", (str(filepath),))
    row = cursor.fetchone()
    inserted_id = row[0] if row else None
    conn.close()
    return inserted_id


def get_active_dataset(db_path=DEFAULT_DB_PATH):
    """
    Retrieves the currently active registered dataset metadata.
    Returns a dictionary or None if no dataset is active.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, filepath, upload_timestamp, total_records, fraud_records, legit_records, file_size_kb, is_active
        FROM dataset_registry
        WHERE is_active = 1
        ORDER BY id DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row["id"],
            "filename": row["filename"],
            "filepath": row["filepath"],
            "upload_timestamp": row["upload_timestamp"],
            "total_records": row["total_records"],
            "fraud_records": row["fraud_records"],
            "legit_records": row["legit_records"],
            "fraud_rate": (row["fraud_records"] / row["total_records"] * 100) if row["total_records"] > 0 else 0.0,
            "file_size_kb": row["file_size_kb"],
            "is_active": bool(row["is_active"])
        }
    return None


def get_registered_datasets(db_path=DEFAULT_DB_PATH):
    """
    Retrieves all registered datasets in descending order of registration.
    Returns: pandas.DataFrame
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    query = """
        SELECT 
            id AS 'ID',
            filename AS 'Filename',
            filepath AS 'Filepath',
            upload_timestamp AS 'Registered At',
            total_records AS 'Total Records',
            fraud_records AS 'Fraud Records',
            legit_records AS 'Legit Records',
            file_size_kb AS 'Size (KB)',
            is_active AS 'Is Active'
        FROM dataset_registry
        ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def set_active_dataset(dataset_id, db_path=DEFAULT_DB_PATH):
    """
    Sets the specified dataset ID as the active dataset, deactivating all others.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE dataset_registry SET is_active = 0")
    cursor.execute("UPDATE dataset_registry SET is_active = 1 WHERE id = ?", (int(dataset_id),))
    conn.commit()
    conn.close()
    return True


def ensure_default_dataset_registered(data_dir="data", db_path=DEFAULT_DB_PATH):
    """
    Ensures that at least one dataset is registered in the database.
    If no dataset is registered and fraudTrain.csv exists, registers it automatically.
    """
    active = get_active_dataset(db_path=db_path)
    if active is not None and os.path.exists(active["filepath"]):
        return active

    train_path = os.path.join(data_dir, "fraudTrain.csv")
    if os.path.exists(train_path):
        try:
            df = pd.read_csv(train_path)
            total = len(df)
            fraud = int(df['is_fraud'].sum()) if 'is_fraud' in df.columns else 0
            legit = total - fraud
            size_kb = os.path.getsize(train_path) / 1024.0

            register_dataset(
                filename="fraudTrain.csv",
                filepath=os.path.abspath(train_path),
                total_records=total,
                fraud_records=fraud,
                legit_records=legit,
                file_size_kb=size_kb,
                is_active=True,
                db_path=db_path
            )
            return get_active_dataset(db_path=db_path)
        except Exception:
            pass
    return None
