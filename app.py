"""
Real-Time Financial Fraud Detection System - Enterprise FinTech Web Application
Built with Scikit-learn, XGBoost, Pandas, SQLite, and Streamlit.
Features human-readable transaction inputs, multi-model benchmarking,
dynamic model switching, persistent dataset management, and enterprise audit logging.
"""

import os
import tempfile
# Ensure Matplotlib uses a writable temporary directory on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, time, date
import random
import time as pytime

# Project modules
from preprocessing import (
    ALL_CATEGORIES,
    load_dataset,
    validate_dataset_schema,
    prepare_data_for_modeling,
    generate_benchmark_dataset_files,
    format_manual_transaction,
    generate_sample_csv_template,
    persist_and_register_dataset
)
from model import (
    DEFAULT_MODEL_PATH,
    BACKUP_MODEL_PATH,
    load_model_bundle,
    save_model_bundle,
    backup_model_bundle,
    restore_model_backup,
    train_and_compare_models,
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

# Page Setup
st.set_page_config(
    page_title="FinGuard | Real-Time Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# FINTECH CUSTOM CSS THEME
# =============================================================================
FINTECH_CSS = """
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Top Brand Bar */
    .brand-header {
        background: linear-gradient(135deg, #0a192f 0%, #172a45 100%);
        padding: 24px 28px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .brand-badge {
        display: inline-block;
        background: rgba(0, 180, 216, 0.2);
        color: #00b4d8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 6px;
        margin-bottom: 8px;
        border: 1px solid rgba(0, 180, 216, 0.4);
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .brand-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        line-height: 1.5;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #00b4d8;
        box-shadow: 0 4px 12px rgba(0, 180, 216, 0.12);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Alert and Card containers */
    .fintech-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
    }
    .fraud-banner {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #f87171;
        border-left: 6px solid #ef4444;
        border-radius: 10px;
        padding: 16px 20px;
        color: #991b1b;
        margin-bottom: 16px;
    }
    .legit-banner {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-left: 6px solid #10b981;
        border-radius: 10px;
        padding: 16px 20px;
        color: #166534;
        margin-bottom: 16px;
    }
    .info-callout {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #00b4d8;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.9rem;
        color: #334155;
        margin-bottom: 16px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.92rem;
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        color: #0077b6 !important;
        border-bottom: 3px solid #00b4d8 !important;
    }

    /* Buttons */
    .stButton>button {
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
        border: none;
        box-shadow: 0 2px 6px rgba(0, 180, 216, 0.25);
    }
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #023e8a 0%, #0096c7 100%);
        box-shadow: 0 4px 12px rgba(0, 180, 216, 0.35);
    }
</style>
"""


def init_app_state():
    """Initializes Streamlit session state and persistent storage."""
    init_db()
    ensure_default_dataset_registered()

    if "model_bundle" not in st.session_state:
        if os.path.exists(DEFAULT_MODEL_PATH):
            try:
                st.session_state.model_bundle = load_model_bundle(DEFAULT_MODEL_PATH)
            except Exception:
                st.session_state.model_bundle = None
        else:
            st.session_state.model_bundle = None

    if "tx_counter" not in st.session_state:
        st.session_state.tx_counter = 5001

    if "risk_threshold_low" not in st.session_state:
        st.session_state.risk_threshold_low = 0.30
    if "risk_threshold_high" not in st.session_state:
        st.session_state.risk_threshold_high = 0.70

    if "last_manual_prediction" not in st.session_state:
        st.session_state.last_manual_prediction = None


def render_brand_header():
    """Renders the top enterprise FinTech brand banner."""
    st.markdown(FINTECH_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="brand-header">
        <div class="brand-badge">⚡ FinTech AI Engine • Production Grade</div>
        <h1 class="brand-title">Real-Time Financial Fraud Detection Platform</h1>
        <div class="brand-subtitle">
            An enterprise-grade fraud intelligence system trained on human-readable transaction attributes
            (Amount, Merchant Category, Geo-distance, Time, Demographics) with multi-model benchmarking
            (Logistic Regression, Random Forest, XGBoost) and persistent SQLite audit tracking.
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    init_app_state()
    render_brand_header()

    # =========================================================================
    # SIDEBAR: MODEL & PERSISTENT DATASET MANAGEMENT
    # =========================================================================
    st.sidebar.markdown("### 🛡️ System Controls")

    # 1. Model Selection & Active Model Section
    bundle = st.session_state.model_bundle
    if bundle is not None:
        candidate_models = bundle.get("candidate_models", {})
        active_model_name = bundle.get("model_name", "Classifier")

        st.sidebar.markdown("#### 🤖 Active ML Model")

        # If multiple candidate models are available, show a dropdown to switch dynamically
        if len(candidate_models) > 1:
            candidate_names = list(candidate_models.keys())
            current_idx = candidate_names.index(active_model_name) if active_model_name in candidate_names else 0
            
            selected_model = st.sidebar.selectbox(
                "Select Active Champion Model:",
                options=candidate_names,
                index=current_idx,
                help="Switch between trained algorithms dynamically. The chosen model will immediately perform all subsequent predictions."
            )
            if selected_model != active_model_name:
                st.session_state.model_bundle = switch_active_model(bundle, selected_model)
                st.sidebar.success(f"Switched active model to: **{selected_model}**")
                st.rerun()
        else:
            # If only one model exists, display clean active model info
            st.sidebar.info(f"Active Model: **{active_model_name}**")

        st.sidebar.caption(
            f"📅 Trained: `{bundle.get('trained_at', 'Pretrained')}`\n\n"
            f"📊 Training Samples: `{bundle.get('train_size', 0):,}` | Test: `{bundle.get('test_size', 0):,}`"
        )
    else:
        st.sidebar.warning("⚠️ No trained model loaded! Train a model below.")

    st.sidebar.markdown("---")

    # 2. Persistent Training Dataset Management Section
    st.sidebar.markdown("#### 📁 Registered Training Dataset")
    active_dataset = get_active_dataset()

    if active_dataset:
        st.sidebar.markdown(f"**Filename:** `{active_dataset['filename']}`")
        st.sidebar.caption(
            f"• Records: **{active_dataset['total_records']:,}**\n\n"
            f"• Fraud Cases: **{active_dataset['fraud_records']:,}** ({active_dataset['fraud_rate']:.2f}%)\n\n"
            f"• Registered: `{active_dataset['upload_timestamp'][:19]}`"
        )

        # Button: Retrain directly from registered dataset without re-uploading
        if st.sidebar.button("🚀 Retrain Model on Registered Dataset", use_container_width=True):
            with st.spinner("Retraining & benchmarking models on registered dataset..."):
                try:
                    df_train = load_dataset(active_dataset["filepath"])
                    # Use test set for clean out-of-sample evaluation
                    train_p, test_p = generate_benchmark_dataset_files()
                    df_test = load_dataset(test_p)

                    X_tr, y_tr = prepare_data_for_modeling(df_train)
                    X_te, y_te = prepare_data_for_modeling(df_test)

                    # Backup current model
                    if os.path.exists(DEFAULT_MODEL_PATH):
                        backup_model_bundle(DEFAULT_MODEL_PATH, BACKUP_MODEL_PATH)

                    new_bundle, comparisons = train_and_compare_models(X_tr, y_tr, X_te, y_te)
                    save_model_bundle(new_bundle, DEFAULT_MODEL_PATH)
                    st.session_state.model_bundle = new_bundle

                    # Log training run into SQLite
                    log_training_run({
                        "model_name": new_bundle["model_name"],
                        "train_records": len(X_tr),
                        "test_records": len(X_te),
                        "accuracy": new_bundle["metrics"]["accuracy"],
                        "precision": new_bundle["metrics"]["precision"],
                        "recall": new_bundle["metrics"]["recall"],
                        "f1_score": new_bundle["metrics"]["f1_score"],
                        "roc_auc": new_bundle["metrics"]["roc_auc"],
                        "pr_auc": new_bundle["metrics"]["pr_auc"],
                        "notes": f"Retrained on registered dataset: {active_dataset['filename']}"
                    })

                    st.sidebar.success(f"Retrained successfully! Champion: **{new_bundle['model_name']}**")
                    st.rerun()
                except Exception as ex:
                    st.sidebar.error(f"Retraining failed: {ex}")
    else:
        st.sidebar.info("No active dataset registered yet. Upload or register one below.")

    # Collapsible: Replace / Upload New Dataset
    with st.sidebar.expander("📤 Register / Replace Dataset", expanded=False):
        st.caption("Upload a new CSV dataset (`amt`, `category`, `trans_date_trans_time`, `is_fraud`).")
        uploaded_new_csv = st.file_uploader("Upload labelled CSV", type=["csv"], key="new_dataset_uploader")

        if uploaded_new_csv is not None:
            if st.button("💾 Validate & Register Dataset", use_container_width=True):
                with st.spinner("Validating and storing dataset..."):
                    success, result = persist_and_register_dataset(uploaded_new_csv, uploaded_new_csv.name)
                    if success:
                        st.success(f"Registered `{result['filename']}` ({result['total_records']:,} rows) as active!")
                        st.rerun()
                    else:
                        st.error(f"Failed to register dataset: {result}")

    # Model Backup & Restore Section
    if os.path.exists(BACKUP_MODEL_PATH):
        with st.sidebar.expander("⏮️ Model Rollback", expanded=False):
            st.caption("A previous model backup is available on disk.")
            if st.button("Restore Previous Model Backup", use_container_width=True):
                if restore_model_backup(BACKUP_MODEL_PATH, DEFAULT_MODEL_PATH):
                    st.session_state.model_bundle = load_model_bundle(DEFAULT_MODEL_PATH)
                    st.success("Restored previous model from backup!")
                    st.rerun()

    # Decision Risk Policy Thresholds
    with st.sidebar.expander("⚖️ Decision Risk Policy Thresholds", expanded=False):
        st.session_state.risk_threshold_low = st.slider(
            "Low Risk Threshold (< X)", 0.05, 0.50, st.session_state.risk_threshold_low, 0.05
        )
        st.session_state.risk_threshold_high = st.slider(
            "High Risk Threshold (> Y)", 0.50, 0.95, st.session_state.risk_threshold_high, 0.05
        )
        st.caption("Demonstration risk thresholds for tiering fraud probabilities into Low, Medium, and High Risk.")

    # =========================================================================
    # MAIN FOUR-PAGE TABBED INTERFACE
    # =========================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 1. Dashboard",
        "📈 2. Model Performance",
        "🔍 3. Transaction Prediction",
        "📡 4. Live Transaction Monitoring"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: DASHBOARD
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Transaction Intelligence Overview")

        # Load benchmark stats
        train_path, test_path = generate_benchmark_dataset_files()
        try:
            train_df = load_dataset(train_path)
            total_records = len(train_df)
            fraud_count = int(train_df['is_fraud'].sum())
            legit_count = total_records - fraud_count
            fraud_rate = (fraud_count / total_records * 100) if total_records > 0 else 0.0
        except Exception:
            total_records, fraud_count, legit_count, fraud_rate = 0, 0, 0, 0.0

        # SQLite persistent monitoring stats
        mon_stats = get_monitoring_stats()

        # Top 4 KPI Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Training Dataset Records", f"{total_records:,}")
        k2.metric("Dataset Fraud Cases", f"{fraud_count:,}", f"{fraud_rate:.2f}% baseline", delta_color="inverse")
        k3.metric("Live Monitored (SQLite)", f"{mon_stats['total']:,}")
        k4.metric("Live Flagged Frauds", f"{mon_stats['fraud']:,}", f"{mon_stats['fraud_rate']:.2f}% rate", delta_color="inverse")

        st.write("")
        c_left, c_right = st.columns([1.1, 1])

        with c_left:
            st.markdown("#### Class Distribution (Training Dataset)")
            if total_records > 0:
                fig_class = plot_class_distribution(legit_count, fraud_count)
                st.pyplot(fig_class)
                plt.close(fig_class)
            else:
                st.info("Dataset loading...")

        with c_right:
            st.markdown("#### Kaggle Kartik2112 Dataset Architecture")
            st.markdown("""
            <div class="info-callout">
                <strong>💡 Human-Readable Financial Attributes:</strong><br>
                This application is trained on genuine transaction attributes instead of anonymized mathematical PCA coordinates ($V_1$–$V_{28}$).
                <ul>
                    <li><strong>Monetary:</strong> Transaction Amount (USD).</li>
                    <li><strong>Merchant:</strong> Category (grocery, dining, online, travel, etc.).</li>
                    <li><strong>Geographic:</strong> Customer-to-Merchant Great Circle Distance (km).</li>
                    <li><strong>Temporal:</strong> Hour of day, Day of week, Weekend indicator.</li>
                    <li><strong>Demographic:</strong> Customer Age and City Population.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Administrative status card
            st.markdown("#### Administrative Status Summary")
            active_ds = get_active_dataset()
            ds_name = active_ds["filename"] if active_ds else "fraudTrain.csv (Baseline)"
            ds_rows = f"{active_ds['total_records']:,}" if active_ds else f"{total_records:,}"
            champion_algo = bundle.get("model_name", "None") if bundle else "None"
            backup_status = "Available ✅" if os.path.exists(BACKUP_MODEL_PATH) else "None ⚪"

            st.markdown(f"""
            - **Registered Training Dataset:** `{ds_name}` ({ds_rows} rows)
            - **Active Production Model:** `{champion_algo}`
            - **Model Safety Backup:** `{backup_status}`
            - **Audit Persistence Engine:** `SQLite (data/monitoring.db)`
            """)

        st.markdown("---")
        st.markdown("#### Recent Live Monitored Transactions (Persistent Audit Log)")
        recent_audit = get_monitoring_history(limit=5)
        if len(recent_audit) > 0:
            st.dataframe(recent_audit, use_container_width=True)
        else:
            st.caption("No monitored transactions logged yet. Test manual input in Tab 3 or Live Monitoring in Tab 4.")

    # -------------------------------------------------------------------------
    # TAB 2: MODEL PERFORMANCE & BENCHMARK COMPARISON
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Model Benchmarking & Diagnostic Metrics")

        if bundle is None:
            st.warning("⚠️ No model bundle found. Please run training from the sidebar.")
        else:
            best_name = bundle.get("model_name", "Best Classifier")
            all_comp = bundle.get("all_comparisons", {})

            st.markdown(f"""
            <div class="info-callout">
                🏆 <strong>Active Champion Algorithm:</strong> <code>{best_name}</code><br>
                Evaluated against held-out test data (<code>fraudTest.csv</code>). 
                You can switch between candidate models dynamically in the sidebar without retraining.
            </div>
            """, unsafe_allow_html=True)

            # Model comparison table
            if all_comp:
                st.markdown("#### ⚔️ Candidate Algorithm Comparison Table")
                comp_rows = []
                for m_name, m_dict in all_comp.items():
                    comp_rows.append({
                        "Algorithm": m_name,
                        "Accuracy": f"{m_dict['accuracy'] * 100:.2f}%",
                        "Precision": f"{m_dict['precision'] * 100:.2f}%",
                        "Recall": f"{m_dict['recall'] * 100:.2f}%",
                        "F1-Score": f"{m_dict['f1_score']:.4f}",
                        "ROC-AUC": f"{m_dict['roc_auc']:.4f}" if m_dict.get('roc_auc') else "N/A",
                        "PR-AUC": f"{m_dict['pr_auc']:.4f}" if m_dict.get('pr_auc') else "N/A"
                    })
                comp_df = pd.DataFrame(comp_rows)
                st.dataframe(comp_df, use_container_width=True)

                # Comparison Bar Chart
                fig_comp = plot_model_comparison(all_comp)
                st.pyplot(fig_comp)
                plt.close(fig_comp)

            st.markdown("---")
            st.markdown(f"#### 🔍 Detailed Diagnostic Visualizations for Active Model: `{best_name}`")

            metrics = bundle.get("metrics", {})
            cm_col, roc_col, pr_col = st.columns(3)

            test_path = os.path.join("data", "fraudTest.csv")
            if os.path.exists(test_path) and "y_pred" in metrics and "y_probs" in metrics:
                test_df = load_dataset(test_path)
                _, y_test = prepare_data_for_modeling(test_df)

                with cm_col:
                    fig_cm = plot_confusion_matrix(y_test, metrics["y_pred"], f"{best_name} Confusion Matrix")
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)

                with roc_col:
                    if metrics["y_probs"] is not None:
                        fig_roc = plot_roc_curve(y_test, metrics["y_probs"], best_name)
                        st.pyplot(fig_roc)
                        plt.close(fig_roc)

                with pr_col:
                    if metrics["y_probs"] is not None:
                        fig_pr = plot_precision_recall_curve(y_test, metrics["y_probs"], best_name)
                        st.pyplot(fig_pr)
                        plt.close(fig_pr)
            else:
                st.info("Evaluation plots rendered upon training.")

            # Educational interview guide card
            with st.expander("💡 Technical Interview Explainer: Why PR-AUC & F1-Score over Accuracy?", expanded=False):
                st.markdown("""
                **1. The Accuracy Paradox in Imbalanced Datasets:**
                In financial fraud detection, only ~0.5% to 1.5% of transactions are fraudulent. A naive model that classifies 100% of transactions as legitimate achieves **99% accuracy**, yet catches **0 fraud cases**. Accuracy is misleading.

                **2. F1-Score & PR-AUC:**
                - **Recall (Sensitivity):** What percentage of all actual frauds did our model catch? Essential for stopping financial loss.
                - **Precision:** When our model raises an alarm, what percentage is truly fraudulent? Essential for customer experience and false alarm cost.
                - **PR-AUC (Precision-Recall Area Under Curve):** Focuses specifically on the minority positive class without being inflated by overwhelming true negatives.
                """)

    # -------------------------------------------------------------------------
    # TAB 3: TRANSACTION PREDICTION (GENUINE MANUAL INPUT)
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Manual Transaction Risk Prediction")
        st.caption(
            "Enter genuine human-readable transaction parameters to calculate the fraud risk score "
            "directly using the trained active model. No anonymized PCA values required."
        )

        if bundle is None:
            st.error("⚠️ Please train or load a model from the sidebar first.")
        else:
            # Preset buttons for demo convenience
            st.markdown("##### ⚡ Quick Demonstration Presets")
            p_col1, p_col2, p_col3 = st.columns(3)

            # Session state defaults for manual inputs
            if "man_amt" not in st.session_state:
                st.session_state.man_amt = 48.50
            if "man_cat" not in st.session_state:
                st.session_state.man_cat = "grocery_pos"
            if "man_dist" not in st.session_state:
                st.session_state.man_dist = 6.2
            if "man_pop" not in st.session_state:
                st.session_state.man_pop = 65000
            if "man_age" not in st.session_state:
                st.session_state.man_age = 36

            with p_col1:
                if st.button("🟢 Preset: Everyday Grocery Store ($34.20)", use_container_width=True):
                    st.session_state.man_amt = 34.20
                    st.session_state.man_cat = "grocery_pos"
                    st.session_state.man_dist = 4.5
                    st.session_state.man_pop = 80000
                    st.session_state.man_age = 42
                    st.rerun()

            with p_col2:
                if st.button("🔴 Preset: Suspicious High-Value Online ($1,850.00)", use_container_width=True):
                    st.session_state.man_amt = 1850.00
                    st.session_state.man_cat = "shopping_net"
                    st.session_state.man_dist = 650.0
                    st.session_state.man_pop = 250000
                    st.session_state.man_age = 29
                    st.rerun()

            with p_col3:
                if st.button("🔄 Reset Form to Standard Baseline", use_container_width=True):
                    st.session_state.man_amt = 50.00
                    st.session_state.man_cat = "food_dining"
                    st.session_state.man_dist = 10.0
                    st.session_state.man_pop = 50000
                    st.session_state.man_age = 35
                    st.rerun()

            st.write("")
            with st.form("manual_prediction_form"):
                st.markdown("#### 📝 Transaction Parameters")
                f1, f2, f3 = st.columns(3)

                with f1:
                    amt_input = st.number_input(
                        "Transaction Amount ($ USD):",
                        min_value=0.01,
                        max_value=50000.0,
                        value=float(st.session_state.man_amt),
                        step=5.0,
                        help="Monetary transaction charge in US Dollars."
                    )
                    category_input = st.selectbox(
                        "Merchant Category:",
                        options=ALL_CATEGORIES,
                        index=ALL_CATEGORIES.index(st.session_state.man_cat) if st.session_state.man_cat in ALL_CATEGORIES else 0,
                        format_func=lambda c: c.replace("_", " ").title()
                    )

                with f2:
                    tx_date = st.date_input("Transaction Date:", value=date.today())
                    tx_time = st.time_input("Transaction Time:", value=datetime.now().time())
                    combined_dt = datetime.combine(tx_date, tx_time)

                    dist_input = st.number_input(
                        "Customer-to-Merchant Distance (km):",
                        min_value=0.1,
                        max_value=20000.0,
                        value=float(st.session_state.man_dist),
                        step=5.0,
                        help="Geographic distance between customer billing location and merchant terminal."
                    )

                with f3:
                    city_pop_input = st.number_input(
                        "Customer City Population:",
                        min_value=100,
                        max_value=25000000,
                        value=int(st.session_state.man_pop),
                        step=5000,
                        help="Population of customer's registered city."
                    )
                    age_input = st.number_input(
                        "Cardholder Age:",
                        min_value=18,
                        max_value=105,
                        value=int(st.session_state.man_age),
                        step=1,
                        help="Age of cardholder calculated from date of birth."
                    )

                analyze_btn = st.form_submit_button("🔍 Analyze Transaction with AI Model", type="primary", use_container_width=True)

            if analyze_btn:
                manual_dict = {
                    "amt": amt_input,
                    "category": category_input,
                    "date_time": combined_dt,
                    "distance_km": dist_input,
                    "city_pop": city_pop_input,
                    "age": age_input
                }

                # Run prediction on active model
                res = predict_manual_transaction(
                    st.session_state.model_bundle,
                    manual_dict,
                    st.session_state.risk_threshold_low,
                    st.session_state.risk_threshold_high
                )
                st.session_state.last_manual_prediction = res

                # Automatically persist manual audit record
                tx_id = f"TX-M{st.session_state.tx_counter:05d}"
                st.session_state.tx_counter += 1
                save_transaction_record({
                    "transaction_id": tx_id,
                    "timestamp": combined_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "amount": amt_input,
                    "category": category_input,
                    "distance_km": dist_input,
                    "city_pop": city_pop_input,
                    "hour": combined_dt.hour,
                    "day_of_week": combined_dt.weekday(),
                    "is_weekend": 1 if combined_dt.weekday() in [5, 6] else 0,
                    "age": age_input,
                    "prediction": res["prediction_label"],
                    "fraud_probability": res["fraud_probability"],
                    "risk_category": res["risk_category"],
                    "source": "Manual Input"
                })

            if st.session_state.last_manual_prediction:
                r = st.session_state.last_manual_prediction
                st.markdown("---")
                st.markdown("### 🎯 Model Prediction & Decision Analysis")

                # Visual FinTech alert banner
                if r["prediction_code"] == 1:
                    st.markdown(f"""
                    <div class="fraud-banner">
                        <h3 style="margin: 0; color: #b91c1c;">🚨 FRAUDULENT TRANSACTION DETECTED (Class 1)</h3>
                        <p style="margin: 6px 0 0 0; font-size: 0.95rem;">
                            High-risk anomalous patterns identified across monetary amount, geographic location, and merchant profile.
                            Recommended action: <strong>Decline transaction immediately or refer to Senior Fraud Analyst</strong>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="legit-banner">
                        <h3 style="margin: 0; color: #15803d;">✅ LEGITIMATE TRANSACTION VERIFIED (Class 0)</h3>
                        <p style="margin: 6px 0 0 0; font-size: 0.95rem;">
                            Transaction parameters align with verified historical purchasing behavior and normal consumer demographics.
                            Recommended action: <strong>Automated instant approval</strong>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                r_col1, r_col2, r_col3 = st.columns([1, 1, 1])

                with r_col1:
                    st.metric("Model Classification", r["prediction_label"])
                    st.caption(f"Evaluated by: **{r.get('model_name', 'Trained Classifier')}**")

                with r_col2:
                    prob_val = r["fraud_probability"]
                    st.metric("Fraud Probability Score", f"{prob_val * 100:.2f}%")
                    st.progress(float(prob_val))

                with r_col3:
                    st.metric("Decision Policy Tier", r["risk_category"])
                    if r["risk_category"] == "Low Risk":
                        st.caption("🟢 Automated clearance (< Low Threshold).")
                    elif r["risk_category"] == "Medium Risk":
                        st.caption("🟡 Step-up verification (SMS / App 2FA OTP).")
                    else:
                        st.caption("🔴 High-risk alert referral (> High Threshold).")

                st.caption(
                    "⚠️ **Disclaimer:** This prediction is a statistical estimate produced by the trained machine learning "
                    "model based on historical behavioral distributions. It is not an absolute financial guarantee."
                )

    # -------------------------------------------------------------------------
    # TAB 4: LIVE TRANSACTION MONITORING (SQLITE PERSISTENCE)
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("📡 Enterprise Transaction Monitoring & Live Ingestion")
        st.caption(
            "High-throughput transaction auditing powered by the trained ML pipeline and SQLite persistence. "
            "Ingest transaction batches via Bulk CSV or simulate continuous live streams with automated risk tiering."
        )

        if bundle is None:
            st.warning("⚠️ Please train or load a model from the sidebar first to enable live transaction monitoring.")
        else:
            # 1. Ingestion Mode Selection
            st.markdown("#### 📥 Select Transaction Ingestion Mode")
            ingestion_mode = st.radio(
                "Choose Ingestion Mode:",
                ["📂 Mode A: Bulk CSV Batch Processing", "⚡ Mode B: Continuous Live Stream Simulator"],
                horizontal=True,
                label_visibility="collapsed"
            )

            st.write("")

            # =====================================================================
            # MODE A: BULK CSV UPLOAD
            # =====================================================================
            if "Mode A" in ingestion_mode:
                with st.container(border=True):
                    st.markdown("### 📂 Bulk Transaction CSV Ingestion")
                    st.markdown(
                        "Upload a CSV file containing multiple transactions for automated batch prediction, "
                        "risk scoring, and persistent SQLite logging."
                    )

                    # Downloadable template
                    col_t1, col_t2 = st.columns([1.5, 3])
                    with col_t1:
                        sample_csv = generate_sample_csv_template()
                        st.download_button(
                            "📥 Download Sample CSV Template",
                            data=sample_csv,
                            file_name="sample_transactions_template.csv",
                            mime="text/csv",
                            help="Template matching the required feature schema."
                        )
                    with col_t2:
                        st.caption("Required columns: `amt`, `category`, `trans_date_trans_time`, `distance_km`, `city_pop`, `age`")

                    bulk_file = st.file_uploader("Upload Transaction CSV Batch", type=["csv"], key="bulk_uploader_widget")

                    if bulk_file is not None:
                        try:
                            bulk_raw_df = pd.read_csv(bulk_file)
                            req_fields = ['amt', 'category']
                            missing_fields = [c for c in req_fields if c not in bulk_raw_df.columns]

                            if missing_fields:
                                st.error(f"Invalid CSV schema! Missing essential columns: {', '.join(missing_fields)}")
                            else:
                                total_rows = len(bulk_raw_df)
                                valid_mask = (bulk_raw_df['amt'] > 0) & (bulk_raw_df['category'].notna())
                                valid_count = int(valid_mask.sum())
                                invalid_count = total_rows - valid_count

                                col_v1, col_v2, col_v3 = st.columns(3)
                                col_v1.metric("Total Records Found", f"{total_rows:,}")
                                col_v2.metric("Valid Rows", f"{valid_count:,}")
                                col_v3.metric("Invalid / Skipped Rows", f"{invalid_count:,}", delta_color="inverse")

                                st.markdown("##### Batch Data Preview (First 5 Rows)")
                                st.dataframe(bulk_raw_df.head(5), use_container_width=True)

                                if valid_count > 0:
                                    if st.button(f"🚀 Process All {valid_count} Transactions", type="primary", use_container_width=True):
                                        with st.spinner("Executing vectorized model inference and logging to SQLite..."):
                                            valid_df = bulk_raw_df[valid_mask].copy()

                                            # Run batch prediction
                                            res_df, records_list = predict_bulk_transactions(
                                                st.session_state.model_bundle,
                                                valid_df,
                                                st.session_state.risk_threshold_low,
                                                st.session_state.risk_threshold_high,
                                                source="Bulk CSV Upload",
                                                start_tx_id=st.session_state.tx_counter
                                            )

                                            # Save to SQLite
                                            save_bulk_transaction_records(records_list)
                                            st.session_state.tx_counter += len(records_list)
                                            st.session_state.last_bulk_results = res_df

                                            st.success(f"Successfully processed and stored {len(records_list):,} transactions in SQLite database!")

                        except Exception as err:
                            st.error(f"Error parsing uploaded file: {err}")

                    if "last_bulk_results" in st.session_state and st.session_state.last_bulk_results is not None:
                        b_res = st.session_state.last_bulk_results
                        st.markdown("---")
                        st.markdown("#### 📊 Batch Processing Results Summary")

                        n_total = len(b_res)
                        n_fraud = int((b_res["raw_pred"] == 1).sum())
                        n_legit = n_total - n_fraud
                        n_high = int((b_res["Risk Tier"] == "High Risk").sum())
                        avg_p = float(b_res["raw_prob"].mean())

                        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
                        m_c1.metric("Processed Batch", f"{n_total:,}")
                        m_c2.metric("Flagged Frauds", f"{n_fraud:,}", f"{n_fraud/n_total*100:.2f}%", delta_color="inverse")
                        m_c3.metric("Legitimate", f"{n_legit:,}")
                        m_c4.metric("High-Risk Alerts", f"{n_high:,}", delta_color="inverse")
                        m_c5.metric("Avg Fraud Risk", f"{avg_p*100:.2f}%")

                        st.markdown("##### Detailed Prediction Results Table")
                        display_batch_cols = ["Transaction ID", "Timestamp", "Amount ($)", "Category", "Distance (km)", "Prediction", "Fraud Probability", "Risk Tier"]
                        st.dataframe(b_res[display_batch_cols], use_container_width=True)

                        csv_batch_export = b_res[display_batch_cols].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Export Batch Prediction Results (CSV)",
                            data=csv_batch_export,
                            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

            # =====================================================================
            # MODE B: CONTINUOUS LIVE SIMULATOR
            # =====================================================================
            else:
                with st.container(border=True):
                    st.markdown("### ⚡ Continuous Live Transaction Simulator")
                    st.caption("Generates synthetic incoming transaction events with realistic human-readable parameters and streams them live through the model.")
                    st.info("⚠️ **Simulation Notice:** All transactions are generated synthetically for testing real-time AI inference and automated risk tiering. They are not actual live bank transactions.")

                    sim_c1, sim_c2, sim_c3 = st.columns(3)
                    with sim_c1:
                        sim_count = st.slider("Number of Transactions to Stream:", min_value=5, max_value=100, value=20, step=5, help="Total simulated transactions to process in this run.")
                    with sim_c2:
                        sim_interval = st.slider("Stream Interval Delay (seconds):", min_value=0.05, max_value=1.5, value=0.15, step=0.05, help="Pacing delay between consecutive incoming transactions.")
                    with sim_c3:
                        sim_scenario = st.selectbox(
                            "Simulation Ingestion Scenario:",
                            [
                                "🟢 Typical Everyday Consumer Stream (~1% baseline fraud)",
                                "🔴 High-Risk Anomaly Surge (~25% attack scenario)",
                                "🎲 Balanced Mixed Test Stream (~10% fraud)"
                            ]
                        )

                    if st.button(f"▶️ Start Live Stream ({sim_count} Transactions)", type="primary", use_container_width=True):
                        progress_bar = st.progress(0)
                        status_box = st.empty()
                        recent_event_box = st.empty()

                        prob_fraud_bias = 0.25 if "High-Risk" in sim_scenario else (0.10 if "Balanced" in sim_scenario else 0.015)
                        generated_records = []

                        for i in range(sim_count):
                            is_anomaly = random.random() < prob_fraud_bias
                            if is_anomaly:
                                amt_val = round(random.uniform(750.0, 3200.0) + random.expovariate(1.0 / 200.0), 2)
                                cat_val = random.choice(["shopping_net", "misc_net", "travel", "grocery_net"])
                                dist_val = round(random.uniform(450.0, 3500.0), 1)
                                hour_val = random.choice([1, 2, 3, 4, 23, 0])
                                pop_val = random.randint(250000, 2000000)
                                age_val = random.randint(21, 45)
                            else:
                                amt_val = round(random.uniform(8.0, 110.0) + random.expovariate(1.0 / 25.0), 2)
                                cat_val = random.choice(ALL_CATEGORIES)
                                dist_val = round(random.uniform(1.0, 35.0), 1)
                                hour_val = random.choice(range(7, 22))
                                pop_val = random.randint(15000, 350000)
                                age_val = random.randint(22, 75)

                            now_dt = datetime.now()
                            tx_date_time = datetime(now_dt.year, now_dt.month, now_dt.day, hour_val, random.randint(0, 59), random.randint(0, 59))

                            input_dict = {
                                "amt": amt_val,
                                "category": cat_val,
                                "date_time": tx_date_time,
                                "distance_km": dist_val,
                                "city_pop": pop_val,
                                "age": age_val
                            }

                            # Live Inference on active model
                            res = predict_manual_transaction(
                                st.session_state.model_bundle,
                                input_dict,
                                st.session_state.risk_threshold_low,
                                st.session_state.risk_threshold_high
                            )

                            tx_id = f"TX-S{st.session_state.tx_counter:05d}"
                            st.session_state.tx_counter += 1

                            # Save directly to SQLite
                            rec = {
                                "transaction_id": tx_id,
                                "timestamp": tx_date_time.strftime("%Y-%m-%d %H:%M:%S"),
                                "amount": amt_val,
                                "category": cat_val,
                                "distance_km": dist_val,
                                "city_pop": pop_val,
                                "hour": hour_val,
                                "day_of_week": tx_date_time.weekday(),
                                "is_weekend": 1 if tx_date_time.weekday() in [5, 6] else 0,
                                "age": age_val,
                                "prediction": res["prediction_label"],
                                "fraud_probability": res["fraud_probability"],
                                "risk_category": res["risk_category"],
                                "source": "Live Simulation"
                            }
                            save_transaction_record(rec)
                            generated_records.append(rec)

                            # UI updates
                            progress_bar.progress((i + 1) / sim_count)
                            status_box.caption(f"Processing event {i+1} of {sim_count}: `{tx_id}`")

                            if res["prediction_code"] == 1:
                                recent_event_box.error(f"🚨 **ALERT ({tx_id}):** {cat_val.replace('_', ' ').title()} | ${amt_val:.2f} | **FRAUDULENT** (Risk: {res['fraud_probability']*100:.1f}%)")
                            else:
                                recent_event_box.success(f"✅ **CLEARED ({tx_id}):** {cat_val.replace('_', ' ').title()} | ${amt_val:.2f} | **LEGITIMATE** (Risk: {res['fraud_probability']*100:.1f}%)")

                            pytime.sleep(sim_interval)

                        st.success(f"Simulation completed! Generated and logged {sim_count} transactions into SQLite audit database.")

            # =====================================================================
            # COMMON MONITORING DASHBOARD & SQLITE AUDIT LOG
            # =====================================================================
            st.markdown("---")
            st.subheader("📊 Unified Monitoring Dashboard (SQLite Audit Database)")

            mon_stats = get_monitoring_stats()
            history_df = get_monitoring_history(limit=2000)

            # Top Lifetime KPIs
            k_c1, k_c2, k_c3, k_c4, k_c5 = st.columns(5)
            k_c1.metric("Lifetime Analyzed", f"{mon_stats['total']:,}")
            k_c2.metric("Flagged Fraudulent", f"{mon_stats['fraud']:,}", f"{mon_stats['fraud_rate']:.2f}% rate", delta_color="inverse")
            k_c3.metric("Legitimate Clear", f"{mon_stats['legit']:,}")
            k_c4.metric("High-Risk Alerts", f"{mon_stats['high_risk']:,}", delta_color="inverse")
            k_c5.metric("Avg Fraud Risk", f"{mon_stats['avg_fraud_prob']*100:.2f}%")

            if len(history_df) > 0:
                st.write("")
                # Visualizations grid: 4 charts
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    fig_mon = plot_monitoring_summary(
                        mon_stats["legit"],
                        mon_stats["fraud"],
                        mon_stats["low_risk"],
                        mon_stats["medium_risk"],
                        mon_stats["high_risk"]
                    )
                    st.pyplot(fig_mon)
                    plt.close(fig_mon)

                with g_col2:
                    fig_time = plot_transactions_over_time(history_df)
                    st.pyplot(fig_time)
                    plt.close(fig_time)

                g_col3, g_col4 = st.columns(2)
                with g_col3:
                    probs = history_df["Fraud Probability"].astype(float).tolist()
                    fig_dist = plot_probability_distribution(probs)
                    st.pyplot(fig_dist)
                    plt.close(fig_dist)

                with g_col4:
                    fig_src = plot_source_breakdown(mon_stats["source_counts"])
                    st.pyplot(fig_src)
                    plt.close(fig_src)

                # Filter bar
                st.markdown("### 📜 Filterable SQLite Transaction Audit Trail")
                st.caption("Filter and audit historical transactions persisted in `data/monitoring.db`.")

                f_c1, f_c2, f_c3, f_c4 = st.columns(4)
                with f_c1:
                    filter_pred = st.selectbox("Filter by Prediction:", ["All", "FRAUDULENT", "LEGITIMATE"])
                with f_c2:
                    filter_risk = st.selectbox("Filter by Risk Tier:", ["All", "High Risk", "Medium Risk", "Low Risk"])
                with f_c3:
                    filter_source = st.selectbox("Filter by Ingestion Source:", ["All", "Bulk CSV Upload", "Live Simulation", "Manual Input"])
                with f_c4:
                    search_query = st.text_input("Search ID / Category:", placeholder="e.g. TX-B or grocery")

                # Apply filters
                filtered_df = history_df.copy()
                if filter_pred != "All":
                    filtered_df = filtered_df[filtered_df["Prediction"] == filter_pred]
                if filter_risk != "All":
                    filtered_df = filtered_df[filtered_df["Risk Tier"] == filter_risk]
                if filter_source != "All":
                    filtered_df = filtered_df[filtered_df["Source"].str.contains(filter_source, na=False)]
                if search_query:
                    q = search_query.strip().lower()
                    filtered_df = filtered_df[
                        filtered_df["Transaction ID"].str.lower().str.contains(q, na=False) |
                        filtered_df["Category"].str.lower().str.contains(q, na=False)
                    ]

                st.write(f"Displaying **{len(filtered_df):,}** of **{len(history_df):,}** audit records.")
                st.dataframe(filtered_df, use_container_width=True)

                # Action buttons
                b_col1, b_col2 = st.columns([1, 4])
                with b_col1:
                    if st.button("🗑️ Clear Audit Database"):
                        clear_monitoring_history()
                        st.success("Monitoring database cleared.")
                        st.rerun()

                with b_col2:
                    export_csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Export Filtered Audit Trail (CSV)",
                        data=export_csv,
                        file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No transaction records logged in SQLite database yet. Ingest transactions using Mode A (Bulk CSV) or Mode B (Live Simulator) above.")


if __name__ == "__main__":
    main()
