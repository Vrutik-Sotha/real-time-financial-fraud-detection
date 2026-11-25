"""
Visualization Utilities for Financial Fraud Detection System
Using Matplotlib and Seaborn for clean, professional charts.
"""

import os
import tempfile
# Ensure Matplotlib uses a writable temporary directory on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import numpy as np

# FinTech dark theme styling helper
def apply_fintech_style(fig, ax=None):
    """Apply the premium FinTech dark theme to Matplotlib figures.
    Adjusts background colors, text colors, grid, and spines to match the
    navy/teal palette defined in the Streamlit config.
    """
    # Figure background
    fig.patch.set_facecolor('#0B1220')  # main background
    # Axes styling
    if ax is None:
        axes = fig.get_axes()
    else:
        axes = [ax]
    for a in axes:
        a.set_facecolor('#111C2E')  # secondary background
        # Tick parameters
        a.tick_params(colors='#F1F5F9', labelsize=10)
        # Axis label colors
        a.xaxis.label.set_color('#F1F5F9')
        a.yaxis.label.set_color('#F1F5F9')
        # Title color
        a.title.set_color('#F1F5F9')
        # Grid style
        a.grid(True, color='#4B5563', linewidth=0.5, alpha=0.4)
        # Spines color
        for spine in a.spines.values():
            spine.set_color('#4B5563')
    # Legend styling if present
    if fig.legends:
        for leg in fig.legends:
            leg.get_frame().set_facecolor('#17243A')
            leg.get_frame().set_edgecolor('#4B5563')
            for text in leg.get_texts():
                text.set_color('#F1F5F9')

    return fig
import pandas as pd


def plot_class_distribution(legit_count, fraud_count):
    """Creates a donut chart displaying Legitimate vs. Fraudulent transactions.
    Shows total count in the center.
    """
    total = legit_count + fraud_count
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ['Legitimate', 'Fraudulent']
    counts = [legit_count, fraud_count]
    colors = ['#16A34A', '#DC2626']  # green for legit, red for fraud

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=(0, 0.08),
        textprops=dict(color='#F1F5F9', fontsize=11, weight='bold')
    )
    # Draw donut hole
    centre_circle = plt.Circle((0,0),0.55,fc='#111C2E')
    ax.add_artist(centre_circle)
    # Center text with total count
    ax.text(0, 0, f"{total}\nTxns", ha='center', va='center', color='#F1F5F9', fontsize=14, weight='bold')
    ax.set_title('Transaction Class Distribution', fontsize=14, weight='bold', pad=15, color='#F1F5F9')
    plt.tight_layout()
    apply_fintech_style(fig, ax)
    return fig



    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ['Legitimate (0)', 'Fraudulent (1)']
    counts = [legit_count, fraud_count]
    colors = ['#2ecc71', '#e74c3c']

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct='%1.2f%%',
        startangle=140,
        colors=colors,
        explode=(0, 0.12),
        textprops=dict(color="black", fontsize=11, weight="bold")
    )

    ax.set_title('Transaction Class Distribution', fontsize=14, weight='bold', pad=15)
    plt.tight_layout()

    return fig



def plot_confusion_matrix(y_true, y_pred, model_title="Model Confusion Matrix"):
    """Plots a Confusion Matrix heatmap with professional styling."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Reds',
        cbar=False,
        xticklabels=['Pred Legit', 'Pred Fraud'],
        yticklabels=['Actual Legit', 'Actual Fraud'],
        annot_kws={"size": 13, "weight": "bold", "color": "#F1F5F9"},
        ax=ax
    )
    
    ax.set_title(model_title, fontsize=12, weight='bold', pad=12, color='#F1F5F9')
    ax.set_ylabel('True Label', fontsize=11, weight='bold', color='#F1F5F9')
    ax.set_xlabel('Predicted Label', fontsize=11, weight='bold', color='#F1F5F9')
    plt.tight_layout()
    apply_fintech_style(fig, ax)
    return fig
    """Plots a Confusion Matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['Pred Legit', 'Pred Fraud'],
        yticklabels=['Actual Legit', 'Actual Fraud'],
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax
    )
    
    ax.set_title(model_title, fontsize=12, weight='bold', pad=12)
    ax.set_ylabel('True Label', fontsize=11, weight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11, weight='bold')
    plt.tight_layout()

    return fig


def plot_roc_curve(y_true, y_probs, model_name="Best Model"):
    """Plots the ROC curve and Area Under Curve (AUC)."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, color='#e67e22', lw=2.5, label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.50)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, weight='bold')
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11, weight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=13, weight='bold', pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig


def plot_precision_recall_curve(y_true, y_probs, model_name="Best Model"):
    """Plots Precision-Recall curve and computes PR-AUC."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, color='#2980b9', lw=2.5, label=f'{model_name} (PR-AUC = {pr_auc:.4f})')
    
    baseline = sum(y_true) / len(y_true)
    ax.plot([0, 1], [baseline, baseline], color='#7f8c8d', lw=1.5, linestyle='--', label=f'Baseline ({baseline:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Detection Rate)', fontsize=11, weight='bold')
    ax.set_ylabel('Precision (True Fraud Ratio)', fontsize=11, weight='bold')
    ax.set_title('Precision-Recall Curve (PR-AUC)', fontsize=13, weight='bold', pad=12)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig


def plot_model_comparison(comparison_results):
    """
    Plots a multi-metric grouped bar chart comparing candidate models:
    Logistic Regression, Random Forest, and XGBoost.
    """
    models = list(comparison_results.keys())
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'roc_auc']
    metric_labels = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.5))

    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']

    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [comparison_results[m].get(metric, 0.0) or 0.0 for m in models]
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, values, width, label=label, color=colors[i], alpha=0.9)

        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, weight='bold')

    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=11, weight='bold')
    ax.set_title('Multi-Model Performance Comparison (Evaluation Dataset)', fontsize=13, weight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, weight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    return fig


def plot_monitoring_summary(legit_count, fraud_count, low_count, med_count, high_count):
    """Plots monitoring predictions and risk tier breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    # Chart 1: Outcome counts
    outcomes = ['Legitimate', 'Fraudulent']
    counts_outcome = [legit_count, fraud_count]
    colors_outcome = ['#2ecc71', '#e74c3c']
    bars1 = ax1.bar(outcomes, counts_outcome, color=colors_outcome, width=0.45)
    ax1.set_title('Monitoring Outcomes', fontsize=11, weight='bold')
    ax1.set_ylabel('Transaction Count', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    max_y1 = max(counts_outcome) if counts_outcome else 1
    ax1.set_ylim(0, max(max_y1 + 1, 3))
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f'{int(yval)}', ha='center', va='bottom', fontsize=10, weight='bold')

    # Chart 2: Risk tiers
    risks = ['Low Risk', 'Medium Risk', 'High Risk']
    counts_risk = [low_count, med_count, high_count]
    colors_risk = ['#27ae60', '#f39c12', '#c0392b']
    bars2 = ax2.bar(risks, counts_risk, color=colors_risk, width=0.5)
    ax2.set_title('Risk Tier Breakdown', fontsize=11, weight='bold')
    ax2.set_ylabel('Transaction Count', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    max_y2 = max(counts_risk) if counts_risk else 1
    ax2.set_ylim(0, max(max_y2 + 1, 3))
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f'{int(yval)}', ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()

    return fig


def plot_probability_distribution(probs_list):
    """Plots histogram of predicted fraud probabilities."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    if not probs_list:
        ax.text(0.5, 0.5, "No probability data yet", ha="center", va="center", fontsize=11, color="gray")
    
    return fig

    sns.histplot(probs_list, bins=15, kde=True, color='#8e44ad', ax=ax)
    ax.set_title('Fraud Probability Distribution (Audit History)', fontsize=12, weight='bold', pad=10)
    ax.set_xlabel('Predicted Fraud Probability', fontsize=10, weight='bold')
    ax.set_ylabel('Count', fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig


def plot_transactions_over_time(history_df):
    """Plots cumulative activity chart of monitored transactions over time."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    if history_df.empty or "Prediction" not in history_df.columns:
        ax.text(0.5, 0.5, "No transaction history recorded yet", ha="center", va="center", fontsize=11, color="gray")
    
    return fig

    df_rev = history_df.iloc[::-1].copy().reset_index(drop=True)
    legit_cum = (df_rev["Prediction"] == "LEGITIMATE").cumsum()
    fraud_cum = (df_rev["Prediction"] == "FRAUDULENT").cumsum()

    x_indices = range(1, len(df_rev) + 1)
    ax.plot(x_indices, legit_cum, color='#2ecc71', lw=2.5, label='Legitimate Transactions')
    ax.plot(x_indices, fraud_cum, color='#e74c3c', lw=2.5, label='Fraudulent Alerts')

    ax.set_title('Cumulative Monitored Transactions', fontsize=12, weight='bold', pad=10)
    ax.set_xlabel('Processed Transaction Sequence', fontsize=10, weight='bold')
    ax.set_ylabel('Cumulative Count', fontsize=10, weight='bold')
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig


def plot_source_breakdown(source_dict):
    """Plots a donut chart of transactions grouped by input ingestion mode."""
    fig, ax = plt.subplots(figsize=(5, 3.8))
    labels = [k for k, v in source_dict.items() if v > 0]
    counts = [v for k, v in source_dict.items() if v > 0]
    if not counts:
        ax.text(0.5, 0.5, "No ingestion source data yet", ha="center", va="center", fontsize=11, color="gray")
    
    return fig

    colors = ['#3498db', '#9b59b6', '#1abc9c'][:len(counts)]
    ax.pie(
        counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        textprops=dict(fontsize=9, weight="bold")
    )
    ax.set_title('Transactions by Ingestion Source', fontsize=11, weight='bold', pad=10)
    plt.tight_layout()

    return fig
