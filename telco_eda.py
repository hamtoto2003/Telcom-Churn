"""
Telco Customer Churn Exploratory Data Analysis

This script performs exploratory data analysis on the Telco customer churn dataset.
It generates visualizations and summary statistics, logging all findings to MLflow
for experiment tracking and documentation.

The analysis includes:
- Basic dataset statistics
- Churn distribution analysis
- Numeric feature correlations
- Categorical feature distributions

Author: Your Name
Date: 2024
"""

import mlflow

# Setup MLflow for EDA experiment
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Telco Customer Churn EDA")

with mlflow.start_run(run_name="EDA Run"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # Load the data
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Log basic dataset information
    mlflow.log_param("dataset_rows", df.shape[0])
    mlflow.log_param("dataset_cols", df.shape[1])

    # Data preprocessing
    # Convert TotalCharges to numeric and handle missing values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Remove non-predictive features
    df.drop(columns=["customerID"], inplace=True)

    # Analyze churn distribution
    churn_counts = df["Churn"].value_counts().to_dict()
    for status, count in churn_counts.items():
        mlflow.log_metric(f"churn_{status}", int(count))
    
    # Create and save churn distribution plot
    plt.figure(figsize=(6,4))
    sns.countplot(x="Churn", data=df)
    plt.title("Churn Distribution")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    churn_plot = plots_dir / "churn_distribution.png"
    plt.savefig(churn_plot)
    plt.close()
    mlflow.log_artifact(str(churn_plot))

    # Calculate and log numeric feature statistics
    numeric_summary = df.select_dtypes(include=["float64", "int64"]).describe().to_dict()
    for col, stats in numeric_summary.items():
        for stat_name, value in stats.items():
            # Sanitize metric name by replacing '%' with 'pct'
            safe_stat = stat_name.replace('%', 'pct')
            metric_name = f"{col}_{safe_stat}"
            mlflow.log_metric(metric_name, float(value))

    # Generate correlation heatmap
    plt.figure(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap of Numeric Features")
    corr_plot = plots_dir / "correlation_heatmap.png"
    plt.savefig(corr_plot)
    plt.close()
    mlflow.log_artifact(str(corr_plot))

    # Analyze categorical feature distributions
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        plt.figure(figsize=(8,4))
        sns.countplot(x=col, data=df)
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        cat_plot = plots_dir / f"{col}_distribution.png"
        plt.savefig(cat_plot)
        plt.close()
        mlflow.log_artifact(str(cat_plot))

    # Log final dataset dimensions after preprocessing
    mlflow.log_param("processed_rows", df.shape[0])
    mlflow.log_param("processed_cols", df.shape[1])
    print("EDA run logged to MLflow successfully.") 