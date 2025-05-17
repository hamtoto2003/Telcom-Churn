import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os

# Set MLflow tracking URI to local file-based path
mlflow.set_tracking_uri("file:./mlruns")

# Create or select experiment
experiment_name = "Telco Customer Churn Analysis"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow experiment: {experiment_name}")

def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Drop customerID
    df = df.drop('customerID', axis=1)
    
    # Convert binary columns to 0/1
    binary_columns = ['Churn', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    return df

def perform_eda(df):
    print("\n=== Exploratory Data Analysis ===")
    print(f"\nDataset Shape: {df.shape}")
    print("\nColumn Types:")
    print(df.dtypes)
    
    print("\nChurn Distribution:")
    print(df['Churn'].value_counts(normalize=True))
    
    print("\nDescriptive Statistics for Numeric Columns:")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(df[numeric_cols].describe())

def prepare_data(df):
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test):
    print(f"\nStarting MLflow run for {model_name}...")
    
    with mlflow.start_run(run_name=model_name) as run:
        print(f"MLflow run ID: {run.info.run_id}")
        
        # Train the model
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Get model parameters
        params = model.get_params()
        for param, value in params.items():
            mlflow.log_param(param, value)
        print(f"Logged {len(params)} parameters")
        
        # Evaluate and log metrics
        metrics = evaluate_model(model, X_test, y_test, model_name)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        print(f"Logged {len(metrics)} metrics")
        
        # Log the model
        if isinstance(model, xgb.XGBClassifier):
            mlflow.xgboost.log_model(model, "model")
            print("Logged XGBoost model")
        else:
            mlflow.sklearn.log_model(model, "model")
            print("Logged scikit-learn model")
        
        print(f"Completed MLflow run for {model_name}")
        print(f"Run details: {run.info.artifact_uri}")

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Perform EDA
    perform_eda(df)
    
    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42)
    }
    
    # Train and evaluate each model
    for model_name, model in models.items():
        train_and_log_model(model, model_name, X_train, y_train, X_test, y_test)
    
    print("\nAll models have been trained and logged to MLflow")
    print("To view results, run: mlflow ui")

if __name__ == "__main__":
    main() 