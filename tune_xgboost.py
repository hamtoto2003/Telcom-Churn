"""
Telco Customer Churn XGBoost Hyperparameter Tuning

This script performs hyperparameter tuning for an XGBoost model using Optuna.
It includes data preprocessing, model training, and evaluation, with all
experiments tracked in MLflow.

Features:
- Automated hyperparameter optimization with Optuna
- Comprehensive model evaluation metrics
- MLflow integration for experiment tracking
- Model and preprocessor persistence

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
import joblib

# Set up MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")
experiment_name = "Telco Customer Churn Analysis"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow experiment: {experiment_name}")

def load_and_preprocess_data():
    """
    Load and preprocess the Telco customer churn dataset.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for modeling
    """
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

def prepare_data(df):
    """
    Prepare data for model training by creating preprocessing pipeline
    and splitting into train/test sets.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame
    
    Returns:
        tuple: (X_train_processed, X_test_processed, y_train, y_test)
    """
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
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor locally
    joblib.dump(preprocessor, "preprocessor.joblib")
    
    # Log preprocessor to MLflow
    mlflow.log_artifact("preprocessor.joblib", artifact_path="preprocessor")
    
    return X_train_processed, X_test_processed, y_train, y_test

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        float: ROC AUC score on validation set
    """
    # Define hyperparameter search space
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
        'random_state': 42,
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }
    
    # Create and train model
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Make predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba)
    }
    
    # Log metrics to MLflow
    with mlflow.start_run(nested=True):
        for param, value in params.items():
            mlflow.log_param(param, value)
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
    
    # Return the metric to optimize (ROC AUC)
    return metrics['roc_auc']

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: True labels
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
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

def main():
    """
    Main function to run the hyperparameter tuning process.
    Includes data preparation, model training, and logging to MLflow.
    """
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("Preparing data for modeling...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Further split training data for validation with stratification
    X_train_tune, X_val, y_train_tune, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print("\nStarting hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train_tune, y_train_tune, X_val, y_val),
        n_trials=10
    )
    
    # Save the study object
    joblib.dump(study, "optuna_study.pkl")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_params = trial.params
    best_params.update({
        'random_state': 42,
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    })
    
    with mlflow.start_run(run_name="XGBoost_Best_Model") as run:
        print(f"MLflow run ID: {run.info.run_id}")
        
        # Add model type tag
        mlflow.set_tag("model_type", "XGBoost_Tuned")
        
        # Train model on full training set
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)
        
        # Log parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        print(f"Logged {len(best_params)} parameters")
        
        # Evaluate and log metrics
        metrics = evaluate_model(final_model, X_test, y_test)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        print(f"Logged {len(metrics)} metrics")
        
        # Log the model and study
        mlflow.xgboost.log_model(final_model, "model", registered_model_name="XGBoost_Churn_Model")
        mlflow.log_artifact("optuna_study.pkl", "study")
        mlflow.log_artifact("preprocessor.joblib", "preprocessor")
        print("Logged XGBoost model, study, and preprocessor")
        
        print("\nFinal Model Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print(f"\nRun details: {run.info.artifact_uri}")

if __name__ == "__main__":
    main() 