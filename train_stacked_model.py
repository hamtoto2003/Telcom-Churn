"""
Telco Customer Churn Model Training

This script trains a stacking ensemble model for predicting customer churn.
It uses XGBoost and Random Forest as base learners and Logistic Regression
as the meta-learner. The model is trained, evaluated, and logged to MLflow
for experiment tracking and model management.

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import joblib

# Set up MLflow experiment
experiment_name = "Telco Churn Stacked Ensemble"
mlflow.set_tracking_uri("file:./mlruns")
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

def load_and_preprocess_data():
    """
    Load and preprocess the Telco customer churn dataset.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for model training
    """
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = df.drop('customerID', axis=1)
    binary_columns = ['Churn', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

def prepare_data(df):
    """
    Prepare data for model training by splitting features and target,
    and creating a preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame
    
    Returns:
        tuple: (X_train_prep, X_test_prep, y_train, y_test) - Preprocessed training and test data
    """
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    cat_cols = X.select_dtypes(include='object').columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ])
    
    # Split and transform data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    preprocessor.fit(X_train)
    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    # Save preprocessor for later use
    joblib.dump(preprocessor, "stacking_preprocessor.joblib")
    mlflow.log_artifact("stacking_preprocessor.joblib", artifact_path="preprocessor")
    
    return X_train_prep, X_test_prep, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: True labels
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

def main():
    """
    Main function to train and evaluate the stacking ensemble model.
    Logs model and metrics to MLflow.
    """
    # Load and prepare data
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Define base learners
    base_learners = [
        ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]

    # Create and train stacking ensemble
    final_estimator = LogisticRegression()
    model = StackingClassifier(estimators=base_learners, final_estimator=final_estimator, cv=5)

    # Train and log model with MLflow
    with mlflow.start_run(run_name="Stacked Ensemble", nested=True) as run:
        model.fit(X_train, y_train)

        # Log model metadata
        mlflow.set_tag("model_type", "StackedEnsemble")
        mlflow.log_param("base_learners", "XGBoost + RandomForest")
        mlflow.log_param("meta_model", "LogisticRegression")

        # Evaluate and log metrics
        metrics = evaluate_model(model, X_test, y_test)
        for name, val in metrics.items():
            mlflow.log_metric(name, val)

        # Log the model
        mlflow.sklearn.log_model(model, "model")
        print("Logged Stacked Ensemble model with metrics:", metrics)

if __name__ == "__main__":
    main() 