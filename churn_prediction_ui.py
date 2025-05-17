"""
Telco Customer Churn Prediction Interface

This module provides a Gradio web interface for making customer churn predictions
using a trained stacking ensemble model. It includes MLflow integration for
prediction tracking and monitoring.

The interface allows users to input customer details and receive churn predictions
with confidence scores. All predictions are logged to MLflow for monitoring and analysis.

Author: Your Name
Date: 2024
"""

import gradio as gr
import numpy as np
import json
import mlflow
import pandas as pd
import joblib
import os
from datetime import datetime

# Set MLflow tracking URI to local file-based path
mlflow.set_tracking_uri("file:./mlruns")

# Define the model URI with the specific run ID
RUN_ID = "9679063b361d4996a7a3b6034e35ceb5"
MODEL_URI = f"runs:/{RUN_ID}/model"

# Create or select experiment for live predictions
experiment_name = "Telco Churn Live Predictions"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Load the model and preprocessor with error handling
try:
    print(f"Attempting to load model from {MODEL_URI}")
    
    # Verify the run exists and get artifacts
    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(RUN_ID)
        print(f"Found run: {run.info.run_id}")
        print(f"Run status: {run.info.status}")
        
        # List all artifacts to verify preprocessor exists
        artifacts = client.list_artifacts(RUN_ID)
        print("Available artifacts:", [a.path for a in artifacts])
        
        # Download preprocessor from artifacts
        artifact_path = client.download_artifacts(RUN_ID, "preprocessor/preprocessor.joblib")
        preprocessor = joblib.load(artifact_path)
        print("Successfully loaded preprocessor")
        
        # Load the model using the specific run URI
        loaded_model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"Successfully loaded model from {MODEL_URI}")
        
    except Exception as e:
        print(f"Error accessing run or artifacts: {str(e)}")
        raise
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Please ensure the model exists at the specified URI and MLflow is running")
    raise

def create_feature_row(tenure, monthly_charges, total_charges, contract, internet_service):
    """
    Create a feature row for prediction from user inputs.
    
    Args:
        tenure (float): Customer tenure in months
        monthly_charges (float): Monthly charges amount
        total_charges (float): Total charges amount
        contract (str): Contract type ('Month-to-month' or 'Two year')
        internet_service (str): Internet service type ('DSL' or 'Fiber optic')
    
    Returns:
        pd.DataFrame: DataFrame containing all features for prediction
    """
    # Create a dictionary with all features
    data = {
        'tenure': float(tenure),
        'MonthlyCharges': float(monthly_charges),
        'TotalCharges': float(total_charges),
        'gender': 'Female',  # Default values for other features
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': internet_service,
        'OnlineSecurity': 'No internet service' if internet_service == 'No' else 'No',
        'OnlineBackup': 'No internet service' if internet_service == 'No' else 'No',
        'DeviceProtection': 'No internet service' if internet_service == 'No' else 'No',
        'TechSupport': 'No internet service' if internet_service == 'No' else 'No',
        'StreamingTV': 'No internet service' if internet_service == 'No' else 'No',
        'StreamingMovies': 'No internet service' if internet_service == 'No' else 'No',
        'Contract': contract,
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check'
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Convert binary columns to 0/1
    binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    return df

def predict_churn(tenure, monthly_charges, total_charges, contract, internet_service):
    """
    Make a churn prediction for a customer and log it to MLflow.
    
    Args:
        tenure (float): Customer tenure in months
        monthly_charges (float): Monthly charges amount
        total_charges (float): Total charges amount
        contract (str): Contract type ('Month-to-month' or 'Two year')
        internet_service (str): Internet service type ('DSL' or 'Fiber optic')
    
    Returns:
        str: Prediction result with confidence score
    """
    try:
        # Create a DataFrame with all features
        df = create_feature_row(tenure, monthly_charges, total_charges, contract, internet_service)
        
        # Preprocess the input data
        X_processed = preprocessor.transform(df)
        
        # Make prediction
        prediction = loaded_model.predict(X_processed)
        churn_probability = float(prediction[0])  # Ensure float type
        
        # Convert probability to class
        result = "Churn" if churn_probability > 0.5 else "No Churn"
        confidence = f"({churn_probability:.2%})"

        # Generate unique run name with timestamp
        run_name = f"Live_Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Log prediction to MLflow
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log input features as parameters
            mlflow.log_params({
                "tenure": float(tenure),
                "monthly_charges": float(monthly_charges),
                "total_charges": float(total_charges),
                "contract": contract,
                "internet_service": internet_service
            })
            
            # Log prediction result as tag
            mlflow.set_tag("prediction", result)
            
            # Log probability score as metric
            mlflow.log_metric("churn_probability", churn_probability)
            
            # Log timestamp
            mlflow.set_tag("timestamp", datetime.now().isoformat())
            
            # Log all features for debugging
            mlflow.log_dict(df.to_dict(orient='records')[0], "input_features.json")
        
        return f"{result} {confidence}"
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)  # Print error for debugging
        return error_msg

# Create Gradio interface
iface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Slider(minimum=0, maximum=100, step=1, label="Tenure (months)"),
        gr.Slider(minimum=0, maximum=200, step=1, label="Monthly Charges ($)"),
        gr.Slider(minimum=0, maximum=10000, step=1, label="Total Charges ($)"),
        gr.Radio(["Month-to-month", "Two year"], label="Contract Type"),
        gr.Radio(["DSL", "Fiber optic"], label="Internet Service")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Telco Customer Churn Predictor",
    description="Enter customer details to predict churn probability",
    examples=[
        [12, 70, 840, "Month-to-month", "Fiber optic"],
        [24, 50, 1200, "Two year", "DSL"],
        [6, 90, 540, "Month-to-month", "Fiber optic"]
    ]
)

if __name__ == "__main__":
    iface.launch() 