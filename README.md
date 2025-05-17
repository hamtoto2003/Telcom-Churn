# Telco Customer Churn Prediction (MLOps Project)

This project implements an end-to-end MLOps pipeline for predicting customer churn in a telecommunications company. It features data analysis, model training, hyperparameter tuning, experiment tracking, and a production-ready prediction interface.

## Features

- **Data Analysis & EDA**
  - Exploratory data analysis with visualizations
  - Feature engineering and preprocessing
  - Baseline model development

- **Model Training**
  - Stacking ensemble (XGBoost + Random Forest + Logistic Regression)
  - Hyperparameter tuning with Optuna
  - Model evaluation and comparison

- **MLOps Integration**
  - MLflow for experiment tracking and model management
  - Live prediction monitoring
  - Model versioning and artifact storage

- **Production Interface**
  - Gradio web interface for predictions
  - Real-time prediction logging
  - User-friendly input forms

## Project Structure

```
├── churn_prediction_ui.py    # Gradio UI for predictions
├── train_stacked_model.py    # Stacking ensemble training
├── tune_xgboost.py          # Optuna hyperparameter tuning
├── telco_churn_analysis.py  # Baseline modeling
├── telco_eda.py            # Exploratory data analysis
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd telco-churn-mlops
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Analysis and EDA

Run the exploratory data analysis:
```bash
python telco_eda.py
```

### 2. Model Training

Train the stacking ensemble model:
```bash
python train_stacked_model.py
```

For hyperparameter tuning:
```bash
python tune_xgboost.py
```

### 3. MLflow Tracking

Start the MLflow tracking server:
```bash
mlflow server --host 127.0.0.1 --port 5000
```

Access the MLflow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000) to:
- View experiment runs
- Compare model metrics
- Download model artifacts
- Monitor live predictions

### 4. Live Predictions

Launch the Gradio prediction interface:
```bash
python churn_prediction_ui.py
```

Access the prediction UI at [http://127.0.0.1:7860](http://127.0.0.1:7860)

## MLflow Integration

The project uses MLflow for:
- Experiment tracking and versioning
- Model artifact storage
- Live prediction monitoring
- Metric logging and visualization

Each prediction made through the Gradio interface is logged to MLflow with:
- Input features
- Prediction results
- Timestamps
- Confidence scores

## Model Performance

The stacking ensemble model achieves:
- Accuracy: ~79%
- Precision: ~63%
- Recall: ~49%
- F1 Score: ~55%
- ROC AUC: ~82%

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Mohammad Ali Ghunaim 