# ML Model Selector

A desktop application for training and comparing **regression** and **classification** machine learning models on your own CSV datasets. Built with Python, Tkinter, and scikit-learn.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Task type**: Choose between **regression** or **classification**
- **Data loading**: Load any CSV file and select the target column
- **Data checks**: Automatic detection of missing values and categorical columns, with optional imputation and one-hot encoding
- **Visualizations**: Missing-data heatmaps, categorical distributions, model comparison charts, residual plots (regression), confusion matrices (classification)
- **Regression models**: Linear, Polynomial (degree 2), Lasso (CV), Ridge (CV), Elastic Net (CV), SVR with GridSearchCV
- **Classification models**: Logistic Regression, K-Nearest Neighbors, SVC with GridSearchCV
- **Best model**: Automatically selects the best model by R² (regression) or accuracy (classification)
- **Export**: Save the best model as a `.joblib` file and metrics to a `.txt` file

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt) for package versions

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Elfer09/ML-model-selector.git
   cd ml-model-selector
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. When prompted, enter **`regression`** or **`classification`**.

3. Select your CSV file via the file dialog.

4. Choose the **target variable** from the dropdown.

5. If the app finds missing values or categorical columns, choose whether to:
   - Fill missing values with mean (numeric) or mode (categorical), and/or  
   - Encode categorical columns with one-hot encoding (dummies).

6. Wait for training to finish. The best model and metrics are shown in dialogs and in the console.

7. If you accept the suggested best model, enter a name for the saved model (e.g. `my_model`). The app will create:
   - `my_model.joblib` – trained model
   - `my_model_metrics.txt` – best model name and metrics (e.g. R², MAE, RMSE or accuracy, classification report)

## Project structure

```
ML_model_selector/
├── main.py                 # Tkinter app: load data, select target, run models, save best
├── train_regression.py     # Regression training and evaluation
├── train_classification.py # Classification training and evaluation
├── requirements.txt
└── README.md
```

## Loading a saved model

```python
import joblib

model = joblib.load("my_model.joblib")

# For prediction, use the same features (and scaling) as during training.
# Scale new data with the same StandardScaler used in training if applicable.
predictions = model.predict(X_new_scaled)
```

**Note:** The app saves only the estimator. For regression, if you used scaling or polynomial features, you must apply the same preprocessing to new data before calling `predict`.

## License

MIT