import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from tkinter import messagebox

def train_svr(X_train, y_train): 
    """
    Trains an SVR model with hyperparameter tuning using GridSearchCV.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training target.

    Returns:
        GridSearchCV: Fitted GridSearchCV object with the best SVR model.
    """
    param_grid = {
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "degree": np.arange(1, 11),
        "gamma": ['scale', 'auto'],
        "C": np.logspace(0, 1, 10),
        "epsilon": [0, 0.01, 0.1, 0.5, 1]
    }
    
    grid_search = GridSearchCV(SVR(), param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    return grid_search

def evaluate_model(model, X_train, y_train, X_test, y_test, is_polynomial=False, \
                   poly_transformer=None):
    """
    Evaluates a trained model on test data and computes performance metrics.
    
    Parameters:
    - model: The trained model or GridSearchCV instance
    - X_train, y_train: Training data
    - X_test, y_test: Testing data
    - is_polynomial: Boolean flag to indicate polynomial transformation
    - poly_transformer: Transformer object if using polynomial features
    
    Returns:
    - Dictionary containing MAE, RMSE, R2 Score, and best parameters (if applicable)
    """
    
    if isinstance(model, GridSearchCV):
        model.fit(X_train, y_train)
        best_model_instance = model.best_estimator_
        best_params = model.best_params_
    else:
        model.fit(X_train, y_train)
        best_model_instance = model
        best_params = None

    # Transform test data if polynomial features are used
    if is_polynomial:
        X_test_transformed = poly_transformer.transform(X_test)
        y_pred = best_model_instance.predict(X_test_transformed)
    else:
        y_pred = best_model_instance.predict(X_test)

    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
        "Best Parameters": best_params
    }


def train_regression_models(df, target_variable):
    """
    Trains multiple regression models on a given dataset and evaluates performance.
    
    Parameters:
    - df: Pandas DataFrame containing the dataset
    - target_variable: Column name of the target variable
    
    Returns:
    - Best model name, instance, R2 score, and testing data
    """
    
    # Inform the user that training is starting for the given target variable
    print(f"\n¤ Training Regression Models for Target Variable: {target_variable}...\n")

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.remove(target_variable)
    if not numerical_cols:
        messagebox.showerror("Error", "No numeric feature columns found. Need at least one.")
        raise ValueError("No numeric feature columns.")

    X = df[numerical_cols]  
    y = df[target_variable]  

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train baseline Linear Regression
    linear_model = LinearRegression()
    linear_metrics = evaluate_model(linear_model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Train Polynomial Regression (Degree=2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)

    poly_model = LinearRegression()
    poly_metrics = evaluate_model(poly_model,
                                  X_train_poly,
                                  y_train,
                                  X_test_scaled,
                                  y_test,
                                  is_polynomial=True,
                                  poly_transformer=poly)

    # Decision: Keep polynomial regression if it improves R² by at least 5%
    use_polynomial = poly_metrics["R2 Score"] > linear_metrics["R2 Score"] * 1.05

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression (CV)": LassoCV(eps=0.001,
                                        max_iter=10000,
                                        cv=10),
        "Ridge Regression (CV)": RidgeCV(alphas=[0.1, 0.2, 0.5, 1.0, 5.0, 10.0],
                                        scoring="r2"),
        "Elastic Net (CV)": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                                        eps=0.001,
                                        max_iter=10000,
                                        cv=10),
        "Support Vector Regressor (SVR)": train_svr(X_train_scaled, y_train)
    }
    
    if use_polynomial:
        models["Polynomial Regression (Degree=2)"] = (poly_model, poly)

    results = {}
    best_model = None
    best_r2 = -np.inf  
    best_y_pred = None  

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n¤ Training {name}...")

        if name == "Polynomial Regression (Degree=2)":
            model, poly_transformer = model
            metrics = evaluate_model(model,
                                     X_train_poly,
                                     y_train,
                                     X_test_scaled,
                                     y_test,
                                     is_polynomial=True,
                                     poly_transformer=poly_transformer)
        else:
            metrics = evaluate_model(model,
                                     X_train_scaled,
                                     y_train,
                                     X_test_scaled,
                                     y_test)

        results[name] = metrics

        if metrics["R2 Score"] > best_r2:
            best_r2 = metrics["R2 Score"]
            if name == "Polynomial Regression (Degree=2)":
                # Return a pipeline so predict() applies poly transform then model
                predictor = Pipeline([("poly", poly_transformer), ("model", model)])
                best_model = (name, predictor, best_r2)
            else:
                best_model = (name, model, best_r2)
            best_y_pred = model.predict(poly_transformer.transform(X_test_scaled)) \
                if name == "Polynomial Regression (Degree=2)" \
                else model.predict(X_test_scaled)

    best_model_name, best_model_instance, best_r2 = best_model

    # Performance Comparison Graph
    plt.figure(figsize=(10, 5))
    for metric in ["MAE", "RMSE", "R2 Score"]:
        values = [res[metric] for res in results.values()]
        plt.bar(results.keys(), values, alpha=0.7, label=metric)

    plt.xticks(rotation=45)
    plt.legend()
    plt.title("Model Performance Comparison")
    plt.show(block=False)
    
    # Residual Plot for the Best Model
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=best_y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", \
        linestyle="dashed")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Residual Plot: {best_model_name}")
    plt.show(block=False)

    # Generate Report
    report = "**Regression Model Report**\n\n"
    for name, res in results.items():
        report += f"¤ **{name}**\n"
        if res["Best Parameters"]:
            report += f"   - ¤ Best Parameters: {res['Best Parameters']}\n"
        report += f"   - ¤ MAE: {res['MAE']:.4f}\n"
        report += f"   - ¤ RMSE: {res['RMSE']:.4f}\n"
        report += f"   - ¤ R² Score: {res['R2 Score']:.4f}\n\n"

    report += f"¤ **Best Model: {best_model_name} with R² Score = {best_r2:.4f}**\n"

    # Show messagebox with report
    messagebox.showinfo("Regression Model Report", report)
    
    return best_model_name, best_model_instance, best_r2, X_test_scaled, y_test