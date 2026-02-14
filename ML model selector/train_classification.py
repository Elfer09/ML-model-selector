import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tkinter import messagebox



def plot_feature_analysis(df, target_variable, feature_columns):
    """Plots boxplots, scatterplots, and 3D visualizations using the given feature columns."""
    if not feature_columns:
        return
    # Boxplot: one feature by target
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=target_variable, y=feature_columns[0])
    plt.title("Boxplot of Features by Target Variable")
    plt.show(block=False)

    # Scatterplot (need at least 2 features)
    if len(feature_columns) >= 2:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=feature_columns[0], y=feature_columns[1], hue=target_variable, alpha=0.6)
        plt.title("Feature Relationship Scatterplot")
        plt.show(block=False)

    # 3D plot (need at least 3 features)
    if len(feature_columns) >= 3:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(df[feature_columns[0]], df[feature_columns[1]], df[feature_columns[2]], c=df[target_variable])
        ax.set_xlabel(feature_columns[0])
        ax.set_ylabel(feature_columns[1])
        ax.set_zlabel(feature_columns[2])
        plt.title("3D Data Representation")
        plt.show(block=False)

def train_classification_models(df, target_variable):
    """
    Trains classification models (Logistic Regression, KNN, SVC) using GridSearchCV 
    for hyperparameter tuning.
    
    Parameters:
    df (DataFrame): The dataset containing features and target variable.
    target_variable (str): The column name of the target variable.
    
    Returns:
    tuple: Best model name, best trained model, best accuracy score, scaled test
    set, and test target values.
    """    
    print(f"\n Training Classification Models for Target Variable: {target_variable}...\n")

    # Work on a copy so the original DataFrame is not mutated
    df = df.copy()

    # Encode the target variable into numeric
    encoder = LabelEncoder()
    df[target_variable] = encoder.fit_transform(df[target_variable])
    class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(f" Target Variable Encoding: {class_mapping}")

    # Extract numerical feature columns (after encoding, target is numeric too)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.remove(target_variable)
    if not numerical_cols:
        messagebox.showerror("Error", "No numeric feature columns found. Need at least one.")
        raise ValueError("No numeric feature columns.")

    # Feature analysis (uses actual feature columns, with guards for 2D/3D)
    plot_feature_analysis(df, target_variable, numerical_cols)

    X = df[numerical_cols]  
    y = df[target_variable]  

    # Split dataset into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models & hyperparameters
    models = {
        "Logistic Regression": (LogisticRegression(solver="saga", max_iter=5000),
                                {
                                    "C": np.linspace(0.1, 4, 10),
                                    "penalty": ['l1', 'l2']
                                }
                            ),
        "K-Nearest Neighbors": (KNeighborsClassifier(), {"n_neighbors": range(1, 30)}),
        "Support Vector Classifier (SVC)": (SVC(probability=True), \
            {"C": [0.1, 1, 10, 100], "kernel": ['linear', 'rbf']})
    }
    
    # Dictionary to store evaluation results
    results = {} 
    best_model_instance = None
    best_model = None
    # Initialize best accuracy tracker
    best_accuracy = 0  

    #  Train, tune and evaluate each model
    for name, (model, params) in models.items():
        print(f"\n Training {name}...")

        # Perform hyperparameter tuning with GridSearchCV
        grid_search = GridSearchCV(model, params, cv=10, scoring="accuracy")
        grid_search.fit(X_train_scaled, y_train)
        best_model_instance = grid_search.best_estimator_

        # Predictions on the test set
        y_pred = best_model_instance.predict(X_test_scaled)

        # Evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        class_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results[name] = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Best Parameters": grid_search.best_params_,
            "Classification Report": class_report,
            "Confusion Matrix": conf_matrix
        }

        # Update best model if current model has higher accuracy
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model_instance = best_model_instance
            

    # Plot accuracy comparison for different models
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=[res["Accuracy"] for res in \
        results.values()], palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Model Accuracy Comparison")
    plt.show(block=False)

    # Confusion matrix for best model
    ConfusionMatrixDisplay.from_estimator(best_model_instance, X_test_scaled, y_test, cmap="Blues")
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.show(block=False)

    # Generate a classification report summary
    report = " **Classification Model Report** \n\n"
    for name, res in results.items():
        report += f" ¤ **{name}**\n"
        report += f"   - ¤ Best Parameters: {res['Best Parameters']}\n"
        report += f"   - ¤ Accuracy: {res['Accuracy']:.4f}\n"
        report += f"   - ¤ Precision: {res['Precision']:.4f}\n"
        report += f"   - ¤ Recall: {res['Recall']:.4f}\n"
        report += f"   - ¤ F1 Score: {res['F1 Score']:.4f}\n\n"

    report += f"¤ **Best Model: {best_model_name} with Accuracy = {best_accuracy:.4f}**\n"

    messagebox.showinfo("Classification Model Report", report)

    return best_model_name, best_model_instance, best_accuracy, X_test_scaled, y_test
