import sys
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter.ttk import Combobox
import matplotlib.pyplot as plt
import seaborn as sns
from train_regression import train_regression_models  
from train_classification import train_classification_models
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
import joblib

class MLModelSelectorApp:
    def __init__(self, root):
        """
        Initializes the MLModelSelectorApp, setting up the root window and prompting the user
        to select the type of machine learning task (regression or classification).
        """
        self.root = root
        self.root.title("ML Model Selector")
        # Stores the loaded dataset
        self.df = None  
        # Stores the type of ML task
        self.ml_type = None
        # Stores the selected target variable
        self.target_variable = None

        # Ask user for ML type
        self.ask_ml_type()
    
    def ask_ml_type(self):
        """Asks the user whether they want to perform regression or classification."""

        self.ml_type = simpledialog.askstring(
                                            "Input", "Enter 'regression' or "
                                            "'classification':",parent=self.root
                                            )
        if self.ml_type not in ["regression", "classification"]:
            messagebox.showerror("[E]", "Invalid input. Please enter 'regression' "
                                 "or 'classification'.")
            sys.exit()
        
        # Load CSV file
        self.load_csv()
    
    def load_csv(self):
        """Prompts the user to select a CSV file and loads the dataset."""
        file_name = filedialog.askopenfilename(title="Select CSV File",
                                               filetypes=[("CSV Files", "*.csv"),
                                                          ("All Files", "*.*")])
        if not file_name:
            messagebox.showerror("[E]", "No file selected.")
            sys.exit()
        
        # Load dataset
        self.df = pd.read_csv(file_name)
        messagebox.showinfo("Success", f"Loaded dataset with {self.df.shape[0]} "
                            f"rows and {self.df.shape[1]} columns.")
        
        # Select target variable
        self.select_target_variable()
    
    def select_target_variable(self):
        """Creates a window for the user to select the target variable from the dataset."""
        self.target_window = tk.Toplevel(self.root)
        self.target_window.title("Select Target Variable")
        self.target_window.geometry("500x300+450+250")
        
        tk.Label(self.target_window, text="Select Target Variable:",
                 font=("Times New Roman", 12, "bold")).pack(pady=10)
        self.column_selector = Combobox(self.target_window,
                                        values=list(self.df.columns),
                                        font=("Times New Roman", 12))
        self.column_selector.pack(pady=10)
        
        tk.Button(self.target_window, text="OK",
                  command=self.validate_target_variable,
                  font=("Times New Roman", 12)).pack(pady=10)
        
    def validate_target_variable(self):
        """Validates the selected target variable and checks its compatibility with the chosen ML type."""
        self.target_variable = self.column_selector.get()
        if self.target_variable not in self.df.columns:
            messagebox.showerror("[E]", "Target column not found in the dataset.")
            return
        
        # Temporarily fill NaN values with a placeholder (e.g., mean for numeric columns)
        temp_column = pd.to_numeric(self.df[self.target_variable], errors='coerce')
        
         # If at least one valid number exists, classify as continuous
        if temp_column.notna().sum() > 0:
            target_type = "continuous"
        else:
            target_type = "categorical"
        
        # Check if the target type matches the ML type
        if (self.ml_type == "regression" and target_type == "continuous") or \
           (self.ml_type == "classification" and target_type == "categorical"):
            messagebox.showinfo("Success", f"The selected target {self.target_variable} "
                                f"and it matches your ML type {self.ml_type}.")
        else:
            messagebox.showerror("[E]", f"Warning: The target {self.target_variable} "
                                 f"is {target_type}, and doesn't match {self.ml_type}.")
            return
                
        # Close the target selection window
        self.target_window.destroy()
        
        # Proceed to data readiness check
        self.check_data()
    
    def check_data(self):
        """Checks dataset for missing values and categorical variables, offering to handle them automatically."""
        issues = []
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        missing_columns = missing_values[missing_values > 0].index.tolist()
        if missing_columns:
            # Show the missing data heatmap 
            self.plot_missing_data()           
            response = messagebox.askyesno("Handle Missing Data", 
                                           "Missing data found in columns: "
                                           f"{', '.join(missing_columns)}.\n\nDo "
                                           "you want to fill missing values with mean/mode?"
                                           )
            if response:
                for col in missing_columns:
                    if self.df[col].dtype in [np.float64, np.int64]:
                        self.df[col] = self.df[col].fillna(self.df[col].mean()) 
                    else:
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0])         
                messagebox.showinfo("Success", "Missing values have been filled.")
            else:
                issues.append("Missing values remain in dataset. Please handle them manually.")
            
        # Check for categorical variables 
        categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        if self.target_variable in categorical_columns:
            categorical_columns.remove(self.target_variable)
        
        if categorical_columns:
            # Show categorical distributions
            self.plot_categorical_distribution()            
            response = messagebox.askyesno("Data Preparation", 
                                           "The dataset contains categorical columns: "
                                           f"{', '.join(categorical_columns)}.\n\n"
                                           "Do you want to create dummies automatically?")
            if response:
                self.df = pd.get_dummies(self.df,
                                        columns=categorical_columns,
                                        drop_first=True)
                messagebox.showinfo("Success", "Categorical variables have been encoded.")
            else:
                issues.append("Categorical variables need encoding before training. "
                              "Please handle them manually.")
        
        if issues:
            print("\n".join(issues))
            messagebox.showerror("[E] Data Not Ready", "\n".join(issues))
            sys.exit()
        
        messagebox.showinfo("Success", "Dataset is ready for machine learning!") 
       
        # Move to train ML models
        self.run_models()
        
    def plot_missing_data(self):
        """Plots a heatmap and bar chart of missing data before displaying the messagebox."""
        
        missing_values = self.df.isnull().sum()
        # Keep only columns with missing values
        missing_values = missing_values[missing_values > 0]  

        # No missing values, so no need to plot
        if missing_values.empty:
            return  
        
        # Plot missing data heatmap
        plt.figure(figsize=(10, 5))
        sns.heatmap(self.df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Data Heatmap")
        plt.show(block=False)
        
        # Plot missing values as a bar chart
        plt.figure(figsize=(8, 4))
        sns.barplot(x=missing_values.index,
                    y=missing_values.values,
                    hue=missing_values.index,
                    palette="viridis",
                    legend=False)
        plt.title("Missing Values Per Column")
        plt.xlabel("Columns")
        plt.ylabel("Count of Missing Values")
        plt.xticks(rotation=45)
        plt.show(block=False)
        plt.pause(2)  
    
    def plot_categorical_distribution(self):
        categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            for col in categorical_columns:
                plt.figure(figsize=(8, 4))
                sns.countplot(x=self.df[col])
                plt.title(f"Distribution of {col}")
                plt.xticks(rotation=45)
                plt.show(block=False)
                plt.pause(2)  
                
    def run_models(self):
        """Trains the best ML model based on the selected ML type and dataset."""
        try:
            if self.ml_type == "regression":
                best_model_name, best_model_instance, best_r2, X_test_scaled, y_test = \
                    train_regression_models(self.df, self.target_variable)
                print(f"\n Model training complete! Best model: {best_model_name} "
                      f"with R² Score = {best_r2:.4f}")
                metric_name = "R² Score"
                metric_value = best_r2
            elif self.ml_type == "classification":
                best_model_name, best_model_instance, best_accuracy, X_test_scaled, \
                    y_test = train_classification_models(self.df, self.target_variable)
                print(f"\n Model training complete! Best model: {best_model_name} with "
                      f"accuracy = {best_accuracy:.4f}")
                metric_name = "Accuracy"
                metric_value = best_accuracy
            else:
                messagebox.showerror("Error", "Invalid ML type.")
                return
        except ValueError:
            sys.exit(1)
            
        response = messagebox.askyesno("Model Confirmation", 
                                   f"The best model is {best_model_name} with "
                                   f"{metric_name} = {metric_value:.4f}.\n\n"
                                   "Do you agree with this feedback?")
        # User agrees
        if response:  
            # Ask the user for a model file name
            model_file_name = simpledialog.askstring("Model File Name", 
                                                    "Enter a name for the model file "
                                                    "(without extension):", 
                                                    parent=self.root)
            
            if model_file_name:
                # Define file paths
                model_file_path = f"{model_file_name}.joblib"
                metrics_file_path = f"{model_file_name}_metrics.txt"

                #  Save the model
                joblib.dump(best_model_instance, model_file_path)

                #  Save the model metrics
                with open(metrics_file_path, "w") as f:
                    f.write(f"Best Model: {best_model_name}\n")
                    f.write(f"{metric_name}: {metric_value:.4f}\n")
                    f.write("\nAdditional Metrics:\n")
                    
                    # Regression metrics
                    if self.ml_type == "regression":
                        f.write(f"MAE: {mean_absolute_error(y_test, \
                            best_model_instance.predict(X_test_scaled)):.4f}\n")
                        f.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, \
                            best_model_instance.predict(X_test_scaled))):.4f}\n")
                    
                    # Classification metrics
                    elif self.ml_type == "classification":
                        f.write(classification_report(y_test, \
                            best_model_instance.predict(X_test_scaled)))
                
                messagebox.showinfo("Success", 
                                    f"Model saved to '{model_file_path}'.\n"
                                    f"Metrics saved to '{metrics_file_path}'.")
            else:
                messagebox.showwarning("Warning", "No file name provided. Model not saved.")
        else:
            messagebox.showinfo("Info", "Model not saved. Please review the feedback.")
    

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  
    app = MLModelSelectorApp(root)
    root.mainloop()