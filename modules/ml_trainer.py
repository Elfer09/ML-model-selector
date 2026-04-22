"""
ml_trainer.py
Trains multiple sklearn models, compares them, and returns results.
Evolved from the original desktop ML-model-selector logic — now web-native.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Elastic Net": ElasticNet(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True, random_state=42),
}

# Hyperparameter grids for GridSearchCV tuning mode.
# Keys use "model__" prefix to match the Pipeline step name.
REGRESSION_PARAM_GRIDS = {
    "Linear Regression": {},
    "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
    "Elastic Net": {"model__alpha": [0.1, 1.0, 10.0], "model__l1_ratio": [0.2, 0.5, 0.8]},
    "Random Forest": {"model__n_estimators": [50, 100], "model__max_depth": [None, 5, 10]},
    "Gradient Boosting": {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1, 0.2]},
    "SVR": {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"], "model__epsilon": [0.01, 0.1]},
}

CLASSIFICATION_PARAM_GRIDS = {
    # Only tune C — lbfgs solver (default) does not support l1 penalty
    "Logistic Regression": {"model__C": [0.1, 1.0, 10.0, 100.0]},
    "Random Forest": {"model__n_estimators": [50, 100], "model__max_depth": [None, 5, 10]},
    "Gradient Boosting": {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1, 0.2]},
    # Cap at 7 so small inner-fold training sets don't raise ValueError
    "KNN": {"model__n_neighbors": [3, 5, 7]},
    "SVC": {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]},
}


def _build_pipeline(model):
    """Wrap any model with imputation + scaling for robustness."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def _run_with_gridsearch(pipe, param_grid, X_train, X_test, y_train, y_test, task_type, primary_scoring):
    """Fit GridSearchCV on training split, evaluate best estimator on test split."""
    if param_grid:
        gs = GridSearchCV(pipe, param_grid, cv=3, scoring=primary_scoring, n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        best_params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    else:
        pipe.fit(X_train, y_train)
        best = pipe
        best_params = {}

    y_pred = best.predict(X_test)

    if task_type == "Regression":
        return {
            "R² Score": round(r2_score(y_test, y_pred), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "Best Params": str(best_params) if best_params else "—",
        }
    else:
        return {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "Best Params": str(best_params) if best_params else "—",
        }


def train_and_compare(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task_type: str,
    use_tuning: bool = False,
) -> tuple[pd.DataFrame, str, go.Figure, list[str]]:
    """
    Train all models and compare performance.
    use_tuning=False  → cross-validation with fixed hyperparameters (fast)
    use_tuning=True   → 70/30 split + GridSearchCV per model (slower, tuned)
    Returns: (results_df, best_model_name, plotly_fig, warnings)
    """
    warnings = []
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop rows where target is null and report how many were lost
    mask = y.notna()
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        warnings.append(f"{n_dropped} row(s) dropped due to missing target values.")
    X, y = X[mask], y[mask]

    # Minimum rows needed for any cross-validation
    if len(y) < 10:
        raise ValueError(
            f"Only {len(y)} usable row(s) after dropping missing targets — need at least 10."
        )

    # Drop feature columns that are entirely NaN (SimpleImputer cannot handle them)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        warnings.append(f"Dropped all-NaN feature column(s): {', '.join(all_nan_cols)}")
        X = X.drop(columns=all_nan_cols)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after dropping all-NaN columns.")

    # Encode categorical target for classification
    if task_type == "Classification" and y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    # Classification-specific validation
    if task_type == "Classification":
        n_classes = int(y.nunique())
        if n_classes < 2:
            raise ValueError("Target column has only 1 unique class — need at least 2 for classification.")
        if n_classes > 50:
            warnings.append(
                f"Target has {n_classes} unique values — this may be a regression problem, not classification."
            )

    # Adapt CV folds to dataset size (minimum 3, maximum 5)
    n_folds = min(5, max(3, len(y) // 10))
    if n_folds < 5:
        warnings.append(f"Using {n_folds}-fold CV (dataset is small — {len(y)} rows).")

    models = REGRESSION_MODELS if task_type == "Regression" else CLASSIFICATION_MODELS

    # Use StratifiedKFold for classification to preserve class distribution per fold
    cv_strategy = (
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        if task_type == "Classification"
        else n_folds
    )

    if task_type == "Regression":
        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }
        primary_metric = "R² Score"
    else:
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
        }
        primary_metric = "Accuracy"

    results = []
    skipped = []

    if use_tuning:
        # --- GridSearchCV mode: 70/30 split, tune each model, evaluate on held-out test set ---
        stratify = y if task_type == "Classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=stratify
        )
        param_grids = (
            REGRESSION_PARAM_GRIDS if task_type == "Regression" else CLASSIFICATION_PARAM_GRIDS
        )
        primary_scoring = "r2" if task_type == "Regression" else "accuracy"

        for name, model in models.items():
            pipe = _build_pipeline(model)
            try:
                metrics = _run_with_gridsearch(
                    pipe, param_grids.get(name, {}),
                    X_train, X_test, y_train, y_test,
                    task_type, primary_scoring,
                )
                results.append({"Model": name, **metrics})
            except Exception as e:
                skipped.append(f"{name} ({e})")

        chart_subtitle = "GridSearchCV, 70/30 split"

    else:
        # --- Fast mode: cross-validation with fixed hyperparameters ---
        for name, model in models.items():
            pipe = _build_pipeline(model)
            try:
                cv = cross_validate(pipe, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
                if task_type == "Regression":
                    results.append({
                        "Model": name,
                        "R² Score": round(cv["test_r2"].mean(), 4),
                        "MAE": round(-cv["test_mae"].mean(), 4),
                        "RMSE": round(-cv["test_rmse"].mean(), 4),
                        "Std Dev (R²)": round(cv["test_r2"].std(), 4),
                    })
                else:
                    results.append({
                        "Model": name,
                        "Accuracy": round(cv["test_accuracy"].mean(), 4),
                        "Precision": round(cv["test_precision"].mean(), 4),
                        "Recall": round(cv["test_recall"].mean(), 4),
                        "F1 Score": round(cv["test_f1"].mean(), 4),
                    })
            except Exception as e:
                skipped.append(f"{name} ({e})")

        chart_subtitle = f"{n_folds}-fold CV"

    if skipped:
        warnings.append(f"Skipped model(s): {'; '.join(skipped)}")

    if not results:
        raise ValueError("All models failed to train. Check your data for issues.")

    results_df = pd.DataFrame(results).sort_values(primary_metric, ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]

    # Build horizontal bar chart using primary metric
    colors = ["#00c7a8" if i == 0 else "#4a9eff" for i in range(len(results_df))]
    error_x = None
    if "Std Dev (R²)" in results_df.columns:
        error_x = dict(type="data", array=results_df["Std Dev (R²)"].tolist())

    fig = go.Figure(
        go.Bar(
            x=results_df[primary_metric],
            y=results_df["Model"],
            orientation="h",
            marker_color=colors,
            error_x=error_x,
            text=results_df[primary_metric].round(4),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Model Comparison — {primary_metric} ({chart_subtitle})",
        xaxis_title=primary_metric,
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=60, t=50, b=20),
    )

    return results_df, best_model_name, fig, warnings