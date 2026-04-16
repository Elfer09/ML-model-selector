"""
ml_trainer.py
Trains multiple sklearn models, compares them, and returns results.
Evolved from the original desktop ML-model-selector logic — now web-native.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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


def _build_pipeline(model):
    """Wrap any model with imputation + scaling for robustness."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def train_and_compare(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task_type: str,
) -> tuple[pd.DataFrame, str, go.Figure]:
    """
    Train all models via cross-validation.
    Returns: (results_df, best_model_name, plotly_fig)
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop rows where target is null
    mask = y.notna()
    X, y = X[mask], y[mask]

    models = REGRESSION_MODELS if task_type == "Regression" else CLASSIFICATION_MODELS
    scoring = "r2" if task_type == "Regression" else "accuracy"
    metric_label = "R² Score" if task_type == "Regression" else "Accuracy"

    results = []
    for name, model in models.items():
        pipe = _build_pipeline(model)
        try:
            scores = cross_val_score(pipe, X, y, cv=5, scoring=scoring, n_jobs=-1)
            results.append({
                "Model": name,
                metric_label: round(scores.mean(), 4),
                "Std Dev": round(scores.std(), 4),
            })
        except Exception:
            # Skip models that fail (e.g. SVC on multiclass with wrong config)
            continue

    results_df = pd.DataFrame(results).sort_values(metric_label, ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]

    # Build horizontal bar chart
    colors = ["#00c7a8" if i == 0 else "#4a9eff" for i in range(len(results_df))]
    fig = go.Figure(
        go.Bar(
            x=results_df[metric_label],
            y=results_df["Model"],
            orientation="h",
            marker_color=colors,
            error_x=dict(type="data", array=results_df["Std Dev"].tolist()),
            text=results_df[metric_label].round(4),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Model Comparison — {metric_label} (5-fold CV)",
        xaxis_title=metric_label,
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=60, t=50, b=20),
    )

    return results_df, best_model_name, fig
