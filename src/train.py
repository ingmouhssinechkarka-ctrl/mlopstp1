"""Entraînement des modèles Iris avec suivi MLflow.

- lit data/iris.csv
- entraîne plusieurs modèles (Logistic Regression, Random Forest)
- évalue en validation croisée + jeu de test
- logue paramètres, métriques, matrice de confusion et modèle dans MLflow
- exporte le meilleur modèle dans model/ pour l'API
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET = "Species"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def load_data(path):
    """Charge le CSV et vérifie son schéma."""
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    df = df.dropna(subset=FEATURES + [TARGET])
    return df[FEATURES], df[TARGET].astype(int)


def build_models():
    """Modèles candidats : chacun est un Pipeline (prétraitement + modèle)."""
    return {
        "logistic_regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=200)),
            ]),
            {"C": 1.0, "max_iter": 200},
        ),
        "random_forest": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=42)),
            ]),
            {"n_estimators": 100, "max_depth": 5},
        ),
    }


def evaluate(model, X_train, X_test, y_train, y_test):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, average="macro"),
        "test_recall": recall_score(y_test, y_pred, average="macro"),
        "test_f1": f1_score(y_test, y_pred, average="macro"),
    }, y_pred


def log_confusion_matrix(y_test, y_pred, name):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=CLASS_NAMES, ax=ax, colorbar=False)
    ax.set_title(name)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"confusion_matrix_{name}.png")
        fig.savefig(path, bbox_inches="tight")
        mlflow.log_artifact(path)
    plt.close(fig)


def train(data_path, model_dir, experiment="iris-classification"):
    """Entraîne tous les modèles, logue dans MLflow et exporte le meilleur."""
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    mlflow.set_experiment(experiment)
    results = {}
    for name, (model, params) in build_models().items():
        with mlflow.start_run(run_name=name) as run:
            metrics, y_pred = evaluate(model, X_train, X_test, y_train, y_test)
            mlflow.log_param("model_type", name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            log_confusion_matrix(y_test, y_pred, name)
            mlflow.sklearn.log_model(
                model, name="model",
                input_example=X_test.head(2).astype(float),
                skops_trusted_types=["sklearn.tree._tree.Tree"])
            results[name] = {"model": model, "metrics": metrics,
                             "run_id": run.info.run_id}
            print(f"{name}: {metrics}")

    best_name = max(results, key=lambda n: results[n]["metrics"]["test_f1"])
    best = results[best_name]
    print(f"Meilleur modèle : {best_name}")

    # Export pour l'API (joblib = léger, pas besoin de MLflow dans l'image)
    model_dir = Path(model_dir)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    joblib.dump(best["model"], model_dir / "model.joblib")
    with open(model_dir / "metadata.json", "w") as f:
        json.dump({"model_name": best_name, "run_id": best["run_id"],
                   "features": FEATURES, "classes": CLASS_NAMES,
                   "metrics": best["metrics"]}, f, indent=2)
    return best_name, best["metrics"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "iris.csv"))
    parser.add_argument("--model-dir", default=str(ROOT / "model"))
    args = parser.parse_args()
    mlflow.set_tracking_uri(os.getenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"))
    train(args.data, args.model_dir)


if __name__ == "__main__":
    main()
