"""
train_model.py

Script de entrenamiento y evaluación del modelo final
Práctica 2 – Modelado y Evaluación (MBID)

A partir del conjunto de datos procesado, este script entrena
el modelo seleccionado, evalúa su rendimiento y registra los
resultados y artefactos en MLflow para asegurar la trazabilidad.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import json
import joblib
from pathlib import Path
import warnings

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def load_data(data_path="data/processed/datos_integrados.csv"):
    """Carga datos y genera splits train / validation / test."""

    df = pd.read_csv(data_path)
    target = "falta_pago"

    X = df.drop(columns=[target])
    y = df[target]

    # Test final (10 %)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )

    # Train + validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.22, stratify=y_temp, random_state=42
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, X


def create_preprocessor(features_X):
    """Crea el preprocesador de variables numéricas y categóricas."""

    num_cols = features_X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = features_X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    return preprocessor


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def train_model(
    data_path="data/processed/datos_integrados.csv",
    model_output_path="models/prod_model.pkl",
    preprocessor_output_path="models/prod_preprocessor.pkl",
    metrics_output_path="metrics/train_metrics.json",
):
    print(">>> Entrando en train_model()")

    # ---------------- MLflow ----------------
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("13MBID_Modelo_Final_MVP")

    # ---------------- Datos -----------------
    X_train, y_train, X_val, y_val, X_test, y_test, features_X = load_data(data_path)

    # ---------------- Pipeline --------------
    modelo = LogisticRegression(max_iter=5000)
    preprocessor = create_preprocessor(features_X)

    pipeline = ImbPipeline(
        steps=[
            ("prep", preprocessor),
            ("undersample", RandomUnderSampler(random_state=42)),
            ("model", modelo),
        ]
    )

    pipeline.fit(X_train, y_train)

    # ---------------- Evaluación -------------
    y_test_pred = pipeline.predict(X_test)
    y_test_score = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision_macro": precision_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_recall_macro": recall_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_f1_macro": f1_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_roc_auc": roc_auc_score(y_test, y_test_score),
    }

    # Imprimir (COMO EL PROFE)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # ---------------- Matriz de confusión ----
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")

    fig_path = Path("docs/figures/confusion_matrix.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.title("Matriz de Confusión - LogisticRegression")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    # ---------------- MLflow -----------------
    signature = infer_signature(X_train, pipeline.predict(X_train))

    with mlflow.start_run(run_name="Pipeline (prod) - LogisticRegression"):
        mlflow.log_params(modelo.get_params())
        mlflow.log_params({
            "train_samples": len(X_train),
            "validation_samples": len(y_val),
            "test_samples": len(X_test),
            "balancing_method": "undersampling",
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(fig_path))
        mlflow.sklearn.log_model(
            pipeline, artifact_path="model", signature=signature
        )

        print("Modelo registrado en MLflow")

    # ---------------- Guardado ---------------
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(preprocessor_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_output_path)
    joblib.dump(pipeline.named_steps["prep"], preprocessor_output_path)

    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Artefactos guardados correctamente")


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    print(">>> Ejecutando script train_model.py")
    train_model()
