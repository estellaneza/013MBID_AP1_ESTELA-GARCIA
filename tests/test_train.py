from src.train_model import train_model
import json
from pathlib import Path
import pytest


def test_train_model(tmp_path):
    """
    Test para validar que el entrenamiento:
    - se ejecuta correctamente
    - genera un archivo de métricas
    - no degrada las métricas respecto al baseline
    """

    project_root = Path(__file__).resolve().parents[1]

    # -----------------------------
    # Baseline de métricas
    # -----------------------------
    baseline_path = project_root / "metrics" / "train_metrics.json"

    if not baseline_path.exists():
        pytest.skip(
            "Baseline metrics file not found. "
            "Run train_model.py to generate it."
        )

    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    # -----------------------------
    # Ejecutar entrenamiento
    # -----------------------------
    data_path = project_root / "data" / "processed" / "datos_integrados.csv"

    model_output_path = tmp_path / "prod_model.pkl"
    preprocessor_output_path = tmp_path / "prod_preprocessor.pkl"
    metrics_output_path = tmp_path / "train_metrics.json"

    # ✅ Ejecutar SIN esperar retorno
    train_model(
        data_path=str(data_path),
        model_output_path=str(model_output_path),
        preprocessor_output_path=str(preprocessor_output_path),
        metrics_output_path=str(metrics_output_path),
    )

    # -----------------------------
    # Comprobar que se generaron métricas
    # -----------------------------
    assert metrics_output_path.exists(), "No se ha generado el archivo de métricas."

    with open(metrics_output_path, "r") as f:
        metrics = json.load(f)

    # -----------------------------
    # Comparar estructura
    # -----------------------------
    assert set(metrics.keys()) == set(
        baseline.keys()
    ), "Las métricas generadas no coinciden con el baseline."

    # -----------------------------
    # Comparar valores
    # -----------------------------
    atol = 1e-9

    for k in baseline.keys():
        assert metrics[k] == pytest.approx(
            baseline[k], abs=atol
        ), (
            f"Métrica {k} cambió: "
            f"baseline={baseline[k]} "
            f"nueva={metrics[k]}"
        )