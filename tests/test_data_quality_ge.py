import pandas as pd
import pytest
from pathlib import Path

# =====================================================
# TEST DE CALIDAD DE DATOS – GREAT EXPECTATIONS STYLE
# Alineado con ISO 25012 / DMBoK
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"


def test_great_expectations():
    """
    Validación de calidad de los datasets de créditos y tarjetas
    mediante expectations explícitas adaptadas a datos reales.
    """

    # -------------------------------------------------
    # Carga de datos
    # -------------------------------------------------
    df_creditos = pd.read_csv(DATA_DIR / "datos_creditos.csv", sep=";")
    df_tarjetas = pd.read_csv(DATA_DIR / "datos_tarjetas.csv", sep=";")

    # -------------------------------------------------
    # Estructura de resultados
    # -------------------------------------------------
    results = {
        "success": True,
        "expectations": [],
        "statistics": {
            "success_count": 0,
            "total_count": 0
        }
    }

    def add_expectation(name, condition, message):
        results["statistics"]["total_count"] += 1
        if condition:
            results["statistics"]["success_count"] += 1
            results["expectations"].append({
                "expectation": name,
                "success": True
            })
        else:
            results["success"] = False
            results["expectations"].append({
                "expectation": name,
                "success": False,
                "message": message
            })

    # =================================================
    # CREDITOS
    # =================================================

    add_expectation(
        "creditos_no_vacio",
        not df_creditos.empty,
        "El dataset de créditos está vacío"
    )

    add_expectation(
        "edad_mayoritariamente_valida",
        df_creditos["edad"].between(18, 100).mean() >= 0.95,
        "Más del 5% de edades fuera del rango 18–100"
    )

    add_expectation(
        "importe_solicitado_positivo",
        (df_creditos["importe_solicitado"] > 0).mean() >= 0.99,
        "Existen importes solicitados no positivos"
    )

    add_expectation(
        "tasa_interes_completitud",
        df_creditos["tasa_interes"].notna().mean() >= 0.85,
        "Más del 15% de tasa_interes son nulos"
    )

    valores_validos_vivienda = {
        "ALQUILER",
        "PROPIA",
        "HIPOTECA",
        "OTROS"
    }

    add_expectation(
        "situacion_vivienda_dom_valido",
        df_creditos["situacion_vivienda"]
        .astype(str)
        .str.upper()
        .isin(valores_validos_vivienda)
        .mean() >= 0.95,
        "Existen valores fuera del dominio de situacion_vivienda"
    )

    # =================================================
    # TARJETAS (VALIDACIONES EXTRA)
    # =================================================

    add_expectation(
        "tarjetas_no_vacio",
        not df_tarjetas.empty,
        "El dataset de tarjetas está vacío"
    )

    # ✅ VALIDACIÓN EXTRA 1 – rango límite de crédito
    add_expectation(
        "limite_credito_rango_realista",
        df_tarjetas["limite_credito_tc"]
        .between(500, 200000)
        .mean() >= 0.95,
        "Más del 5% de límites de crédito fuera de rango"
    )

    # ✅ VALIDACIÓN EXTRA 2 – estado civil (ROBUSTA Y DEFINITIVA)
    estado_civil_normalizado = (
        df_tarjetas["estado_civil"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    estados_civiles_validos = {
        "SINGLE",
        "MARRIED",
        "DIVORCED",
        "WIDOWED",
        "UNKNOWN"
    }

    add_expectation(
        "estado_civil_dom_valido",
        estado_civil_normalizado
        .isin(estados_civiles_validos)
        .mean() >= 0.70,
        "El estado civil presenta alta heterogeneidad de valores"
    )

    add_expectation(
        "gastos_no_negativos",
        (df_tarjetas["gastos_ult_12m"] >= 0).mean() >= 0.99,
        "Existen gastos negativos"
    )

    add_expectation(
        "operaciones_no_negativas",
        (df_tarjetas["operaciones_ult_12m"] >= 0).mean() >= 0.99,
        "Existen operaciones negativas"
    )

    # =================================================
    # VALIDACIÓN FINAL
    # =================================================
    if not results["success"]:
        failed = [
            e for e in results["expectations"]
            if not e["success"]
        ]
        pytest.fail(
            f"Fallaron {len(failed)} expectations: {failed}"
        )