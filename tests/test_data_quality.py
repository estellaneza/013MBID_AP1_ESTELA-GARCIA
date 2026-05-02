import pandas as pd
import pytest
from pandera.pandas import DataFrameSchema, Column, Check

# =====================================================
# FIXTURES: CARGA DE DATOS
# =====================================================

@pytest.fixture
def datos_creditos():
    return pd.read_csv("data/raw/datos_creditos.csv", sep=";")


@pytest.fixture
def datos_tarjetas():
    return pd.read_csv("data/raw/datos_tarjetas.csv", sep=";")

# =====================================================
# INSPECCIÓN PREVIA (INFO)
# =====================================================

def test_info_datos_creditos(datos_creditos):
    print("\nINFO DATASET CRÉDITOS")
    print(datos_creditos.info())


def test_info_datos_tarjetas(datos_tarjetas):
    print("\nINFO DATASET TARJETAS")
    print(datos_tarjetas.info())

# =====================================================
# EXACTITUD – TESTS BÁSICOS (CRÉDITOS)
# =====================================================

def test_basicos_creditos(datos_creditos):
    df = datos_creditos
    assert not df.empty, "El dataset de créditos está vacío."
    assert df.shape[1] == 12, "Número incorrecto de columnas en créditos."
    assert df.isnull().sum().sum() == 0, "Existen valores nulos en créditos."

# =====================================================
# EXACTITUD – ESQUEMA (CRÉDITOS)
# =====================================================

def test_esquema_datos_creditos(datos_creditos):
    esquema = DataFrameSchema({
        "edad": Column(int, Check.greater_than_or_equal_to(18)),
        "importe_solicitado": Column(float, Check.greater_than(0)),
        "duracion_credito": Column(int, Check.greater_than(0)),
        "antiguedad_empleado": Column(float, nullable=True),
        "situacion_vivienda": Column(str),
        "objetivo_credito": Column(str),
        "pct_ingreso": Column(float),
        "tasa_interes": Column(float),
        "estado_credito": Column(int),
        "ingresos": Column(float),
        "falta_pago": Column(object)
    })
    esquema.validate(datos_creditos)

# =====================================================
# COMPLETITUD – CRÉDITOS
# =====================================================

def test_completitud_creditos(datos_creditos):
    porcentaje_nulos = datos_creditos.isna().mean()
    print("\nCOMPLETITUD CRÉDITOS")
    print(porcentaje_nulos)
    assert (porcentaje_nulos <= 0.05).all(), "Completitud insuficiente en créditos."

# =====================================================
# CONSISTENCIA – UNICIDAD ID (CRÉDITOS)
# =====================================================

def test_unicidad_id_creditos(datos_creditos):
    duplicados = datos_creditos["id_cliente"].duplicated().sum()
    assert duplicados == 0, "Existen id_cliente duplicados en créditos."

# =====================================================
# EXACTITUD – TESTS BÁSICOS (TARJETAS)
# =====================================================

def test_basicos_tarjetas(datos_tarjetas):
    df = datos_tarjetas
    assert not df.empty, "El dataset de tarjetas está vacío."
    assert df.shape[1] == 11, "Número incorrecto de columnas en tarjetas."
    assert df.isnull().sum().sum() == 0, "Existen valores nulos en tarjetas."

# =====================================================
# EXACTITUD – ESQUEMA (TARJETAS)
# =====================================================

def test_esquema_datos_tarjetas(datos_tarjetas):
    esquema = DataFrameSchema({
        "id_cliente": Column(float),
        "antiguedad_cliente": Column(float),
        "estado_civil": Column(str),
        "estado_cliente": Column(str),
        "gastos_ult_12m": Column(float),
        "genero": Column(str),
        "limite_credito_tc": Column(float),
        "nivel_educativo": Column(str),
        "nivel_tarjeta": Column(str),
        "operaciones_ult_12m": Column(float),
        "personas_a_cargo": Column(float)
    })
    esquema.validate(datos_tarjetas)

# =====================================================
# COMPLETITUD – TARJETAS
# =====================================================

def test_completitud_tarjetas(datos_tarjetas):
    porcentaje_nulos = datos_tarjetas.isna().mean()
    print("\nCOMPLETITUD TARJETAS")
    print(porcentaje_nulos)
    assert (porcentaje_nulos <= 0.05).all(), "Completitud insuficiente en tarjetas."

# =====================================================
# CONSISTENCIA – UNICIDAD ID (TARJETAS)
# =====================================================

def test_unicidad_id_tarjetas(datos_tarjetas):
    duplicados = datos_tarjetas["id_cliente"].duplicated().sum()
    assert duplicados == 0, "Existen id_cliente duplicados en tarjetas."

# =====================================================
# CONSISTENCIA – INTEGRIDAD REFERENCIAL (PROFE STYLE)
# =====================================================

def test_integridad_referencial(datos_creditos, datos_tarjetas):
    df_ids = datos_creditos[["id_cliente"]].merge(
        datos_tarjetas[["id_cliente"]],
        on="id_cliente",
        how="outer",
        indicator=True
    )

    integridad_schema = DataFrameSchema({
        "_merge": Column(str, Check.isin(["both"]))
    })

    integridad_schema.validate(df_ids)