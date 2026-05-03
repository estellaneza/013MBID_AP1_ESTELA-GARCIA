import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# -------------------------------------------------
# Función principal de preparación de datos
# -------------------------------------------------
def process_data(
    datos_creditos: str = "data/raw/datos_creditos.csv",
    datos_tarjetas: str = "data/raw/datos_tarjetas.csv",
    output_dir: str = "data/processed"
):
    """
    Lee los datos de créditos y tarjetas, realiza limpieza,
    transformación y feature engineering, y guarda el dataset final.
    """

    # -------------------------------------------------
    # Rutas
    # -------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Lectura de datos
    # -------------------------------------------------
    df_creditos = pd.read_csv(datos_creditos, sep=";")
    df_tarjetas = pd.read_csv(datos_tarjetas, sep=";")

    # -------------------------------------------------
    # Limpieza básica - Créditos
    # -------------------------------------------------
    df_creditos_filtrado = df_creditos.copy()

    # Filtro de edad
    df_creditos_filtrado = df_creditos_filtrado[
        df_creditos_filtrado["edad"] < 90
    ]

    # Imputación tasa_interes (mediana por objetivo_credito)
    df_creditos_filtrado["tasa_interes"] = (
        df_creditos_filtrado
        .groupby("objetivo_credito")["tasa_interes"]
        .transform(lambda x: x.fillna(x.median()))
    )

    # Imputación antiguedad_empleado (mediana por edad)
    df_creditos_filtrado["antiguedad_empleado"] = (
        df_creditos_filtrado
        .groupby("edad")["antiguedad_empleado"]
        .transform(lambda x: x.fillna(x.median()))
    )

    # Eliminación final de nulos residuales
    df_creditos_filtrado.dropna(inplace=True)

    # -------------------------------------------------
    # Integración Crédito + Tarjetas
    # -------------------------------------------------
    df_integrado = pd.merge(
        df_creditos_filtrado,
        df_tarjetas,
        on="id_cliente",
        how="inner"
    )

    # -------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------

    # Capacidad de pago
    df_integrado["capacidad_pago"] = (
        df_integrado["importe_solicitado"] / df_integrado["ingresos"]
    )

    # Estabilidad laboral
    df_integrado["estabilidad_laboral"] = (
        df_integrado["antiguedad_empleado"] / df_integrado["edad"]
    )

    # Operaciones mensuales
    df_integrado["operaciones_mensuales"] = (
        df_integrado["operaciones_ult_12m"] / 12
    )

    # Gasto medio mensual
    df_integrado["gasto_medio_mensual"] = (
        df_integrado["gastos_ult_12m"] / 12
    )

    # Gasto promedio por operación
    df_integrado["gasto_promedio_operacion"] = (
        df_integrado["gastos_ult_12m"] / df_integrado["operaciones_ult_12m"]
    )

    # -------------------------------------------------
    # Eliminación de columnas redundantes
    # -------------------------------------------------
    columnas_a_eliminar = [
        "id_cliente",
        "operaciones_ult_12m",
        "importe_solicitado",
        "duracion_credito",
        "gastos_ult_12m",
        "antiguedad_empleado"
    ]

    df_integrado.drop(columns=columnas_a_eliminar, inplace=True)

    # -------------------------------------------------
    # Exportación
    # -------------------------------------------------
    output_file = output_path / "datos_integrados.csv"
    df_integrado.to_csv(output_file, index=False)

    # -------------------------------------------------
    # Log final
    # -------------------------------------------------
    print("✅ Dataset procesado correctamente")
    print(f"📁 Ruta de salida: {output_file}")
    print(f"📊 Filas finales: {df_integrado.shape[0]}")
    print(f"📊 Columnas finales: {df_integrado.shape[1]}")


# -------------------------------------------------
# Ejecución directa
# -------------------------------------------------
if __name__ == "__main__":
    process_data()
    