from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import joblib

# ============================
# Inicialización FastAPI
# ============================
app = FastAPI(
    title="Modelo de Predicción de Mora en Créditos",
    description=(
        "Una API para predecir la probabilidad de mora en créditos "
        "utilizando un modelo de machine learning entrenado con datos históricos."
    ),
    version="1.0.0",
)

# ============================
# Esquema de entrada
# (COINCIDE 100% con datos_integrados.csv
#  + valores por defecto reales)
# ============================
class PredictionRequest(BaseModel):
    edad: int = 21
    situacion_vivienda: str = "PROPIA"
    ingresos: int = 9600
    objetivo_credito: str = "EDUCACIÓN"
    pct_ingreso: float = 0.10
    tasa_interes: float = 11.14
    estado_credito: int = 0
    antiguedad_cliente: float = 39.0
    estado_civil: str = "CASADO"
    estado_cliente: str = "ACTIVO"
    genero: str = "M"
    limite_credito_tc: float = 12691.0
    nivel_educativo: str = "SECUNDARIO_COMPLETO"
    nivel_tarjeta: str = "Blue"
    personas_a_cargo: float = 3.0
    capacidad_pago: float = 0.104167
    estabilidad_laboral: float = 0.238095
    operaciones_mensuales: float = 3.5
    gasto_medio_mensual: float = 95.333333
    gasto_promedio_operacion: float = 27.238095

# ============================
# Esquema de salida
# ============================
class PredictionResponse(BaseModel):
    prediction: str
    probability: Dict[str, float]
    class_labels: Dict[str, str]
    model_info: Dict[str, str]

# ============================
# Carga del modelo
# ============================
MODEL_PATH = "models/prod_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# ============================
# Endpoints
# ============================

# Entrar a / abre Swagger directamente
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"} if model else {"status": "error"}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")

    try:
        # DataFrame con TODAS las columnas esperadas por el modelo
        df = pd.DataFrame([req.dict()])

        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        classes = model.named_steps["model"].classes_
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error en la inferencia: {e}")

    return PredictionResponse(
        prediction=str(prediction),
        probability={
            str(classes[i]): float(probabilities[i])
            for i in range(len(classes))
        },
        class_labels={
            "N": "No entra en mora",
            "Y": "Entra en mora"
        },
        model_info={
            "model_type": type(model.named_steps["model"]).__name__,
            "version": "1.0.0"
        }
    )
