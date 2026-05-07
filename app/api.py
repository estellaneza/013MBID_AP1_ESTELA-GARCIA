from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Dict

# ============================
# Inicialización FastAPI
# ============================
app = FastAPI(
    title="Modelo de Predicción de Mora en Créditos",
    description=(
        "Una API para predecir la probabilidad de mora en créditos "
        "utilizando un modelo de machine learning entrenado con datos históricos."
    ),
    version="1.0.0"
)

# ============================
# Esquema de entrada (EXACTO a datos_integrados.csv)
# ============================
class PredictionRequest(BaseModel):
    edad: int
    situacion_vivienda: str
    ingresos: int
    objetivo_credito: str
    pct_ingreso: float
    tasa_interes: float
    estado_credito: int
    antiguedad_cliente: float
    estado_civil: str
    estado_cliente: str
    genero: str
    limite_credito_tc: float
    nivel_educativo: str
    nivel_tarjeta: str
    personas_a_cargo: float
    capacidad_pago: float
    estabilidad_laboral: float
    operaciones_mensuales: float
    gasto_medio_mensual: float
    gasto_promedio_operacion: float

    class Config:
        json_schema_extra = {
            "example": {
                "edad": 21,
                "situacion_vivienda": "PROPIA",
                "ingresos": 9600,
                "objetivo_credito": "EDUCACION",
                "pct_ingreso": 0.10,
                "tasa_interes": 11.14,
                "estado_credito": 0,
                "antiguedad_cliente": 39.0,
                "estado_civil": "CASADO",
                "estado_cliente": "ACTIVO",
                "genero": "M",
                "limite_credito_tc": 12691.0,
                "nivel_educativo": "SECUNDARIO_COMPLETO",
                "nivel_tarjeta": "Blue",
                "personas_a_cargo": 3.0,
                "capacidad_pago": 0.184167,
                "estabilidad_laboral": 0.238095,
                "operaciones_mensuales": 3.5,
                "gasto_medio_mensual": 95.333333,
                "gasto_promedio_operacion": 27.238095
            }
        }

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

# ✅ OPCIÓN B: entrar a / abre directamente Swagger
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["default"])
def health_check():
    return {"status": "ok"} if model else {"status": "error"}

@app.post("/predict", tags=["default"], response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")

    input_df = pd.DataFrame([request.dict()])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    class_labels_model = model.named_steps["model"].classes_

    return PredictionResponse(
        prediction=str(pred),
        probability={
            str(class_labels_model[i]): float(proba[i])
            for i in range(len(class_labels_model))
        },
        class_labels={
            "N": "No entra en mora",
            "Y": "Entra en mora"
        },
        model_info={
            "model_version": "1.0.0",
            "model_type": type(model.named_steps["model"]).__name__
        }
    )
