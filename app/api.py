from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import joblib
import os

# ============================
# Inicialización FastAPI
# ============================
app = FastAPI(
    title="Modelo de Predicción de Mora en Créditos",
    description="API para predecir la probabilidad de mora en créditos",
    version="1.0.0",
)

# ============================
# Esquema de entrada
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

# ============================
# Esquema de salida
# ============================
class PredictionResponse(BaseModel):
    prediction: str
    probability: Dict[str, float]

# ============================
# Carga del modelo y preprocesador (CLAVE)
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATH = os.path.join(MODELS_DIR, "prod_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "prod_preprocessor.pkl")

model = None
preprocessor = None

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    print("✅ Modelo y preprocesador cargados correctamente")
except Exception as e:
    print(f"❌ Error cargando modelo o preprocesador: {e}")
    model = None
    preprocessor = None

# ============================
# Endpoints
# ============================

@app.get("/", tags=["default"])
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["default"])
def health():
    if model is not None and preprocessor is not None:
        return {"status": "ok"}
    return {"status": "error"}

@app.post("/predict", tags=["default"], response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")

    try:
        df = pd.DataFrame([req.dict()])
        X = preprocessor.transform(df)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {e}")

    return PredictionResponse(
        prediction=str(prediction),
        probability={
            "0": float(proba[0]),
            "1": float(proba[1])
        }
    )

