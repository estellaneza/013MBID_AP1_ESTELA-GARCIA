from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Dict
import pandas as pd
import joblib

# =============================================================================
# Inicialización FastAPI
# =============================================================================

app = FastAPI(
    title="Modelo de Predicción de Mora en Créditos",
    description=(
        "API para predecir la probabilidad de mora en créditos "
        "utilizando un modelo de machine learning entrenado"
    ),
    version="1.0.0",
)

# =============================================================================
# INPUT
# =============================================================================

class PredictionRequest(BaseModel):
    edad: int = Field(..., ge=18)
    situacion_vivienda: str
    ingresos: float = Field(..., ge=0)
    objetivo_credito: str
    pct_ingreso: float = Field(..., ge=0, le=1)
    tasa_interes: float = Field(..., ge=0)
    estado_credito: int = Field(..., ge=0, le=1)
    antiguedad_cliente: float = Field(..., ge=0)
    estado_civil: str
    estado_cliente: str
    genero: str
    limite_credito_tc: float = Field(..., ge=0)
    nivel_educativo: str
    nivel_tarjeta: str
    personas_a_cargo: int = Field(..., ge=0)
    capacidad_pago: float = Field(..., ge=0, le=1)
    estabilidad_laboral: float = Field(..., ge=0, le=1)
    operaciones_mensuales: float = Field(..., ge=0)
    gasto_medio_mensual: float = Field(..., ge=0)
    gasto_promedio_operacion: float = Field(..., ge=0)

# =============================================================================
# Carga del modelo
# =============================================================================

MODEL_PATH = "models/prod_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    model = None

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["default"])
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["default"])
def health():
    return {"status": "ok"} if model else {"status": "error"}

@app.post("/predict", tags=["default"])
def predict(request: PredictionRequest):
    """
    Devuelve:
    - Predicción final (N / Y)
    - Probabilidad de cada clase
    """

    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")

    try:
        df = pd.DataFrame([request.dict()])

        # Predicción
        prediction = model.predict(df)[0]

        # Probabilidades
        proba = model.predict_proba(df)[0]
        classes = model.classes_

        probability_dict: Dict[str, float] = {
            str(classes[i]): float(proba[i])
            for i in range(len(classes))
        }

        return {
            "prediction": prediction,
            "probability": probability_dict,
            "class_labels": {
                "N": "No entra en mora",
                "Y": "Entra en mora"
            },
            "model_info": {
                "model_type": type(model.named_steps["model"]).__name__
                if hasattr(model, "named_steps")
                else type(model).__name__,
                "version": "1.0.0"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
