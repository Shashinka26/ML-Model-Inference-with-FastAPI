from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import joblib

# Load the trained model bundle on startup
try:
    BUNDLE = joblib.load("model.pkl")
    MODEL = BUNDLE["model"]
    CLASSES = BUNDLE["classes"]       # e.g. ["setosa","versicolor","virginica"]
    FEATURE_NAMES = BUNDLE["feature_names"]
    TEST_ACC = BUNDLE.get("test_accuracy", None)
except Exception as e:
    BUNDLE = MODEL = CLASSES = FEATURE_NAMES = TEST_ACC = None
    LOAD_ERR = e
else:
    LOAD_ERR = None

app = FastAPI(
    title="Iris Classifier API",
    description="FastAPI inference server for an Iris flower classifier",
    version="1.0.0",
)

# Input schema (Pydantic)
class PredictionInput(BaseModel):
    sepal_length: float = Field(..., description="Sepal length (cm)")
    sepal_width:  float = Field(..., description="Sepal width (cm)")
    petal_length: float = Field(..., description="Petal length (cm)")
    petal_width:  float = Field(..., description="Petal width (cm)")

class PredictionOutput(BaseModel):
    prediction: str
    confidence: Optional[float] = None

@app.get("/")
def health_check():
    if LOAD_ERR:
        return {
            "status": "error",
            "message": f"Model not loaded: {str(LOAD_ERR)}. "
                       f"Run 'python train_model.py' first to create model.pkl."
        }
    return {"status": "healthy", "message": "ML Model API is running"}

@app.get("/model-info")
def model_info():
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    model_type = type(MODEL.named_steps['clf']).__name__
    return {
        "model_type": model_type,
        "problem_type": "classification",
        "features": FEATURE_NAMES,
        "classes": CLASSES,
        "test_accuracy": TEST_ACC,
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    try:
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]], dtype=float)

        # Prediction with probability
        if hasattr(MODEL.named_steps['clf'], "predict_proba"):
            proba_arr = MODEL.predict_proba(features)[0]
            pred_idx = int(np.argmax(proba_arr))
            pred_label = CLASSES[pred_idx]   # Map index → class name
            proba = float(proba_arr[pred_idx])
        else:
            pred_idx = MODEL.predict(features)[0]
            pred_label = CLASSES[int(pred_idx)]
            proba = None

        return PredictionOutput(prediction=str(pred_label), confidence=proba)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad input or model error: {e}")
