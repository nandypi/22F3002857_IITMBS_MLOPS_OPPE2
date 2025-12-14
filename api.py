from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import time
import json
import joblib
import pandas as pd
import os

# -------------------------------
# OpenTelemetry (MANUAL TRACING ONLY)
# -------------------------------
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# -------------------------------
# Structured JSON Logging
# -------------------------------
logger = logging.getLogger("mlops-oppe2")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record),
        }
        for key in ["event", "trace_id", "latency_ms", "path", "error"]:
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)
        return json.dumps(log_obj)

handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="Heart Disease Prediction API")

model = None
features = None

app_state = {
    "ready": False,
    "alive": True
}

# -------------------------------
# Input Schema
# -------------------------------
class InputSchema(BaseModel): 
    age: int
    gender: int 
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int 
    thalach: float
    exang: int
    oldpeak : float
    slope: int
    ca: int
    thal:int    


# -------------------------------
# Startup: Load Artifacts
# -------------------------------
@app.on_event("startup")
async def startup():
    global model, features
    try:
        model = joblib.load("artifacts/model.pkl")
        features = joblib.load("artifacts/features.pkl")

        app_state["ready"] = True
        logger.info("Artifacts loaded successfully", extra={"event": "startup"})

    except Exception as e:
        app_state["ready"] = False
        logger.error(
            "Failed to load artifacts",
            extra={"event": "startup_error", "error": str(e)}
        )
        raise e

# -------------------------------
# Health Probes
# -------------------------------
@app.get("/live_check")
def liveness():
    if app_state["alive"]:
        return {"status": "alive"}
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check")
def readiness():
    if app_state["ready"]:
        return {"status": "ready"}
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.get("/")
def home():
    return {"message": "MLOps OPPE-2 Inference Service"}

# -------------------------------
# Middleware: Latency
# -------------------------------
@app.middleware("http")
async def add_latency(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = round((time.time() - start) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(latency)
    return response

# -------------------------------
# Prediction Endpoint
# -------------------------------
def normalize_input(input_dict):
    MAP = {
        "gender": {"male": 1, "female": 0},
        "fbs": {"yes": 1, "no": 0},
        "exang": {"yes": 1, "no": 0},
        "thal": {"fixed": 0, "normal": 1, "reversible": 2},
    }

    normalized = {}

    for key, val in input_dict.items():
        if key in MAP:
            if isinstance(val, str):
                val = val.lower()
                if val not in MAP[key]:
                    raise ValueError(f"Invalid value '{val}' for {key}")
                normalized[key] = MAP[key][val]
            else:
                normalized[key] = int(val)
        else:
            normalized[key] = float(val)

    return normalized


@app.post("/predict")
async def predict(data: InputSchema, request: Request):
    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")

    with tracer.start_as_current_span("model_inference") as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
        start = time.time()

        try:
            input_dict = data.dict()
            
            input_df = pd.DataFrame(
                [[input_dict[col] for col in features]],
                columns=features
            )
            
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]


            
            latency = round((time.time() - start) * 1000, 2)

            logger.info(
                "Prediction successful",
                extra={
                    "event": "prediction_success",
                    "trace_id": trace_id,
                    "latency_ms": latency,
                    "path": str(request.url)
                }
            )

            return {
                "prediction": prediction,
                "probability": round(probability, 4),
                "trace_id": trace_id
            }

        except Exception as e:
            logger.exception(
                "Prediction failed",
                extra={
                    "event": "prediction_error",
                    "trace_id": trace_id,
                    "error": str(e),
                    "path": str(request.url)
                }
            )
            raise HTTPException(status_code=500, detail="Prediction failed")
