"""
main.py
"""

import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from forecasting import build_forecast
import model as model_module
from aggregation import build_batch_summary
from model import predict
from schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    PredictRequest,
    SentimentResult,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Nigerian e-commerce review sentiment service", version="1.0.0")


@app.on_event("startup")
def preload_model() -> None:
    model_module.model.load()


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    details = exc.errors()
    message = details[0]["msg"] if details else "invalid request body"
    return JSONResponse(status_code=400, content={"detail": message})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    status = model_module.get_status()
    return HealthResponse(status="ok", model=status["model_name"], mode=status["mode"])


@app.post("/predict", response_model=SentimentResult)
def predict_single(payload: PredictRequest):
    try:
        result = predict(payload.text)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive catch
        raise HTTPException(status_code=500, detail=f"model inference failed: {str(exc)}") from exc


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest):
    try:
        results = [predict(item) for item in payload.texts]
        return BatchPredictResponse(
            results=results, summary=build_batch_summary(payload.texts, results)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive catch
        raise HTTPException(status_code=500, detail=f"batch inference failed: {str(exc)}") from exc


@app.post("/forecast", response_model=ForecastResponse)
def forecast(payload: ForecastRequest):
    try:
        return build_forecast(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive catch
        raise HTTPException(status_code=500, detail=f"forecast generation failed: {str(exc)}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
