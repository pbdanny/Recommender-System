from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import time
from pathlib import Path
import json
import redis
from contextlib import asynccontextmanager

# Import the existing ModelPredictor which loads the mlflow/Surprise pyfunc and helper
from predict import ModelPredictor, alldataset, get_redis_host

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="Production-ready ML model serving with monitoring and logging (Surprise recommender)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model predictor (adapter)
predictor: Optional[Any] = None

# Prediction statistics
prediction_stats = {
    "total_predictions": 0,
    "total_errors": 0,
    "avg_latency_ms": 0.0,
    "start_time": datetime.now().isoformat()
}


# ========== Redis helpers ==========

def get_redis_client():
    host = get_redis_host()
    try:
        return redis.Redis(host=host, port=6379, db=0, decode_responses=True)
    except Exception as e:
        logger.error(f"Failed to create Redis client: {e}")
        return None

def fetch_cached_recommendations(user: str, k: int):
    """
    Try to fetch cached top-N item list for a user from Redis.
    Returns list of items (possibly shorter than k) or None if missing/error.
    """
    r = get_redis_client()
    if not r:
        return None
    try:
        key = f"rec:batch:{user}"
        val = r.get(key)
        if not val:
            return None
        items = json.loads(val)
        if not isinstance(items, list):
            return None
        return items[:k]
    except Exception as e:
        logger.error(f"Error fetching cache for user {user}: {e}")
        return None


# ========== Adapter for Surprise-based predictor ==========

class SurpriseAdapter:
    """
    Adapter around the mlflow pyfunc predictor that the repo's `predict.ModelPredictor`
    returns. Exposes `.predict(df)` where df has columns `user` and `item`, and a
    convenience `.recommend(user, k)` method to get top-k items for a user.
    """
    def __init__(self, model_name="svd_production_model"):
        self._inner = ModelPredictor(model_name=model_name)
        self.model = self._inner.model
        self.model_name = model_name
        # Build list of all items from the trainset available in predict.py
        # try:
        #     self._trainset = alldataset
        #     self.all_items = [self._trainset.to_raw_iid(i) for i in self._trainset.all_items()]
        # except Exception:
        #     self.all_items = []

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Expects DataFrame with 'user' and 'item' columns.
        Returns numpy array of float scores (predicted ratings).
        """
        if not {'user', 'item'}.issubset(set(df.columns)):
            raise ValueError("DataFrame must contain 'user' and 'item' columns for Surprise predict.")
        preds = self._inner.predict(df)
        return np.array(preds)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Not applicable for Surprise rating prediction. Return placeholder zeros
        to keep compatibility with existing endpoints that expect this method.
        """
        return np.zeros((len(df), 1))

    def recommend(self, user: str, k: int = 10):
        """
        Return top-k recommended items for `user` by scoring all items.
        """
        if not self.all_items:
            raise RuntimeError("No item list available to build recommendations.")
        predict_df = pd.DataFrame({'user': [user] * len(self.all_items), 'item': self.all_items})
        scores = self.predict(predict_df)
        predict_df['score'] = scores
        top = predict_df.sort_values('score', ascending=False).head(k)
        return top['item'].tolist(), top['score'].tolist()


# ========== Pydantic Models (adjusted for recommender) ==========

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_name: Optional[str] = None
    uptime_seconds: float
    total_predictions: int

class PredictionInput(BaseModel):
    """
    For this recommender, `features` should be a list of pairs: [user, item]
    or a list of lists where each row is [user, item].
    If you want top-k recommendations for users, provide rows with only the user (single column)
    and set `recommend_k`.
    """
    features: List[List[Any]] = Field(..., description="2D array rows: [user, item] or [user]")
    feature_names: Optional[List[str]] = Field(None, description="Optional names: ['user','item'] or ['user']")
    recommend_k: Optional[int] = Field(None, description="If set and only `user` provided, return top-k recommendations per user")

    @field_validator("features", mode="before")
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features cannot be empty")
        if not all(isinstance(row, list) for row in v):
            raise ValueError("Features must be a 2D array (list of lists)")
        return v

class SinglePredictionInput(BaseModel):
    """Single prediction: must include 'user' and 'item' keys"""
    user: Any = Field(..., description="User id")
    item: Any = Field(..., description="Item id")

class PredictionOutput(BaseModel):
    """
    For pair predictions: `predictions` is list of floats (predicted ratings).
    For recommendations: `predictions` is list of dicts with `user` and `recommendations`.
    """
    predictions: List[Any]
    prediction_time_ms: float
    model_version: Optional[str] = None

class SinglePredictionOutput(BaseModel):
    prediction: float
    prediction_time_ms: float
    model_version: Optional[str] = None

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    feature_count: Optional[int] = None
    classes: Optional[List[str]] = None
    loaded_at: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str


# ========== Startup/Shutdown Events ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        logger.info("Loading Surprise-based model...")
        predictor = SurpriseAdapter(model_name="svd_production_model")
        logger.info("✓ Model loaded successfully")
        yield
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}")
        predictor = None
        yield
    finally:
        logger.info("Shutting down API...")


# ========== Helper Functions ==========

def log_prediction(input_size: int, latency_ms: float, success: bool):
    global prediction_stats
    prediction_stats["total_predictions"] += input_size
    if not success:
        prediction_stats["total_errors"] += 1
    n = prediction_stats["total_predictions"]
    old_avg = prediction_stats["avg_latency_ms"]
    prediction_stats["avg_latency_ms"] = (old_avg * (n - input_size) + latency_ms) / n

async def save_prediction_log(data: Dict[str, Any]):
    try:
        log_dir = Path("logs/predictions")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
    except Exception as e:
        logger.error(f"Failed to save prediction log: {str(e)}")


# ========== API Endpoints ==========

@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "ML Model Serving API (Surprise recommender)",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start_time = datetime.fromisoformat(prediction_stats["start_time"])
    uptime = (datetime.now() - start_time).total_seconds()
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model_loaded=(predictor.model is not None),
        model_name=str(predictor.model_name),
        uptime_seconds=uptime,
        total_predictions=prediction_stats["total_predictions"]
    )

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start_time = datetime.fromisoformat(prediction_stats["start_time"])
    uptime = (datetime.now() - start_time).total_seconds()
    error_rate = (prediction_stats["total_errors"] / max(prediction_stats["total_predictions"], 1)) * 100
    return {
        "uptime_seconds": uptime,
        "total_predictions": prediction_stats["total_predictions"],
        "total_errors": prediction_stats["total_errors"],
        "error_rate_percent": round(error_rate, 2),
        "avg_latency_ms": round(prediction_stats["avg_latency_ms"], 2),
        "predictions_per_second": round(prediction_stats["total_predictions"] / max(uptime, 1), 2)
    }

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    model = predictor.model
    return ModelInfo(
        model_name=str(predictor.model_name),
        model_type=type(model).__name__,
        feature_count=None,
        classes=None,
        loaded_at=prediction_stats["start_time"]
    )

@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict_batch(
    input_data: PredictionInput,
    background_tasks: BackgroundTasks
):
    """
    Batch prediction for user-item pairs or top-k recommendations for users.
    - To predict ratings: provide rows [user, item] and feature_names ['user','item'] (or omit names and use list-of-lists)
    - To get recommendations: provide rows [user] and set `recommend_k`
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start_time = time.time()
    try:
        # Build DataFrame from input
        if input_data.feature_names:
            df = pd.DataFrame(input_data.features, columns=input_data.feature_names)
        else:
            # infer: if rows are length 2 then assume [user,item]
            first_row = input_data.features[0]
            if len(first_row) == 2:
                df = pd.DataFrame(input_data.features, columns=['user', 'item'])
            elif len(first_row) == 1:
                df = pd.DataFrame([r[0] for r in input_data.features], columns=['user'])
            else:
                raise HTTPException(status_code=400, detail="Unable to infer input format. Use feature_names or provide [user,item] rows.")
        # Case 1: rating predictions for user-item pairs
        if {'user', 'item'}.issubset(set(df.columns)):
            preds = predictor.predict(df)
            latency_ms = (time.time() - start_time) * 1000
            log_prediction(len(df), latency_ms, success=True)
            background_tasks.add_task(save_prediction_log, {
                "timestamp": datetime.now().isoformat(),
                "input_size": len(df),
                "latency_ms": latency_ms,
                "predictions": preds.tolist()
            })
            return PredictionOutput(
                predictions=preds.tolist(),
                prediction_time_ms=round(latency_ms, 2),
                model_version=str(predictor.model_name)
            )
        # Case 2: recommendations for users (try Redis cache first)
        elif 'user' in df.columns and input_data.recommend_k:
            results = []
            k = int(input_data.recommend_k)
            for user in df['user'].tolist():
                cached = fetch_cached_recommendations(user, k)
                if cached:
                    # cached contains items only (no scores); return items and empty scores
                    results.append({"user": user, "recommendations": cached, "scores": []})
                else:
                    # fallback to online scoring
                    items, scores = predictor.recommend(user, k=k)
                    results.append({"user": user, "recommendations": items, "scores": scores})
            latency_ms = (time.time() - start_time) * 1000
            log_prediction(len(df), latency_ms, success=True)
            background_tasks.add_task(save_prediction_log, {
                "timestamp": datetime.now().isoformat(),
                "input_size": len(df),
                "latency_ms": latency_ms,
                "recommend_k": k,
                "results_count": len(results)
            })
            return PredictionOutput(
                predictions=results,
                prediction_time_ms=round(latency_ms, 2),
                model_version=str(predictor.model_name)
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid input for prediction or missing `recommend_k` for recommendations.")
    except Exception as e:
        log_prediction(len(input_data.features), 0, success=False)
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/single", response_model=SinglePredictionOutput, tags=["Predictions"])
async def predict_single(
    input_data: SinglePredictionInput,
    background_tasks: BackgroundTasks
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start_time = time.time()
    try:
        df = pd.DataFrame([{"user": input_data.user, "item": input_data.item}])
        pred = predictor.predict(df)[0]
        latency_ms = (time.time() - start_time) * 1000
        log_prediction(1, latency_ms, success=True)
        background_tasks.add_task(save_prediction_log, {
            "timestamp": datetime.now().isoformat(),
            "input": {"user": input_data.user, "item": input_data.item},
            "prediction": float(pred),
            "latency_ms": latency_ms
        })
        return SinglePredictionOutput(
            prediction=float(pred),
            prediction_time_ms=round(latency_ms, 2),
            model_version=str(predictor.model_name)
        )
    except Exception as e:
        log_prediction(1, 0, success=False)
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/csv", tags=["Predictions"])
async def predict_from_csv(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV")
    start_time = time.time()
    try:
        import io
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if not {'user', 'item'}.issubset(set(df.columns)):
            raise HTTPException(status_code=400, detail="CSV must contain 'user' and 'item' columns for rating predictions.")
        preds = predictor.predict(df)
        latency_ms = (time.time() - start_time) * 1000
        log_prediction(len(df), latency_ms, success=True)
        result_df = df.copy()
        result_df['prediction'] = preds
        results = result_df.to_dict(orient='records')
        return {
            "predictions": results,
            "total_rows": len(df),
            "prediction_time_ms": round(latency_ms, 2)
        }
    except Exception as e:
        logger.error(f"CSV prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/reload", tags=["Model"])
async def reload_model(model_name: Optional[str] = None):
    global predictor
    try:
        if model_name:
            predictor = SurpriseAdapter(model_name=model_name)
        else:
            predictor = SurpriseAdapter(model_name="svd_production_model")
        logger.info(f"✓ Model reloaded from {predictor.model_name}")
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_name": str(predictor.model_name)
        }
    except Exception as e:
        logger.error(f"Failed to reload model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/feature-importance", tags=["Model"])
async def get_feature_importance():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    model = predictor.model
    if not hasattr(model, 'feature_importances_'):
        raise HTTPException(status_code=400, detail="Model does not support feature importance")
    feature_names = (
        model.feature_names_in_.tolist()
        if hasattr(model, 'feature_names_in_')
        else [f"feature_{i}" for i in range(len(model.feature_importances_))]
    )
    importance = {name: float(imp) for name, imp in zip(feature_names, model.feature_importances_)}
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return {
        "feature_importance": sorted_importance,
        "top_5_features": dict(list(sorted_importance.items())[:5])
    }


# ========== Exception Handlers ==========

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ========== Run Server ==========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
