import os
import json
import redis
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Configuration ---
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = 6379
REDIS_DB = 0

# --- Pydantic Models ---
class RecommendRequest(BaseModel):
    user_id: str

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[str]

# --- Global State ---
redis_client: Optional[redis.Redis] = None

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client
    print(f"Connecting to Redis at {REDIS_HOST}...")
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )
    # Ping to check connection (optional but good for logs)
    try:
        redis_client.ping()
        print("Successfully connected to Redis.")
    except redis.ConnectionError:
        print(f"Warning: Could not connect to Redis at {REDIS_HOST}.")
    
    yield
    
    # Shutdown
    print("Closing Redis connection...")
    if redis_client:
        redis_client.close()

# --- App Definition ---
app = FastAPI(title="Hybrid RecSys Service", lifespan=lifespan)

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    user_id = request.user_id
    candidates_json = None
    
    if redis_client:
        try:
            candidates_json = redis_client.get(f"rec:batch:{user_id}")
        except redis.ConnectionError:
            print("Redis connection error during request.")
    
    if not candidates_json:
        # Fallback logic
        return RecommendResponse(
            user_id=user_id,
            recommendations=["fallback_1", "fallback_2"]
        )
    
    try:
        candidates = json.loads(candidates_json)
        # Simple re-ranking or selection logic (Take top 10)
        final_recs = candidates[:10]
        
        return RecommendResponse(
            user_id=user_id,
            recommendations=final_recs
        )
    except json.JSONDecodeError:
        print(f"Error decoding JSON for user {user_id}")
        return RecommendResponse(
            user_id=user_id,
            recommendations=["fallback_error"]
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
