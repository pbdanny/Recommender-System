import bentoml
import redis
import json
import os
from typing import Dict, Any

# 1. Define the Service with a Decorator (Modern Way)
@bentoml.service(name="hybrid_recsys")
class HybridRecSys:
    
    # 2. Use __init__ for setup (Better resource management)
    def __init__(self):
        # We determine the Redis host at startup
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        print(f"Connecting to Redis at {redis_host}...")
        
        # Keep the connection alive in the class instance
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=6379, 
            db=0, 
            decode_responses=True
        )

    # 3. Define the API using the new @bentoml.api decorator
    # Note: We use standard Python type hints instead of bentoml.io.JSON()
    @bentoml.api
    def recommend(self, user_id: str) -> Dict[str, Any]:
        
        try:
            # Use the class instance's redis client
            candidates_json = self.redis_client.get(f"rec:batch:{user_id}")
        except redis.exceptions.ConnectionError:
            print(f"Warning: Could not connect to Redis. Returning fallback.")
            candidates_json = None
        
        if not candidates_json:
             # Fallback logic
            return {"user_id": user_id, "recommendations": ["fallback_1", "fallback_2"]}
        
        candidates = json.loads(candidates_json)
        
        # Simple re-ranking logic (Take top 10)
        final_recs = candidates[:10]
        
        return {"user_id": user_id, "recommendations": final_recs}