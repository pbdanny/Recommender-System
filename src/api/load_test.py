"""
Load testing script for API
"""
import asyncio
import aiohttp
import time
import statistics
from typing import List
import argparse

async def make_request(session: aiohttp.ClientSession, url: str, payload: dict) -> float:
    """Make single request and return latency"""
    start = time.time()
    try:
        async with session.post(url, json=payload) as response:
            await response.json()
            return (time.time() - start) * 1000  # Convert to ms
    except Exception as e:
        print(f"Error: {e}")
        return -1

async def load_test(
    url: str,
    num_requests: int,
    concurrency: int,
    payload: dict
):
    """Run load test"""
    print(f"Starting load test...")
    print(f"  URL: {url}")
    print(f"  Requests: {num_requests}")
    print(f"  Concurrency: {concurrency}")
    print()
    
    latencies: List[float] = []
    errors = 0
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # Create tasks in batches
        for i in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - i)
            tasks = [
                make_request(session, url, payload)
                for _ in range(batch_size)
            ]
            
            results = await asyncio.gather(*tasks)
            
            for latency in results:
                if latency > 0:
                    latencies.append(latency)
                else:
                    errors += 1
            
            # Progress update
            completed = i + batch_size
            print(f"Progress: {completed}/{num_requests} requests", end='\r')
        
        total_time = time.time() - start_time
    
    # Calculate statistics
    print(f"\n\nLoad Test Results:")
    print(f"{'='*50}")
    print(f"Total requests:      {num_requests}")
    print(f"Successful:          {len(latencies)}")
    print(f"Failed:              {errors}")
    print(f"Total time:          {total_time:.2f}s")
    print(f"Requests/sec:        {num_requests/total_time:.2f}")
    print(f"\nLatency Statistics:")
    print(f"  Min:               {min(latencies):.2f}ms")
    print(f"  Max:               {max(latencies):.2f}ms")
    print(f"  Mean:              {statistics.mean(latencies):.2f}ms")
    print(f"  Median:            {statistics.median(latencies):.2f}ms")
    print(f"  StdDev:            {statistics.stdev(latencies):.2f}ms")
    
    # Percentiles
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[int(len(sorted_lat) * 0.50)]
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
    
    print(f"\nPercentiles:")
    print(f"  P50:               {p50:.2f}ms")
    print(f"  P95:               {p95:.2f}ms")
    print(f"  P99:               {p99:.2f}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load test ML API')
    parser.add_argument('--url', default='http://localhost:8000/predict/single')
    parser.add_argument('--requests', type=int, default=1000)
    parser.add_argument('--concurrency', type=int, default=10)
    
    args = parser.parse_args()
    
    payload = {
        "user": "123",
        "item": "456"
        }
    
    asyncio.run(load_test(
        args.url,
        args.requests,
        args.concurrency,
        payload
    ))
