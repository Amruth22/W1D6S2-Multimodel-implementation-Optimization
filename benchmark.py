#!/usr/bin/env python3
"""
Benchmark script for the Multimodal Engine library
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from multimodal_engine import MultimodalEngine

# Configuration
NUM_REQUESTS = 10
CONCURRENT_REQUESTS = 5

def make_text_request(engine, prompt, use_cache=True):
    """Make a single text request to the engine"""
    start_time = time.time()
    
    try:
        result = engine.generate_from_text(prompt)
        end_time = time.time()
        
        return {
            "status": "success",
            "response_time": end_time - start_time,
            "cached": result.cached,
            "text_length": len(result.text)
        }
    except Exception as e:
        end_time = time.time()
        return {
            "status": "error",
            "response_time": end_time - start_time,
            "error": str(e)
        }

def benchmark_sequential(engine):
    """Benchmark sequential requests"""
    print("Running sequential benchmark...")
    results = []
    
    for i in range(NUM_REQUESTS):
        prompt = f"Explain concept {i} in computer science"
        result = make_text_request(engine, prompt)
        results.append(result)
        status = "CACHED" if result.get('cached', False) else "NEW"
        print(f"Request {i+1}: {result['response_time']:.2f}s [{status}]")
    
    return results

def benchmark_concurrent(engine):
    """Benchmark concurrent requests"""
    print("\nRunning concurrent benchmark...")
    results = []
    
    def worker(i):
        prompt = f"Explain concept {i} in computer science"
        result = make_text_request(engine, prompt)
        status = "CACHED" if result.get('cached', False) else "NEW"
        print(f"Request {i+1}: {result['response_time']:.2f}s [{status}]")
        return result
    
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(worker, i) for i in range(NUM_REQUESTS)]
        results = [future.result() for future in futures]
    
    return results

def benchmark_with_caching(engine):
    """Benchmark with caching enabled"""
    print("\nRunning benchmark with caching...")
    
    # First request (not cached)
    result1 = make_text_request(engine, "Explain quantum computing")
    print(f"First request: {result1['response_time']:.2f}s [NEW]")
    
    # Second request (should be cached)
    result2 = make_text_request(engine, "Explain quantum computing")
    print(f"Second request: {result2['response_time']:.2f}s [CACHED]")
    
    return [result1, result2]

def benchmark_without_caching(engine):
    """Benchmark without caching"""
    print("\nRunning benchmark without caching...")
    
    # Disable caching
    engine.enable_caching = False
    print("Caching disabled")
    
    results = []
    for i in range(3):  # Fewer requests to save time
        prompt = f"Explain concept {i} in mathematics"
        result = make_text_request(engine, prompt)
        results.append(result)
        print(f"Request {i+1}: {result['response_time']:.2f}s [NEW]")
    
    # Re-enable caching
    engine.enable_caching = True
    
    return results

def calculate_stats(results):
    """Calculate statistics from benchmark results"""
    if not results:
        return {}
    
    successful_results = [r for r in results if r['status'] == 'success']
    response_times = [r['response_time'] for r in successful_results]
    cached_count = sum(1 for r in successful_results if r.get('cached', False))
    
    if not response_times:
        return {}
    
    return {
        "total_requests": len(results),
        "successful_requests": len(successful_results),
        "average_response_time": sum(response_times) / len(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
        "cached_requests": cached_count
    }

def main():
    """Main benchmark function"""
    print("Benchmarking Multimodal Engine Library")
    print("=" * 50)
    
    try:
        # Initialize engine
        engine = MultimodalEngine()
        print(f"Using model: {engine.model_name}")
        print(f"Caching enabled: {engine.enable_caching}")
        print(f"Compression enabled: {engine.enable_compression}")
        print()
        
        # Run benchmarks
        seq_results = benchmark_sequential(engine)
        conc_results = benchmark_concurrent(engine)
        cache_results = benchmark_with_caching(engine)
        no_cache_results = benchmark_without_caching(engine)
        
        # Calculate and display statistics
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        
        seq_stats = calculate_stats(seq_results)
        print("\nSequential Requests:")
        print(f"  Total Requests: {seq_stats.get('total_requests', 0)}")
        print(f"  Successful: {seq_stats.get('successful_requests', 0)}")
        print(f"  Average Response Time: {seq_stats.get('average_response_time', 0):.2f}s")
        print(f"  Min Response Time: {seq_stats.get('min_response_time', 0):.2f}s")
        print(f"  Max Response Time: {seq_stats.get('max_response_time', 0):.2f}s")
        print(f"  Cached Requests: {seq_stats.get('cached_requests', 0)}/{seq_stats.get('successful_requests', 0)}")
        
        conc_stats = calculate_stats(conc_results)
        print("\nConcurrent Requests:")
        print(f"  Total Requests: {conc_stats.get('total_requests', 0)}")
        print(f"  Successful: {conc_stats.get('successful_requests', 0)}")
        print(f"  Average Response Time: {conc_stats.get('average_response_time', 0):.2f}s")
        print(f"  Min Response Time: {conc_stats.get('min_response_time', 0):.2f}s")
        print(f"  Max Response Time: {conc_stats.get('max_response_time', 0):.2f}s")
        print(f"  Cached Requests: {conc_stats.get('cached_requests', 0)}/{conc_stats.get('successful_requests', 0)}")
        
        cache_stats = calculate_stats(cache_results)
        print("\nCaching Benefits:")
        if len(cache_results) >= 2:
            first_time = cache_results[0]['response_time']
            second_time = cache_results[1]['response_time']
            print(f"  First Request: {first_time:.2f}s [NEW]")
            print(f"  Cached Request: {second_time:.2f}s [CACHED]")
            if first_time > 0:
                improvement = (1 - second_time / first_time) * 100
                print(f"  Improvement: {improvement:.1f}%")
        
        no_cache_stats = calculate_stats(no_cache_results)
        print("\nWithout Caching:")
        print(f"  Total Requests: {no_cache_stats.get('total_requests', 0)}")
        print(f"  Successful: {no_cache_stats.get('successful_requests', 0)}")
        print(f"  Average Response Time: {no_cache_stats.get('average_response_time', 0):.2f}s")
        print(f"  Min Response Time: {no_cache_stats.get('min_response_time', 0):.2f}s")
        print(f"  Max Response Time: {no_cache_stats.get('max_response_time', 0):.2f}s")
        
        # Display cache statistics
        cache_stats = engine.get_cache_stats()
        print("\nCache Statistics:")
        print(f"  Cache Hits: {cache_stats['hits']}")
        print(f"  Cache Misses: {cache_stats['misses']}")
        print(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
        print(f"  Cache Size: {cache_stats['cache_size']}")
        
        print("\nBenchmark completed successfully!")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()