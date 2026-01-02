#!/usr/bin/env python3
"""
Performance Testing Script for PDF RAG System

This script performs comprehensive performance testing:
1. Load testing with concurrent requests
2. Response time analysis
3. Memory and resource monitoring
4. Cache performance evaluation
5. Stress testing with various query types

Usage:
    python examples/performance_test.py --concurrent 5 --queries 20 --duration 60
    python examples/performance_test.py --load-test --users 10 --ramp-up 30
"""

import sys
import requests
import json
import time
import threading
import statistics
from pathlib import Path
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class TestResult:
    """Individual test result."""
    timestamp: float
    response_time: float
    status_code: int
    success: bool
    error: str = ""
    query: str = ""
    answer_length: int = 0
    confidence: str = ""
    citations_count: int = 0
    processing_time_ms: float = 0


class PerformanceTester:
    """Performance testing client for PDF RAG System."""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.results = []
        self.lock = threading.Lock()
        self.start_time = None
        
    def check_health(self):
        """Check API health."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            health = response.json()
            print(f"✅ API server healthy")
            print(f"   Status: {health['status']}")
            print(f"   Memory: {health['memory_usage_mb']:.1f}MB")
            print(f"   Chunks: {health['total_chunks']}")
            return True
        except Exception as e:
            print(f"❌ API server not accessible: {e}")
            return False
    
    def single_query_test(self, query: str, session: requests.Session = None) -> TestResult:
        """Perform a single query test."""
        if session is None:
            session = requests.Session()
        
        start_time = time.time()
        
        try:
            payload = {
                "query": query,
                "include_citations": True,
                "check_sufficiency": True
            }
            
            response = session.post(f"{self.base_url}/query", json=payload, timeout=30)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                return TestResult(
                    timestamp=start_time,
                    response_time=response_time,
                    status_code=response.status_code,
                    success=True,
                    query=query,
                    answer_length=len(result_data.get('answer', '')),
                    confidence=result_data.get('confidence', ''),
                    citations_count=len(result_data.get('citations', [])),
                    processing_time_ms=result_data.get('processing_time_ms', 0)
                )
            else:
                return TestResult(
                    timestamp=start_time,
                    response_time=response_time,
                    status_code=response.status_code,
                    success=False,
                    error=f"HTTP {response.status_code}",
                    query=query
                )
                
        except Exception as e:
            end_time = time.time()
            return TestResult(
                timestamp=start_time,
                response_time=end_time - start_time,
                status_code=0,
                success=False,
                error=str(e),
                query=query
            )
    
    def concurrent_test(self, queries: List[str], concurrent_users: int = 5, iterations: int = 1) -> List[TestResult]:
        """Run concurrent query tests."""
        print(f"\n🚀 Running concurrent test:")
        print(f"   Queries: {len(queries)}")
        print(f"   Concurrent users: {concurrent_users}")
        print(f"   Iterations per query: {iterations}")
        
        all_queries = queries * iterations
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Create session for each worker
            sessions = [requests.Session() for _ in range(concurrent_users)]
            
            future_to_query = {}
            for i, query in enumerate(all_queries):
                session = sessions[i % len(sessions)]
                future = executor.submit(self.single_query_test, query, session)
                future_to_query[future] = query
            
            completed = 0
            for future in as_completed(future_to_query):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 10 == 0 or completed == len(all_queries):
                    print(f"   Progress: {completed}/{len(all_queries)} completed")
        
        return results
    
    def load_test(self, queries: List[str], max_users: int = 10, ramp_up_time: int = 30, test_duration: int = 60) -> List[TestResult]:
        """Run load test with gradual user ramp-up."""
        print(f"\n📈 Running load test:")
        print(f"   Max users: {max_users}")
        print(f"   Ramp-up time: {ramp_up_time}s")
        print(f"   Test duration: {test_duration}s")
        
        results = []
        self.start_time = time.time()
        end_time = self.start_time + test_duration
        
        def worker_thread(worker_id: int, start_delay: float):
            """Worker thread for load testing."""
            time.sleep(start_delay)
            session = requests.Session()
            
            while time.time() < end_time:
                query = queries[worker_id % len(queries)]
                result = self.single_query_test(query, session)
                
                with self.lock:
                    results.append(result)
                
                # Small delay between requests
                time.sleep(0.1)
        
        # Start workers with staggered delays
        threads = []
        for i in range(max_users):
            delay = (i / max_users) * ramp_up_time
            thread = threading.Thread(target=worker_thread, args=(i, delay))
            thread.start()
            threads.append(thread)
        
        # Monitor progress
        while time.time() < end_time:
            elapsed = time.time() - self.start_time
            with self.lock:
                current_results = len(results)
            print(f"   Progress: {elapsed:.1f}s elapsed, {current_results} requests completed")
            time.sleep(5)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return results
    
    def cache_performance_test(self, query: str, iterations: int = 10) -> Dict[str, Any]:
        """Test cache performance by repeating the same query."""
        print(f"\n💾 Testing cache performance:")
        print(f"   Query: {query[:50]}...")
        print(f"   Iterations: {iterations}")
        
        # Clear cache first
        try:
            requests.delete(f"{self.base_url}/cache", timeout=10)
            print("   Cache cleared")
        except:
            print("   Warning: Could not clear cache")
        
        results = []
        session = requests.Session()
        
        for i in range(iterations):
            result = self.single_query_test(query, session)
            results.append(result)
            
            cache_status = "miss" if i == 0 else "hit"
            print(f"   Iteration {i+1}: {result.response_time:.3f}s ({cache_status})")
        
        if len(results) > 1:
            first_time = results[0].response_time
            avg_cached_time = statistics.mean([r.response_time for r in results[1:]])
            speedup = first_time / avg_cached_time if avg_cached_time > 0 else 0
            
            return {
                "first_request_time": first_time,
                "average_cached_time": avg_cached_time,
                "speedup_factor": speedup,
                "results": results
            }
        
        return {"results": results}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Warning: Could not get system metrics: {e}")
            return {}
    
    def analyze_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """Analyze test results and generate statistics."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if successful_results:
            response_times = [r.response_time for r in successful_results]
            processing_times = [r.processing_time_ms for r in successful_results]
            
            analysis = {
                "total_requests": len(results),
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": len(successful_results) / len(results) * 100,
                "response_time_stats": {
                    "min": min(response_times),
                    "max": max(response_times),
                    "mean": statistics.mean(response_times),
                    "median": statistics.median(response_times),
                    "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                    "p95": sorted(response_times)[int(len(response_times) * 0.95)] if len(response_times) > 1 else response_times[0],
                    "p99": sorted(response_times)[int(len(response_times) * 0.99)] if len(response_times) > 1 else response_times[0]
                },
                "processing_time_stats": {
                    "min": min(processing_times),
                    "max": max(processing_times),
                    "mean": statistics.mean(processing_times),
                    "median": statistics.median(processing_times)
                } if processing_times else {},
                "confidence_distribution": {},
                "error_distribution": {}
            }
            
            # Confidence distribution
            for result in successful_results:
                conf = result.confidence
                analysis["confidence_distribution"][conf] = analysis["confidence_distribution"].get(conf, 0) + 1
            
            # Error distribution
            for result in failed_results:
                error = result.error
                analysis["error_distribution"][error] = analysis["error_distribution"].get(error, 0) + 1
            
            # Calculate throughput if we have timing data
            if results:
                test_duration = max(r.timestamp for r in results) - min(r.timestamp for r in results)
                if test_duration > 0:
                    analysis["throughput_rps"] = len(results) / test_duration
                else:
                    analysis["throughput_rps"] = 0
        
        return analysis
    
    def generate_report(self, test_name: str, results: List[TestResult], analysis: Dict[str, Any], additional_data: Dict = None) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            "test_name": test_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis": analysis,
            "raw_results": [
                {
                    "timestamp": r.timestamp,
                    "response_time": r.response_time,
                    "success": r.success,
                    "status_code": r.status_code,
                    "error": r.error,
                    "query": r.query[:100] + "..." if len(r.query) > 100 else r.query,
                    "answer_length": r.answer_length,
                    "confidence": r.confidence,
                    "citations_count": r.citations_count,
                    "processing_time_ms": r.processing_time_ms
                }
                for r in results
            ]
        }
        
        if additional_data:
            report.update(additional_data)
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str):
        """Save report to JSON file."""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📄 Report saved to: {filename}")
    
    def plot_results(self, results: List[TestResult], title: str = "Performance Test Results"):
        """Create performance visualization plots."""
        if not results:
            return
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            print("No successful results to plot")
            return
        
        # Prepare data
        timestamps = [(r.timestamp - successful_results[0].timestamp) for r in successful_results]
        response_times = [r.response_time for r in successful_results]
        processing_times = [r.processing_time_ms / 1000 for r in successful_results]  # Convert to seconds
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Response time over time
        axes[0, 0].plot(timestamps, response_times, 'b-', alpha=0.7)
        axes[0, 0].set_title('Response Time Over Time')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Response Time (seconds)')
        axes[0, 0].grid(True)
        
        # Response time histogram
        axes[0, 1].hist(response_times, bins=20, alpha=0.7, color='blue')
        axes[0, 1].set_title('Response Time Distribution')
        axes[0, 1].set_xlabel('Response Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True)
        
        # Processing time comparison
        axes[1, 0].scatter(processing_times, response_times, alpha=0.6)
        axes[1, 0].set_title('Processing Time vs Total Response Time')
        axes[1, 0].set_xlabel('Processing Time (seconds)')
        axes[1, 0].set_ylabel('Total Response Time (seconds)')
        axes[1, 0].grid(True)
        
        # Confidence distribution
        confidence_counts = {}
        for r in successful_results:
            conf = r.confidence
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        if confidence_counts:
            axes[1, 1].bar(confidence_counts.keys(), confidence_counts.values())
            axes[1, 1].set_title('Confidence Distribution')
            axes[1, 1].set_xlabel('Confidence Level')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def print_analysis(self, analysis: Dict[str, Any]):
        """Print analysis results."""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        print(f"\n📈 Request Statistics:")
        print(f"   Total requests: {analysis['total_requests']}")
        print(f"   Successful: {analysis['successful_requests']}")
        print(f"   Failed: {analysis['failed_requests']}")
        print(f"   Success rate: {analysis['success_rate']:.1f}%")
        
        if 'throughput_rps' in analysis:
            print(f"   Throughput: {analysis['throughput_rps']:.2f} requests/second")
        
        rt_stats = analysis.get('response_time_stats', {})
        if rt_stats:
            print(f"\n⏱️  Response Time Statistics:")
            print(f"   Min: {rt_stats['min']:.3f}s")
            print(f"   Max: {rt_stats['max']:.3f}s")
            print(f"   Mean: {rt_stats['mean']:.3f}s")
            print(f"   Median: {rt_stats['median']:.3f}s")
            print(f"   Std Dev: {rt_stats['std_dev']:.3f}s")
            print(f"   95th percentile: {rt_stats['p95']:.3f}s")
            print(f"   99th percentile: {rt_stats['p99']:.3f}s")
        
        pt_stats = analysis.get('processing_time_stats', {})
        if pt_stats:
            print(f"\n🔧 Processing Time Statistics:")
            print(f"   Min: {pt_stats['min']:.1f}ms")
            print(f"   Max: {pt_stats['max']:.1f}ms")
            print(f"   Mean: {pt_stats['mean']:.1f}ms")
            print(f"   Median: {pt_stats['median']:.1f}ms")
        
        conf_dist = analysis.get('confidence_distribution', {})
        if conf_dist:
            print(f"\n🎯 Confidence Distribution:")
            for conf, count in conf_dist.items():
                print(f"   {conf}: {count} requests")
        
        error_dist = analysis.get('error_distribution', {})
        if error_dist:
            print(f"\n❌ Error Distribution:")
            for error, count in error_dist.items():
                print(f"   {error}: {count} requests")


def load_sample_queries() -> List[str]:
    """Load sample queries for testing."""
    sample_queries = [
        "What is this document about?",
        "What are the main safety features?",
        "How do I maintain this system?",
        "What warranty information is provided?",
        "What are the technical specifications?",
        "How do I troubleshoot problems?",
        "What are the installation requirements?",
        "What are the operating procedures?",
        "What compliance standards are mentioned?",
        "What are the performance characteristics?"
    ]
    
    # Try to load from file if available
    queries_file = Path("examples/sample_queries.txt")
    if queries_file.exists():
        try:
            with open(queries_file, 'r') as f:
                file_queries = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        file_queries.append(line)
                if file_queries:
                    return file_queries
        except Exception as e:
            print(f"Warning: Could not load queries from file: {e}")
    
    return sample_queries


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Performance testing for PDF RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic concurrent test
  python examples/performance_test.py --concurrent 5 --queries 20
  
  # Load test with ramp-up
  python examples/performance_test.py --load-test --users 10 --ramp-up 30 --duration 60
  
  # Cache performance test
  python examples/performance_test.py --cache-test --iterations 10
  
  # Full test suite
  python examples/performance_test.py --full-test
        """
    )
    
    # Test types
    parser.add_argument('--concurrent', type=int, help='Run concurrent test with N users')
    parser.add_argument('--load-test', action='store_true', help='Run load test')
    parser.add_argument('--cache-test', action='store_true', help='Run cache performance test')
    parser.add_argument('--full-test', action='store_true', help='Run full test suite')
    
    # Test parameters
    parser.add_argument('--queries', type=int, default=10, help='Number of queries to test')
    parser.add_argument('--users', type=int, default=5, help='Number of concurrent users for load test')
    parser.add_argument('--ramp-up', type=int, default=10, help='Ramp-up time in seconds')
    parser.add_argument('--duration', type=int, default=30, help='Test duration in seconds')
    parser.add_argument('--iterations', type=int, default=5, help='Iterations for cache test')
    
    # Options
    parser.add_argument('--url', default='http://localhost:8000', help='API server URL')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--plot', action='store_true', help='Generate performance plots')
    
    args = parser.parse_args()
    
    print("🚀 PDF RAG System - Performance Testing")
    print("=" * 50)
    
    # Initialize tester
    tester = PerformanceTester(args.url)
    
    # Check API health
    if not tester.check_health():
        sys.exit(1)
    
    # Load sample queries
    sample_queries = load_sample_queries()
    test_queries = sample_queries[:args.queries]
    
    print(f"📝 Using {len(test_queries)} test queries")
    
    # Run tests based on arguments
    all_results = []
    
    if args.concurrent or args.full_test:
        users = args.concurrent or 5
        print(f"\n🔄 Running concurrent test with {users} users...")
        results = tester.concurrent_test(test_queries, users, 1)
        analysis = tester.analyze_results(results)
        tester.print_analysis(analysis)
        
        if args.plot:
            tester.plot_results(results, f"Concurrent Test ({users} users)")
        
        all_results.extend(results)
    
    if args.load_test or args.full_test:
        print(f"\n📈 Running load test...")
        results = tester.load_test(test_queries, args.users, args.ramp_up, args.duration)
        analysis = tester.analyze_results(results)
        tester.print_analysis(analysis)
        
        if args.plot:
            tester.plot_results(results, f"Load Test ({args.users} users)")
        
        all_results.extend(results)
    
    if args.cache_test or args.full_test:
        print(f"\n💾 Running cache performance test...")
        cache_query = test_queries[0]  # Use first query for cache test
        cache_results = tester.cache_performance_test(cache_query, args.iterations)
        
        if 'speedup_factor' in cache_results:
            print(f"\n📊 Cache Performance Results:")
            print(f"   First request: {cache_results['first_request_time']:.3f}s")
            print(f"   Avg cached: {cache_results['average_cached_time']:.3f}s")
            print(f"   Speedup: {cache_results['speedup_factor']:.1f}x")
        
        all_results.extend(cache_results.get('results', []))
    
    # Save results if requested
    if args.output and all_results:
        analysis = tester.analyze_results(all_results)
        report = tester.generate_report("Performance Test", all_results, analysis)
        tester.save_report(report, args.output)
    
    # Get final system metrics
    print(f"\n🔧 Final system metrics:")
    final_metrics = tester.get_system_metrics()
    if final_metrics:
        system = final_metrics.get('system', {})
        print(f"   Memory usage: {system.get('memory_usage_mb', 0):.1f}MB")
        print(f"   CPU usage: {system.get('cpu_percent', 0):.1f}%")
        print(f"   Total requests: {system.get('total_requests', 0)}")
    
    print("\n✅ Performance testing completed!")


if __name__ == "__main__":
    main()