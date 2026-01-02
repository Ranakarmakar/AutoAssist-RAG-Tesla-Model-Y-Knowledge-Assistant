"""Test script for new metrics endpoints."""

import requests
import time
import json
from pathlib import Path


def test_metrics_endpoints():
    """Test the new metrics endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing metrics endpoints...")
    
    # Test basic metrics endpoint
    print("\n1. Testing /metrics endpoint...")
    response = requests.get(f"{base_url}/metrics")
    if response.status_code == 200:
        data = response.json()
        print("✓ Basic metrics endpoint working")
        print(f"  - System uptime: {data.get('system', {}).get('uptime_seconds', 0):.2f}s")
        print(f"  - Memory usage: {data.get('system', {}).get('memory_usage_mb', 0):.2f}MB")
        print(f"  - Active requests: {data.get('system', {}).get('active_requests', 0)}")
        
        # Check if performance metrics are included
        if 'performance_metrics' in data:
            print("  - Performance metrics included ✓")
        else:
            print("  - Performance metrics missing ✗")
    else:
        print(f"✗ Basic metrics endpoint failed: {response.status_code}")
    
    # Test performance metrics endpoint
    print("\n2. Testing /metrics/performance endpoint...")
    response = requests.get(f"{base_url}/metrics/performance")
    if response.status_code == 200:
        data = response.json()
        print("✓ Performance metrics endpoint working")
        
        # Check sections
        sections = ['performance_summary', 'timing_metrics', 'counter_metrics', 'error_metrics']
        for section in sections:
            if section in data:
                print(f"  - {section}: ✓")
            else:
                print(f"  - {section}: ✗")
    else:
        print(f"✗ Performance metrics endpoint failed: {response.status_code}")
    
    # Test system metrics endpoint
    print("\n3. Testing /metrics/system endpoint...")
    response = requests.get(f"{base_url}/metrics/system")
    if response.status_code == 200:
        data = response.json()
        print("✓ System metrics endpoint working")
        
        system_health = data.get('system_health', {})
        print(f"  - System status: {system_health.get('status', 'unknown')}")
        print(f"  - CPU usage: {system_health.get('metrics', {}).get('system_cpu_percent', 0):.1f}%")
        print(f"  - Memory usage: {system_health.get('metrics', {}).get('system_memory_percent', 0):.1f}%")
    else:
        print(f"✗ System metrics endpoint failed: {response.status_code}")
    
    # Test workflow metrics endpoint
    print("\n4. Testing /metrics/workflow endpoint...")
    response = requests.get(f"{base_url}/metrics/workflow")
    if response.status_code == 200:
        data = response.json()
        print("✓ Workflow metrics endpoint working")
        
        workflow_timings = data.get('workflow_timing_metrics', {})
        workflow_counters = data.get('workflow_counter_metrics', {})
        
        print(f"  - Workflow timing metrics: {len(workflow_timings)}")
        print(f"  - Workflow counter metrics: {len(workflow_counters)}")
    else:
        print(f"✗ Workflow metrics endpoint failed: {response.status_code}")
    
    # Test API metrics endpoint
    print("\n5. Testing /metrics/api endpoint...")
    response = requests.get(f"{base_url}/metrics/api")
    if response.status_code == 200:
        data = response.json()
        print("✓ API metrics endpoint working")
        
        api_timings = data.get('api_timing_metrics', {})
        api_counters = data.get('api_counter_metrics', {})
        
        print(f"  - API timing metrics: {len(api_timings)}")
        print(f"  - API counter metrics: {len(api_counters)}")
        print(f"  - Active requests: {data.get('active_requests', 0)}")
        print(f"  - Total requests: {data.get('total_requests', 0)}")
    else:
        print(f"✗ API metrics endpoint failed: {response.status_code}")
    
    # Generate some activity to test metrics collection
    print("\n6. Generating activity to test metrics collection...")
    
    # Make a few API calls
    for i in range(3):
        print(f"  Making test query {i+1}...")
        query_data = {
            "query": f"Test query {i+1} for metrics collection",
            "include_citations": True
        }
        response = requests.post(f"{base_url}/query", json=query_data)
        if response.status_code == 200:
            print(f"    ✓ Query {i+1} successful")
        else:
            print(f"    ✗ Query {i+1} failed: {response.status_code}")
        
        time.sleep(0.5)  # Small delay between requests
    
    # Check metrics after activity
    print("\n7. Checking metrics after activity...")
    response = requests.get(f"{base_url}/metrics/api")
    if response.status_code == 200:
        data = response.json()
        
        api_counters = data.get('api_counter_metrics', {})
        if 'api_requests_total' in api_counters:
            total_requests = api_counters['api_requests_total']['total_count']
            print(f"  - Total API requests recorded: {total_requests}")
        
        if 'api_query_requests' in api_counters:
            query_requests = api_counters['api_query_requests']['total_count']
            print(f"  - Query requests recorded: {query_requests}")
        
        api_timings = data.get('api_timing_metrics', {})
        if 'api_process_query' in api_timings:
            query_timing = api_timings['api_process_query']
            print(f"  - Average query time: {query_timing['average_time']:.3f}s")
            print(f"  - Query count: {query_timing['count']}")
    
    # Test metrics reset
    print("\n8. Testing metrics reset...")
    response = requests.delete(f"{base_url}/metrics")
    if response.status_code == 200:
        print("✓ Metrics reset successful")
        
        # Check that metrics are cleared
        response = requests.get(f"{base_url}/metrics/performance")
        if response.status_code == 200:
            data = response.json()
            timing_metrics = data.get('timing_metrics', {})
            counter_metrics = data.get('counter_metrics', {})
            
            print(f"  - Timing metrics after reset: {len(timing_metrics)}")
            print(f"  - Counter metrics after reset: {len(counter_metrics)}")
    else:
        print(f"✗ Metrics reset failed: {response.status_code}")
    
    print("\n✓ Metrics endpoints testing completed!")


if __name__ == "__main__":
    try:
        test_metrics_endpoints()
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"✗ Test failed with error: {e}")