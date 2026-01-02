"""Unit tests for metrics collection system."""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from collections import deque

from src.metrics import (
    MetricPoint,
    TimingMetric,
    CounterMetric,
    MetricsCollector,
    get_metrics_collector,
    reset_metrics_collector,
    time_operation,
    record_timing,
    increment_counter,
    record_gauge,
    record_error,
    timed,
    counted,
    error_tracked
)


class TestMetricPoint:
    """Test MetricPoint class."""
    
    def test_metric_point_creation(self):
        """Test creating a metric point."""
        timestamp = time.time()
        labels = {"service": "test", "version": "1.0"}
        
        point = MetricPoint(
            timestamp=timestamp,
            value=42.5,
            labels=labels
        )
        
        assert point.timestamp == timestamp
        assert point.value == 42.5
        assert point.labels == labels
    
    def test_metric_point_to_dict(self):
        """Test converting metric point to dictionary."""
        timestamp = time.time()
        point = MetricPoint(timestamp=timestamp, value=100.0)
        
        result = point.to_dict()
        
        assert result["timestamp"] == timestamp
        assert result["value"] == 100.0
        assert "datetime" in result
        assert result["labels"] == {}


class TestTimingMetric:
    """Test TimingMetric class."""
    
    def test_timing_metric_creation(self):
        """Test creating a timing metric."""
        metric = TimingMetric("test_operation")
        
        assert metric.name == "test_operation"
        assert metric.total_time == 0.0
        assert metric.count == 0
        assert metric.min_time == float('inf')
        assert metric.max_time == 0.0
        assert len(metric.recent_times) == 0
    
    def test_add_timing(self):
        """Test adding timing measurements."""
        metric = TimingMetric("test_operation")
        
        metric.add_timing(1.5)
        metric.add_timing(2.0)
        metric.add_timing(0.5)
        
        assert metric.count == 3
        assert metric.total_time == 4.0
        assert metric.min_time == 0.5
        assert metric.max_time == 2.0
        assert metric.average_time == 4.0 / 3
        assert len(metric.recent_times) == 3
    
    def test_recent_average(self):
        """Test recent average calculation."""
        metric = TimingMetric("test_operation")
        
        metric.add_timing(1.0)
        metric.add_timing(2.0)
        metric.add_timing(3.0)
        
        assert metric.recent_average == 2.0
    
    def test_recent_times_limit(self):
        """Test that recent times are limited."""
        metric = TimingMetric("test_operation")
        
        # Add more than the limit (100)
        for i in range(150):
            metric.add_timing(i)
        
        assert len(metric.recent_times) == 100
        assert metric.count == 150
    
    def test_to_dict(self):
        """Test converting timing metric to dictionary."""
        metric = TimingMetric("test_operation")
        metric.add_timing(1.0)
        metric.add_timing(2.0)
        
        result = metric.to_dict()
        
        assert result["name"] == "test_operation"
        assert result["count"] == 2
        assert result["total_time"] == 3.0
        assert result["average_time"] == 1.5
        assert result["min_time"] == 1.0
        assert result["max_time"] == 2.0


class TestCounterMetric:
    """Test CounterMetric class."""
    
    def test_counter_metric_creation(self):
        """Test creating a counter metric."""
        metric = CounterMetric("test_counter")
        
        assert metric.name == "test_counter"
        assert metric.count == 0
        assert len(metric.recent_counts) == 0
        assert len(metric.recent_timestamps) == 0
    
    def test_increment(self):
        """Test incrementing counter."""
        metric = CounterMetric("test_counter")
        
        metric.increment()
        metric.increment(5)
        
        assert metric.count == 6
        assert len(metric.recent_counts) == 2
        assert sum(metric.recent_counts) == 6
    
    def test_rate_calculation(self):
        """Test rate per minute calculation."""
        metric = CounterMetric("test_counter")
        
        metric.increment(10)
        metric.increment(20)
        
        # Should be 30 per minute (since all within last 60 seconds)
        assert metric.rate_per_minute == 30
    
    @patch('time.time')
    def test_old_entries_cleanup(self, mock_time):
        """Test that old entries are cleaned up."""
        metric = CounterMetric("test_counter")
        
        # Add entry at time 0
        mock_time.return_value = 0
        metric.increment(5)
        
        # Add entry at time 70 (should clean up old entry)
        mock_time.return_value = 70
        metric.increment(10)
        
        assert metric.count == 15  # Total count preserved
        assert metric.rate_per_minute == 10  # Only recent count
    
    def test_to_dict(self):
        """Test converting counter metric to dictionary."""
        metric = CounterMetric("test_counter")
        metric.increment(5)
        metric.increment(3)
        
        result = metric.to_dict()
        
        assert result["name"] == "test_counter"
        assert result["total_count"] == 8
        assert result["rate_per_minute"] == 8
        assert result["recent_count"] == 8


class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector(max_history=10)
    
    def test_collector_creation(self):
        """Test creating metrics collector."""
        collector = MetricsCollector(max_history=100)
        
        assert collector.max_history == 100
        assert collector.start_time <= time.time()
        assert len(collector._timing_metrics) == 0
        assert len(collector._counter_metrics) == 0
        assert len(collector._gauge_metrics) == 0
        assert len(collector._error_counts) == 0
    
    def test_record_timing(self):
        """Test recording timing metrics."""
        self.collector.record_timing("test_op", 1.5)
        self.collector.record_timing("test_op", 2.0)
        
        timing_metrics = self.collector.get_timing_metrics()
        assert "test_op" in timing_metrics
        assert timing_metrics["test_op"]["count"] == 2
        assert timing_metrics["test_op"]["average_time"] == 1.75
    
    def test_time_operation_context_manager(self):
        """Test timing operation context manager."""
        with self.collector.time_operation("test_context"):
            time.sleep(0.01)  # Small delay
        
        timing_metrics = self.collector.get_timing_metrics()
        assert "test_context" in timing_metrics
        assert timing_metrics["test_context"]["count"] == 1
        assert timing_metrics["test_context"]["average_time"] > 0
    
    def test_increment_counter(self):
        """Test incrementing counters."""
        self.collector.increment_counter("test_counter", 5)
        self.collector.increment_counter("test_counter", 3)
        
        counter_metrics = self.collector.get_counter_metrics()
        assert "test_counter" in counter_metrics
        assert counter_metrics["test_counter"]["total_count"] == 8
    
    def test_record_gauge(self):
        """Test recording gauge metrics."""
        self.collector.record_gauge("test_gauge", 42.5)
        self.collector.record_gauge("test_gauge", 43.0)
        
        gauge_metrics = self.collector.get_gauge_metrics()
        assert "test_gauge" in gauge_metrics
        assert len(gauge_metrics["test_gauge"]) == 2
        assert gauge_metrics["test_gauge"][-1]["value"] == 43.0
    
    def test_record_error(self):
        """Test recording errors."""
        self.collector.record_error("ValueError", "test_operation")
        self.collector.record_error("TypeError", "test_operation")
        
        error_metrics = self.collector.get_error_metrics()
        assert "test_operation_ValueError" in error_metrics
        assert "test_operation_TypeError" in error_metrics
        assert error_metrics["test_operation_ValueError"]["total_count"] == 1
    
    def test_gauge_history_limit(self):
        """Test that gauge history is limited."""
        for i in range(15):
            self.collector.record_gauge("test_gauge", i)
        
        gauge_metrics = self.collector.get_gauge_metrics()
        assert len(gauge_metrics["test_gauge"]) == 10  # Limited by max_history
    
    def test_get_recent_gauge_values(self):
        """Test getting recent gauge values."""
        current_time = time.time()
        
        # Add some values
        self.collector.record_gauge("test_gauge", 10.0)
        time.sleep(0.01)
        self.collector.record_gauge("test_gauge", 20.0)
        
        recent_values = self.collector.get_recent_gauge_values("test_gauge", 60)
        assert len(recent_values) == 2
        assert recent_values[-1]["value"] == 20.0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.Process')
    @patch('psutil.disk_usage')
    def test_update_system_metrics(self, mock_disk, mock_process, mock_memory, mock_cpu):
        """Test updating system metrics."""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0, used=1024*1024*1024, available=512*1024*1024)
        
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(rss=256*1024*1024, vms=512*1024*1024)
        mock_process_instance.cpu_percent.return_value = 25.0
        mock_process.return_value = mock_process_instance
        
        mock_disk.return_value = MagicMock(total=100*1024*1024*1024, used=50*1024*1024*1024, free=50*1024*1024*1024)
        
        self.collector.update_system_metrics()
        
        gauge_metrics = self.collector.get_gauge_metrics()
        assert "system_cpu_percent" in gauge_metrics
        assert "system_memory_percent" in gauge_metrics
        assert "process_memory_rss_mb" in gauge_metrics
    
    def test_get_system_health(self):
        """Test getting system health."""
        # Add some system metrics
        self.collector.record_gauge("system_cpu_percent", 75.0)
        self.collector.record_gauge("system_memory_percent", 80.0)
        
        health = self.collector.get_system_health()
        
        assert "status" in health
        assert "uptime_seconds" in health
        assert "metrics" in health
        assert "thresholds" in health
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        # Add some metrics
        self.collector.increment_counter("request_total", 100)
        self.collector.record_error("Error", "operation")
        self.collector.record_timing("response_time", 0.5)
        
        summary = self.collector.get_performance_summary()
        
        assert "uptime_seconds" in summary
        assert "total_requests" in summary
        assert "total_errors" in summary
        assert "error_rate_percent" in summary
        assert "average_response_times" in summary
    
    def test_reset_metrics(self):
        """Test resetting all metrics."""
        # Add some metrics
        self.collector.record_timing("test", 1.0)
        self.collector.increment_counter("test", 1)
        self.collector.record_gauge("test", 1.0)
        self.collector.record_error("Error", "test")
        
        # Verify metrics exist
        assert len(self.collector.get_timing_metrics()) > 0
        assert len(self.collector.get_counter_metrics()) > 0
        assert len(self.collector.get_gauge_metrics()) > 0
        assert len(self.collector.get_error_metrics()) > 0
        
        # Reset and verify empty
        self.collector.reset_metrics()
        
        assert len(self.collector.get_timing_metrics()) == 0
        assert len(self.collector.get_counter_metrics()) == 0
        assert len(self.collector.get_gauge_metrics()) == 0
        assert len(self.collector.get_error_metrics()) == 0
    
    def test_thread_safety(self):
        """Test thread safety of metrics collection."""
        def worker():
            for i in range(100):
                self.collector.increment_counter("thread_test")
                self.collector.record_timing("thread_timing", 0.001)
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        counter_metrics = self.collector.get_counter_metrics()
        timing_metrics = self.collector.get_timing_metrics()
        
        assert counter_metrics["thread_test"]["total_count"] == 500
        assert timing_metrics["thread_timing"]["count"] == 500


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        reset_metrics_collector()
    
    def test_get_metrics_collector(self):
        """Test getting global metrics collector."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2  # Should be singleton
    
    def test_record_timing_function(self):
        """Test global record_timing function."""
        record_timing("global_test", 1.5)
        
        collector = get_metrics_collector()
        timing_metrics = collector.get_timing_metrics()
        
        assert "global_test" in timing_metrics
        assert timing_metrics["global_test"]["count"] == 1
    
    def test_increment_counter_function(self):
        """Test global increment_counter function."""
        increment_counter("global_counter", 5)
        
        collector = get_metrics_collector()
        counter_metrics = collector.get_counter_metrics()
        
        assert "global_counter" in counter_metrics
        assert counter_metrics["global_counter"]["total_count"] == 5
    
    def test_record_gauge_function(self):
        """Test global record_gauge function."""
        record_gauge("global_gauge", 42.0)
        
        collector = get_metrics_collector()
        gauge_metrics = collector.get_gauge_metrics()
        
        assert "global_gauge" in gauge_metrics
        assert gauge_metrics["global_gauge"][-1]["value"] == 42.0
    
    def test_record_error_function(self):
        """Test global record_error function."""
        record_error("TestError", "global_operation")
        
        collector = get_metrics_collector()
        error_metrics = collector.get_error_metrics()
        
        assert "global_operation_TestError" in error_metrics
    
    def test_time_operation_function(self):
        """Test global time_operation function."""
        with time_operation("global_timing"):
            time.sleep(0.01)
        
        collector = get_metrics_collector()
        timing_metrics = collector.get_timing_metrics()
        
        assert "global_timing" in timing_metrics
        assert timing_metrics["global_timing"]["count"] == 1


class TestDecorators:
    """Test metric decorators."""
    
    def setup_method(self):
        """Set up test fixtures."""
        reset_metrics_collector()
    
    def test_timed_decorator(self):
        """Test @timed decorator."""
        @timed("decorated_function")
        def test_function():
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        
        assert result == "result"
        
        collector = get_metrics_collector()
        timing_metrics = collector.get_timing_metrics()
        
        assert "decorated_function" in timing_metrics
        assert timing_metrics["decorated_function"]["count"] == 1
    
    def test_timed_decorator_default_name(self):
        """Test @timed decorator with default name."""
        @timed()
        def test_function():
            return "result"
        
        test_function()
        
        collector = get_metrics_collector()
        timing_metrics = collector.get_timing_metrics()
        
        # Should use module.function_name format
        # The actual name will be the wrapper function name, so let's check for the test_function name
        function_names = list(timing_metrics.keys())
        assert len(function_names) > 0
        assert any("test_function" in name for name in function_names)
    
    def test_counted_decorator(self):
        """Test @counted decorator."""
        @counted("decorated_counter")
        def test_function():
            return "result"
        
        test_function()
        test_function()
        
        collector = get_metrics_collector()
        counter_metrics = collector.get_counter_metrics()
        
        assert "decorated_counter" in counter_metrics
        assert counter_metrics["decorated_counter"]["total_count"] == 2
    
    def test_error_tracked_decorator(self):
        """Test @error_tracked decorator."""
        @error_tracked("test_operation")
        def failing_function():
            raise ValueError("Test error")
        
        @error_tracked("test_operation")
        def success_function():
            return "success"
        
        # Test successful call
        result = success_function()
        assert result == "success"
        
        # Test error call
        with pytest.raises(ValueError):
            failing_function()
        
        collector = get_metrics_collector()
        error_metrics = collector.get_error_metrics()
        
        assert "test_operation_ValueError" in error_metrics
        assert error_metrics["test_operation_ValueError"]["total_count"] == 1
    
    def test_combined_decorators(self):
        """Test combining multiple decorators."""
        @timed("combined_timing")
        @counted("combined_counter")
        @error_tracked("combined_operation")
        def test_function(should_fail=False):
            if should_fail:
                raise RuntimeError("Test error")
            time.sleep(0.01)
            return "success"
        
        # Successful calls
        test_function()
        test_function()
        
        # Failed call
        with pytest.raises(RuntimeError):
            test_function(should_fail=True)
        
        collector = get_metrics_collector()
        
        # Check timing metrics
        timing_metrics = collector.get_timing_metrics()
        assert "combined_timing" in timing_metrics
        assert timing_metrics["combined_timing"]["count"] == 3  # All calls timed
        
        # Check counter metrics
        counter_metrics = collector.get_counter_metrics()
        assert "combined_counter" in counter_metrics
        assert counter_metrics["combined_counter"]["total_count"] == 3  # All calls counted
        
        # Check error metrics
        error_metrics = collector.get_error_metrics()
        assert "combined_operation_RuntimeError" in error_metrics
        assert error_metrics["combined_operation_RuntimeError"]["total_count"] == 1


class TestMetricsWithLabels:
    """Test metrics with labels."""
    
    def setup_method(self):
        """Set up test fixtures."""
        reset_metrics_collector()
    
    def test_timing_with_labels(self):
        """Test timing metrics with labels."""
        labels = {"service": "test", "version": "1.0"}
        
        record_timing("labeled_timing", 1.0, labels)
        
        collector = get_metrics_collector()
        gauge_metrics = collector.get_gauge_metrics()
        
        # Should create gauge metric for timing history
        assert "labeled_timing_timing" in gauge_metrics
        point = gauge_metrics["labeled_timing_timing"][-1]
        assert point["labels"] == labels
    
    def test_counter_with_labels(self):
        """Test counter metrics with labels."""
        labels = {"endpoint": "/api/test", "method": "POST"}
        
        increment_counter("labeled_counter", 1, labels)
        
        collector = get_metrics_collector()
        gauge_metrics = collector.get_gauge_metrics()
        
        # Should create gauge metric for counter history
        assert "labeled_counter_counter" in gauge_metrics
        point = gauge_metrics["labeled_counter_counter"][-1]
        assert point["labels"] == labels
    
    def test_gauge_with_labels(self):
        """Test gauge metrics with labels."""
        labels = {"instance": "server1", "region": "us-west"}
        
        record_gauge("labeled_gauge", 42.0, labels)
        
        collector = get_metrics_collector()
        gauge_metrics = collector.get_gauge_metrics()
        
        assert "labeled_gauge" in gauge_metrics
        point = gauge_metrics["labeled_gauge"][-1]
        assert point["labels"] == labels
        assert point["value"] == 42.0