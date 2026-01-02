"""Performance metrics collection and monitoring for PDF RAG System."""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import psutil
import os
from contextlib import contextmanager

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Individual metric data point."""
    
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "labels": self.labels,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class TimingMetric:
    """Timing metric with statistics."""
    
    name: str
    total_time: float = 0.0
    count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_timing(self, duration: float) -> None:
        """Add a timing measurement."""
        self.total_time += duration
        self.count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.recent_times.append(duration)
    
    @property
    def average_time(self) -> float:
        """Get average timing."""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def recent_average(self) -> float:
        """Get recent average timing."""
        if not self.recent_times:
            return 0.0
        return sum(self.recent_times) / len(self.recent_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "total_time": self.total_time,
            "count": self.count,
            "average_time": self.average_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0.0,
            "max_time": self.max_time,
            "recent_average": self.recent_average,
            "recent_count": len(self.recent_times)
        }


@dataclass
class CounterMetric:
    """Counter metric with rate calculation."""
    
    name: str
    count: int = 0
    recent_counts: deque = field(default_factory=lambda: deque(maxlen=60))  # Last 60 seconds
    recent_timestamps: deque = field(default_factory=lambda: deque(maxlen=60))
    
    def increment(self, amount: int = 1) -> None:
        """Increment counter."""
        self.count += amount
        current_time = time.time()
        self.recent_counts.append(amount)
        self.recent_timestamps.append(current_time)
        
        # Clean old entries (older than 60 seconds)
        cutoff_time = current_time - 60
        while self.recent_timestamps and self.recent_timestamps[0] < cutoff_time:
            self.recent_timestamps.popleft()
            self.recent_counts.popleft()
    
    @property
    def rate_per_minute(self) -> float:
        """Get rate per minute."""
        if not self.recent_counts:
            return 0.0
        return sum(self.recent_counts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "total_count": self.count,
            "rate_per_minute": self.rate_per_minute,
            "recent_count": sum(self.recent_counts) if self.recent_counts else 0
        }


class MetricsCollector:
    """Central metrics collection system."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metric points to keep in history
        """
        self.max_history = max_history
        self.start_time = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metric storage
        self._timing_metrics: Dict[str, TimingMetric] = {}
        self._counter_metrics: Dict[str, CounterMetric] = {}
        self._gauge_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._error_counts: Dict[str, CounterMetric] = {}
        
        # System metrics
        self._system_metrics_enabled = True
        self._last_system_update = 0
        self._system_update_interval = 5.0  # Update every 5 seconds
        
        logger.info("Metrics collector initialized", max_history=max_history)
    
    @contextmanager
    def time_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
            labels: Optional labels for the metric
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(operation_name, duration, labels)
    
    def record_timing(
        self,
        name: str,
        duration: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a timing metric.
        
        Args:
            name: Metric name
            duration: Duration in seconds
            labels: Optional labels for the metric
        """
        with self._lock:
            if name not in self._timing_metrics:
                self._timing_metrics[name] = TimingMetric(name)
            
            self._timing_metrics[name].add_timing(duration)
            
            # Also record as gauge for history
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=duration,
                labels=labels or {}
            )
            self._gauge_metrics[f"{name}_timing"].append(metric_point)
    
    def increment_counter(
        self,
        name: str,
        amount: int = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            amount: Amount to increment
            labels: Optional labels for the metric
        """
        with self._lock:
            if name not in self._counter_metrics:
                self._counter_metrics[name] = CounterMetric(name)
            
            self._counter_metrics[name].increment(amount)
            
            # Also record as gauge for history
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=amount,
                labels=labels or {}
            )
            self._gauge_metrics[f"{name}_counter"].append(metric_point)
    
    def record_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a gauge metric.
        
        Args:
            name: Gauge name
            value: Current value
            labels: Optional labels for the metric
        """
        with self._lock:
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            )
            self._gauge_metrics[name].append(metric_point)
    
    def record_error(
        self,
        error_type: str,
        operation: str,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record an error occurrence.
        
        Args:
            error_type: Type of error
            operation: Operation where error occurred
            labels: Optional labels for the metric
        """
        error_name = f"{operation}_{error_type}"
        
        with self._lock:
            if error_name not in self._error_counts:
                self._error_counts[error_name] = CounterMetric(error_name)
            
            self._error_counts[error_name].increment()
            
            # Also record as gauge for history
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=1,
                labels={**(labels or {}), "error_type": error_type, "operation": operation}
            )
            self._gauge_metrics["errors"].append(metric_point)
    
    def update_system_metrics(self) -> None:
        """Update system resource metrics."""
        if not self._system_metrics_enabled:
            return
        
        current_time = time.time()
        if current_time - self._last_system_update < self._system_update_interval:
            return
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_gauge("system_cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_gauge("system_memory_percent", memory.percent)
            self.record_gauge("system_memory_used_mb", memory.used / 1024 / 1024)
            self.record_gauge("system_memory_available_mb", memory.available / 1024 / 1024)
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            self.record_gauge("process_memory_rss_mb", process_memory.rss / 1024 / 1024)
            self.record_gauge("process_memory_vms_mb", process_memory.vms / 1024 / 1024)
            self.record_gauge("process_cpu_percent", process.cpu_percent())
            
            # Disk metrics for current directory
            disk_usage = psutil.disk_usage('.')
            self.record_gauge("disk_usage_percent", (disk_usage.used / disk_usage.total) * 100)
            self.record_gauge("disk_free_gb", disk_usage.free / 1024 / 1024 / 1024)
            
            self._last_system_update = current_time
            
        except Exception as e:
            logger.warning("Failed to update system metrics", error=str(e))
    
    def get_timing_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all timing metrics."""
        with self._lock:
            return {name: metric.to_dict() for name, metric in self._timing_metrics.items()}
    
    def get_counter_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all counter metrics."""
        with self._lock:
            return {name: metric.to_dict() for name, metric in self._counter_metrics.items()}
    
    def get_error_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all error metrics."""
        with self._lock:
            return {name: metric.to_dict() for name, metric in self._error_counts.items()}
    
    def get_gauge_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all gauge metrics."""
        with self._lock:
            return {
                name: [point.to_dict() for point in points]
                for name, points in self._gauge_metrics.items()
            }
    
    def get_recent_gauge_values(self, name: str, seconds: int = 60) -> List[Dict[str, Any]]:
        """
        Get recent gauge values for a specific metric.
        
        Args:
            name: Gauge name
            seconds: Number of seconds to look back
            
        Returns:
            List of recent metric points
        """
        with self._lock:
            if name not in self._gauge_metrics:
                return []
            
            cutoff_time = time.time() - seconds
            return [
                point.to_dict()
                for point in self._gauge_metrics[name]
                if point.timestamp >= cutoff_time
            ]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        self.update_system_metrics()
        
        with self._lock:
            # Get latest system metrics
            latest_metrics = {}
            for metric_name in ["system_cpu_percent", "system_memory_percent", "process_memory_rss_mb", "process_cpu_percent"]:
                if metric_name in self._gauge_metrics and self._gauge_metrics[metric_name]:
                    latest_metrics[metric_name] = self._gauge_metrics[metric_name][-1].value
            
            # Calculate health status
            cpu_healthy = latest_metrics.get("system_cpu_percent", 0) < 80
            memory_healthy = latest_metrics.get("system_memory_percent", 0) < 85
            
            health_status = "healthy"
            if not cpu_healthy or not memory_healthy:
                health_status = "degraded"
            if latest_metrics.get("system_cpu_percent", 0) > 95 or latest_metrics.get("system_memory_percent", 0) > 95:
                health_status = "critical"
            
            return {
                "status": health_status,
                "uptime_seconds": time.time() - self.start_time,
                "metrics": latest_metrics,
                "thresholds": {
                    "cpu_warning": 80,
                    "cpu_critical": 95,
                    "memory_warning": 85,
                    "memory_critical": 95
                }
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all metrics."""
        with self._lock:
            # Update system metrics first
            self.update_system_metrics()
            
            # Calculate error rates
            total_requests = sum(metric.count for metric in self._counter_metrics.values() if "request" in metric.name)
            total_errors = sum(metric.count for metric in self._error_counts.values())
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
            
            # Get average response times
            avg_response_times = {}
            for name, metric in self._timing_metrics.items():
                if "response" in name or "processing" in name:
                    avg_response_times[name] = metric.average_time
            
            # Get cache hit rates (if available)
            cache_metrics = {}
            for name, metric in self._counter_metrics.items():
                if "cache" in name:
                    cache_metrics[name] = metric.to_dict()
            
            return {
                "uptime_seconds": time.time() - self.start_time,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate_percent": error_rate,
                "average_response_times": avg_response_times,
                "cache_metrics": cache_metrics,
                "system_health": self.get_system_health()
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        return {
            "timing_metrics": self.get_timing_metrics(),
            "counter_metrics": self.get_counter_metrics(),
            "error_metrics": self.get_error_metrics(),
            "gauge_metrics": self.get_gauge_metrics(),
            "system_health": self.get_system_health(),
            "performance_summary": self.get_performance_summary(),
            "collection_info": {
                "start_time": self.start_time,
                "uptime_seconds": time.time() - self.start_time,
                "max_history": self.max_history,
                "metrics_count": {
                    "timing": len(self._timing_metrics),
                    "counter": len(self._counter_metrics),
                    "gauge": len(self._gauge_metrics),
                    "error": len(self._error_counts)
                }
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._timing_metrics.clear()
            self._counter_metrics.clear()
            self._gauge_metrics.clear()
            self._error_counts.clear()
            self.start_time = time.time()
            logger.info("All metrics reset")
    
    def enable_system_metrics(self, enabled: bool = True) -> None:
        """Enable or disable system metrics collection."""
        self._system_metrics_enabled = enabled
        logger.info("System metrics collection", enabled=enabled)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector (useful for testing)."""
    global _metrics_collector
    _metrics_collector = None


# Convenience functions for common metrics operations

def time_operation(operation_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator/context manager for timing operations."""
    return get_metrics_collector().time_operation(operation_name, labels)


def record_timing(name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Record a timing metric."""
    get_metrics_collector().record_timing(name, duration, labels)


def increment_counter(name: str, amount: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
    """Increment a counter metric."""
    get_metrics_collector().increment_counter(name, amount, labels)


def record_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Record a gauge metric."""
    get_metrics_collector().record_gauge(name, value, labels)


def record_error(error_type: str, operation: str, labels: Optional[Dict[str, str]] = None) -> None:
    """Record an error occurrence."""
    get_metrics_collector().record_error(error_type, operation, labels)


# Decorators for automatic metrics collection

def timed(operation_name: Optional[str] = None, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to automatically time function execution.
    
    Args:
        operation_name: Name for the timing metric (defaults to function name)
        labels: Optional labels for the metric
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            with time_operation(name, labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def counted(counter_name: Optional[str] = None, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to automatically count function calls.
    
    Args:
        counter_name: Name for the counter metric (defaults to function name)
        labels: Optional labels for the metric
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            name = counter_name or f"{func.__module__}.{func.__name__}_calls"
            increment_counter(name, labels=labels)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def error_tracked(operation_name: Optional[str] = None):
    """
    Decorator to automatically track errors in function execution.
    
    Args:
        operation_name: Name for the operation (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            try:
                return func(*args, **kwargs)
            except Exception as e:
                record_error(type(e).__name__, operation)
                raise
        return wrapper
    return decorator