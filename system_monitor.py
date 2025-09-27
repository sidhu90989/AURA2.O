"""
AURA System Monitor - Neural-based system health monitoring and predictive maintenance
This module provides comprehensive system monitoring with anomaly detection capabilities.
Python 3.8.10 compatible.
"""

import psutil
import time
import logging
import platform
import socket
import threading
import numpy as np
from collections import deque
from typing import Dict, Union, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from datetime import datetime

# Conditionally import scikit-learn to handle missing dependencies gracefully
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aura_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AURA.SystemMonitor')


@dataclass
class SystemHealth:
    """Data structure to store comprehensive system health metrics"""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    process_count: int = 0
    temperatures: Dict[str, float] = field(default_factory=dict)
    power_status: Dict[str, Union[float, bool, None]] = field(default_factory=dict)
    anomalies: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Human-readable representation of system health"""
        return (
            f"System Health @ {datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  CPU: {self.cpu_usage:.1f}%, Memory: {self.memory_usage:.1f}%, "
            f"Disk: {self.disk_usage:.1f}%, Processes: {self.process_count}"
        )


@dataclass
class SystemRecommendation:
    """Data structure for system recommendations"""
    type: str
    priority: str
    action: str
    message: str


class ThresholdManager:
    """Manager for adaptive system thresholds"""
    
    def __init__(self) -> None:
        """Initialize default thresholds"""
        self.thresholds = {
            'cpu_warning': 75.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'thermal_warning': 70.0,  # Celsius
            'disk_warning': 85.0,
            'disk_critical': 95.0,
        }
        
    def adjust_thresholds(self, history: Deque[SystemHealth]) -> None:
        """Dynamically adjust thresholds based on historical data"""
        if len(history) < 100:
            return
            
        # Extract recent CPU metrics
        recent_cpu = [h.cpu_usage for h in list(history)[-100:]]
        if recent_cpu:
            avg = np.mean(recent_cpu)
            std = np.std(recent_cpu)
            
            # Adjust thresholds with safety caps
            self.thresholds['cpu_warning'] = min(90, avg + 2*std)
            self.thresholds['cpu_critical'] = min(95, avg + 3*std)
            
    def get_threshold(self, metric: str) -> float:
        """Get the current threshold for a metric"""
        return self.thresholds.get(metric, 0.0)


class AnomalyDetector:
    """Machine learning-based system anomaly detection"""
    
    def __init__(self) -> None:
        """Initialize anomaly detection capabilities"""
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize the machine learning model if dependencies are available"""
        if SKLEARN_AVAILABLE:
            self.model = IsolationForest(
                n_estimators=100, 
                contamination=0.01,
                random_state=42  # For reproducibility
            )
            logger.info("Anomaly detection model initialized")
        else:
            logger.warning("scikit-learn not available, anomaly detection disabled")
            
    def is_available(self) -> bool:
        """Check if anomaly detection is available"""
        return self.model is not None
        
    def detect_anomalies(self, history: Deque[SystemHealth]) -> Dict[str, float]:
        """Detect system anomalies using machine learning"""
        if not self.is_available() or len(history) < 100:
            return {}
        
        try:
            # Extract features for anomaly detection
            features = np.array([
                [
                    h.cpu_usage,
                    h.memory_usage,
                    h.disk_usage,
                    h.process_count
                ] 
                for h in history
            ])
            
            # Train the model with all historical data
            self.model.fit(features)
            
            # Predict anomalies for recent data points
            recent_features = features[-10:] if len(features) >= 10 else features
            scores = self.model.decision_function(recent_features)
            
            return {
                'anomaly_score': float(np.mean(scores)),
                'anomaly_min': float(np.min(scores)),
                'anomaly_max': float(np.max(scores))
            }
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {'anomaly_score': 0.0, 'error': str(e)}


class SystemMetricsCollector:
    """Component for collecting system metrics"""
    
    def collect_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=1, percpu=False)
        except Exception as e:
            logger.error(f"Failed to collect CPU usage: {e}")
            return 0.0
    
    def collect_memory_usage(self) -> float:
        """Get memory usage percentage"""
        try:
            return psutil.virtual_memory().percent
        except Exception as e:
            logger.error(f"Failed to collect memory usage: {e}")
            return 0.0
    
    def collect_disk_usage(self) -> float:
        """Get critical partition usage percentage"""
        try:
            partitions = [p for p in psutil.disk_partitions() if p.fstype]
            if not partitions:
                return 0.0
                
            # Find partition with highest usage
            max_usage = 0.0
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint).percent
                    max_usage = max(max_usage, usage)
                except (PermissionError, FileNotFoundError) as e:
                    logger.debug(f"Skipping partition {partition.mountpoint}: {e}")
                    
            return max_usage
        except Exception as e:
            logger.error(f"Failed to collect disk usage: {e}")
            return 0.0
    
    def collect_network_stats(self) -> Dict[str, float]:
        """Get network I/O statistics"""
        try:
            net = psutil.net_io_counters()
            return {
                'bytes_sent': float(net.bytes_sent),
                'bytes_recv': float(net.bytes_recv),
                'packets_sent': float(net.packets_sent),
                'packets_recv': float(net.packets_recv),
                'errin': float(net.errin) if hasattr(net, 'errin') else 0.0,
                'errout': float(net.errout) if hasattr(net, 'errout') else 0.0
            }
        except Exception as e:
            logger.error(f"Failed to collect network stats: {e}")
            return {}
    
    def collect_process_count(self) -> int:
        """Get running process count"""
        try:
            return len(psutil.pids())
        except Exception as e:
            logger.error(f"Failed to collect process count: {e}")
            return 0
    
    def collect_temperatures(self) -> Dict[str, float]:
        """Get hardware temperatures"""
        temps = {}
        try:
            if hasattr(psutil, "sensors_temperatures"):
                for name, entries in psutil.sensors_temperatures().items():
                    if entries:  # Check if entries exist before max operation
                        temps[name] = max(entry.current for entry in entries)
        except Exception as e:
            logger.debug(f"Temperature read error: {e}")
        return temps
    
    def collect_power_status(self) -> Dict[str, Union[float, bool, None]]:
        """Get power/battery status"""
        status = {}
        try:
            battery = psutil.sensors_battery()
            if battery:
                status.update({
                    'percent': float(battery.percent),
                    'power_plugged': bool(battery.power_plugged),
                    'time_left': float(battery.secsleft) if battery.secsleft > 0 else None
                })
        except Exception as e:
            logger.debug(f"Power status error: {e}")
        return status


class TrendAnalyzer:
    """Component for analyzing system metric trends"""
    
    def analyze_cpu_trend(self, history: Deque[SystemHealth]) -> Tuple[float, str]:
        """Analyze CPU usage trend and return slope and description"""
        if len(history) < 10:
            return 0.0, "Insufficient data"
            
        # Simple linear regression for trend analysis
        x = np.arange(len(history))
        y_cpu = [h.cpu_usage for h in history]
        
        try:
            slope = np.polyfit(x, y_cpu, 1)[0]
            
            # Interpret the trend
            if slope > 0.5:
                trend = f"Increasing significantly ({slope:.2f}%/sample)"
            elif slope > 0.1:
                trend = f"Increasing gradually ({slope:.2f}%/sample)"
            elif slope < -0.5:
                trend = f"Decreasing significantly ({slope:.2f}%/sample)"
            elif slope < -0.1:
                trend = f"Decreasing gradually ({slope:.2f}%/sample)"
            else:
                trend = f"Stable ({slope:.2f}%/sample)"
                
            return slope, trend
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return 0.0, f"Analysis error: {e}"
    
    def analyze_memory_trend(self, history: Deque[SystemHealth]) -> Tuple[float, str]:
        """Analyze memory usage trend"""
        if len(history) < 10:
            return 0.0, "Insufficient data"
            
        x = np.arange(len(history))
        y_mem = [h.memory_usage for h in history]
        
        try:
            slope = np.polyfit(x, y_mem, 1)[0]
            if slope > 0.5:
                trend = f"Memory usage increasing rapidly ({slope:.2f}%/sample)"
            elif slope > 0.1:
                trend = f"Memory usage increasing gradually ({slope:.2f}%/sample)"
            else:
                trend = f"Memory usage stable"
                
            return slope, trend
        except Exception as e:
            logger.error(f"Memory trend analysis failed: {e}")
            return 0.0, f"Analysis error: {e}"
    
    def calculate_trend(self, data: List[float]) -> float:
        """Calculate linear trend coefficient for generic data"""
        if len(data) <= 1:
            return 0.0
            
        try:
            x = np.arange(len(data))
            return np.polyfit(x, data, 1)[0]
        except Exception as e:
            logger.error(f"Trend calculation error: {e}")
            return 0.0


class RecommendationEngine:
    """Engine for generating system health recommendations"""
    
    def __init__(self, threshold_manager: ThresholdManager) -> None:
        """Initialize with threshold manager for accessing thresholds"""
        self.threshold_manager = threshold_manager
    
    def generate_recommendations(self, health: SystemHealth) -> List[SystemRecommendation]:
        """Generate recommendations based on system health"""
        recommendations = []
        
        # CPU recommendations
        if health.cpu_usage > self.threshold_manager.get_threshold('cpu_critical'):
            recommendations.append(SystemRecommendation(
                type='resource',
                priority='critical',
                action='reduce_cpu_load',
                message=f"Critical CPU usage ({health.cpu_usage:.1f}%) detected"
            ))
        elif health.cpu_usage > self.threshold_manager.get_threshold('cpu_warning'):
            recommendations.append(SystemRecommendation(
                type='resource',
                priority='high',
                action='scale_background_processes',
                message=f"High CPU usage ({health.cpu_usage:.1f}%) detected"
            ))
        
        # Memory recommendations
        if health.memory_usage > self.threshold_manager.get_threshold('memory_critical'):
            recommendations.append(SystemRecommendation(
                type='resource',
                priority='critical',
                action='free_memory',
                message=f"Critical memory usage ({health.memory_usage:.1f}%) detected"
            ))
        elif health.memory_usage > self.threshold_manager.get_threshold('memory_warning'):
            recommendations.append(SystemRecommendation(
                type='resource',
                priority='high',
                action='memory_optimization',
                message=f"High memory usage ({health.memory_usage:.1f}%) detected"
            ))
        
        # Disk recommendations
        if health.disk_usage > self.threshold_manager.get_threshold('disk_critical'):
            recommendations.append(SystemRecommendation(
                type='storage',
                priority='critical',
                action='free_disk_space',
                message=f"Critical disk usage ({health.disk_usage:.1f}%) detected"
            ))
        elif health.disk_usage > self.threshold_manager.get_threshold('disk_warning'):
            recommendations.append(SystemRecommendation(
                type='storage',
                priority='high',
                action='disk_cleanup',
                message=f"High disk usage ({health.disk_usage:.1f}%) detected"
            ))
        
        # Temperature recommendations
        high_temps = [
            (sensor, temp) for sensor, temp in health.temperatures.items()
            if temp > self.threshold_manager.get_threshold('thermal_warning')
        ]
        
        if high_temps:
            sensor, temp = max(high_temps, key=lambda x: x[1])
            recommendations.append(SystemRecommendation(
                type='hardware',
                priority='critical',
                action='thermal_management',
                message=f"Critical temperature detected: {sensor} at {temp:.1f}°C"
            ))
        
        # Anomaly recommendations
        if health.anomalies.get('anomaly_score', 0) < -0.5:
            recommendations.append(SystemRecommendation(
                type='system',
                priority='medium',
                action='investigate_anomaly',
                message="System behavior anomaly detected"
            ))
        
        # Power recommendations
        if health.power_status.get('percent', 100) < 15 and not health.power_status.get('power_plugged', True):
            recommendations.append(SystemRecommendation(
                type='power',
                priority='high',
                action='connect_power',
                message=f"Low battery: {health.power_status.get('percent', 0):.1f}% remaining"
            ))
        
        return recommendations


class EmergencyResponseHandler:
    """Handler for critical system conditions requiring immediate action"""
    
    def __init__(self, threshold_manager: ThresholdManager) -> None:
        """Initialize with threshold manager for accessing thresholds"""
        self.threshold_manager = threshold_manager
    
    def handle_emergencies(self, health: SystemHealth) -> List[str]:
        """Execute critical system preservation measures and return actions taken"""
        actions = []
        
        # Handle critical CPU usage
        if health.cpu_usage > self.threshold_manager.get_threshold('cpu_critical'):
            actions.extend(self._handle_critical_cpu())
        
        # Handle critical memory usage
        if health.memory_usage > self.threshold_manager.get_threshold('memory_critical'):
            actions.extend(self._handle_critical_memory())
        
        # Handle critical temperatures
        critical_temps = [
            sensor for sensor, temp in health.temperatures.items()
            if temp > self.threshold_manager.get_threshold('thermal_warning')
        ]
        if critical_temps:
            actions.extend(self._handle_critical_temperature())
        
        return actions
    
    def _handle_critical_cpu(self) -> List[str]:
        """Handle critical CPU usage scenario"""
        actions = []
        try:
            # Find and kill resource-intensive processes
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    if proc.info['cpu_percent'] > 50:
                        proc_name = proc.info.get('name', f"PID {proc.info.get('pid', 'unknown')}")
                        proc.kill()
                        actions.append(f"Killed high-CPU process: {proc_name}")
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    continue
        except Exception as e:
            logger.error(f"Critical CPU handling error: {e}")
        
        return actions
    
    def _handle_critical_memory(self) -> List[str]:
        """Handle critical memory usage scenario"""
        actions = []
        try:
            # Find and kill memory-intensive processes
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                try:
                    if hasattr(proc, 'memory_percent') and proc.memory_percent() > 20:
                        proc_name = proc.name()
                        proc.kill()
                        actions.append(f"Killed high-memory process: {proc_name}")
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    continue
        except Exception as e:
            logger.error(f"Critical memory handling error: {e}")
        
        return actions
    
    def _handle_critical_temperature(self) -> List[str]:
        """Handle critical temperature scenario"""
        # This would ideally interface with system-specific cooling controls
        # For now, just log the action
        actions = ["Activated emergency cooling protocol"]
        logger.critical("Critical temperatures detected - emergency cooling needed")
        return actions


class NeuroSystemMonitor:
    """Autonomous system monitoring with predictive maintenance capabilities"""
    
    def __init__(self) -> None:
        """Initialize the neural system monitor"""
        # Component initialization
        self.threshold_manager = ThresholdManager()
        self.anomaly_detector = AnomalyDetector()
        self.metrics_collector = SystemMetricsCollector()
        self.trend_analyzer = TrendAnalyzer()
        self.recommendation_engine = RecommendationEngine(self.threshold_manager)
        self.emergency_handler = EmergencyResponseHandler(self.threshold_manager)
        
        # Data storage
        self.history: Deque[SystemHealth] = deque(maxlen=1000)
        
        # Thread management
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        # Start monitoring
        self._start_monitoring_daemon()
        logger.info("Neural System Monitor initialized")
        
    def full_health_check(self) -> SystemHealth:
        """Perform comprehensive system diagnostics"""
        # Collect all system metrics
        health = SystemHealth(
            timestamp=time.time(),
            cpu_usage=self.metrics_collector.collect_cpu_usage(),
            memory_usage=self.metrics_collector.collect_memory_usage(),
            disk_usage=self.metrics_collector.collect_disk_usage(),
            network_io=self.metrics_collector.collect_network_stats(),
            process_count=self.metrics_collector.collect_process_count(),
            temperatures=self.metrics_collector.collect_temperatures(),
            power_status=self.metrics_collector.collect_power_status()
        )
        
        # Add anomaly detection if we have enough history
        if self.history:
            health.anomalies = self.anomaly_detector.detect_anomalies(self.history)
        
        return health
    
    def _start_monitoring_daemon(self) -> None:
        """Start background monitoring thread with proper management"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="AURA-SystemMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Monitoring daemon started")
    
    def stop_monitoring(self) -> None:
        """Gracefully stop the monitoring thread"""
        if self.monitoring_active:
            logger.info("Stopping monitoring daemon...")
            self.monitoring_active = False
            
            # Give the thread time to terminate gracefully
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
                if self.monitor_thread.is_alive():
                    logger.warning("Monitoring thread did not terminate gracefully")
                else:
                    logger.info("Monitoring daemon stopped")
    
    def _monitoring_loop(self) -> None:
        """Continuous system monitoring loop"""
        last_thresholds_update = time.time()
        sampling_interval = 5  # seconds
        
        while self.monitoring_active:
            try:
                # Collect system health data
                health = self.full_health_check()
                self.history.append(health)
                
                # Log basic system stats periodically
                if len(self.history) % 12 == 0:  # Log every ~60 seconds
                    logger.info(str(health))
                
                # Analyze trends in system metrics
                self._analyze_trends()
                
                # Auto-adjust thresholds periodically (every 10 minutes)
                current_time = time.time()
                if current_time - last_thresholds_update > 600:
                    self.threshold_manager.adjust_thresholds(self.history)
                    last_thresholds_update = current_time
                
                # Check for critical conditions requiring immediate action
                if (health.cpu_usage > self.threshold_manager.get_threshold('cpu_critical') or
                        health.memory_usage > self.threshold_manager.get_threshold('memory_critical')):
                    actions = self.emergency_actions()
                    if actions:
                        logger.warning(f"Emergency actions taken: {', '.join(actions)}")
                
                # Sleep until next sampling interval
                time.sleep(sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                # Use longer sleep on error to avoid fast error loops
                time.sleep(30)
    
    def _analyze_trends(self) -> None:
        """Analyze system metric trends"""
        if len(self.history) < 100:
            return
        
        try:
            # Analyze CPU trend
            cpu_slope, cpu_trend = self.trend_analyzer.analyze_cpu_trend(self.history)
            
            # Log significant trends
            if abs(cpu_slope) > 0.3:
                logger.info(f"CPU trend: {cpu_trend}")
            
            # Analyze memory trend
            mem_slope, mem_trend = self.trend_analyzer.analyze_memory_trend(self.history)
            
            if mem_slope > 0.3:
                logger.info(f"Memory trend: {mem_trend}")
        
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
    
    def get_recommendations(self) -> List[SystemRecommendation]:
        """Generate autonomous system recommendations"""
        if not self.history:
            return []
            
        # Use the most recent health data for recommendations
        latest_health = self.history[-1]
        return self.recommendation_engine.generate_recommendations(latest_health)
    
    def emergency_actions(self) -> List[str]:
        """Execute critical system preservation measures"""
        if not self.history:
            return []
            
        # Use the most recent health data for emergency handling
        latest_health = self.history[-1]
        return self.emergency_handler.handle_emergencies(latest_health)
    
    def generate_report(self, period: int = 3600) -> Dict[str, Any]:
        """Generate comprehensive system health report for the specified period"""
        current_time = time.time()
        recent = [h for h in self.history if h.timestamp > current_time - period]
        
        if not recent:
            return {"error": "No data available for the specified period"}
        
        # Extract data for analysis
        cpu_values = [h.cpu_usage for h in recent]
        memory_values = [h.memory_usage for h in recent]
        disk_values = [h.disk_usage for h in recent]
        
        try:
            # Calculate trends
            cpu_trend = self.trend_analyzer.calculate_trend(cpu_values)
            memory_trend = self.trend_analyzer.calculate_trend(memory_values)
            disk_trend = self.trend_analyzer.calculate_trend(disk_values)
            
            # Count anomalies
            anomaly_count = sum(1 for h in recent if h.anomalies.get('anomaly_score', 0) < -0.5)
            
            # Build comprehensive report
            report = {
                'period_start': datetime.fromtimestamp(recent[0].timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                'period_end': datetime.fromtimestamp(recent[-1].timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                'samples': len(recent),
                'cpu': {
                    'avg': np.mean(cpu_values) if cpu_values else 0,
                    'max': max(cpu_values) if cpu_values else 0,
                    'min': min(cpu_values) if cpu_values else 0,
                    'trend': cpu_trend
                },
                'memory': {
                    'avg': np.mean(memory_values) if memory_values else 0,
                    'max': max(memory_values) if memory_values else 0,
                    'min': min(memory_values) if memory_values else 0,
                    'trend': memory_trend
                },
                'disk': {
                    'critical_usage': max(disk_values) if disk_values else 0,
                    'trend': disk_trend
                },
                'anomalies': anomaly_count,
                'system_info': {
                    'platform': platform.platform(),
                    'processor': platform.processor(),
                    'hostname': socket.gethostname()
                }
            }
            
            # Add power information if available
            for health in recent:
                if health.power_status:
                    report['power'] = {
                        'on_battery': not health.power_status.get('power_plugged', True),
                        'last_battery': health.power_status.get('percent', None)
                    }
                    break
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {
                'error': str(e),
                'period': period,
                'samples': len(recent)
            }


# Example usage
if __name__ == "__main__":
    try:
        # Create monitor instance
        monitor = NeuroSystemMonitor()
        
        # Run for a specified duration to collect data
        print("Collecting system data...")
        time.sleep(60)  # Collect data for 60 seconds
        
        # Generate and display recommendations
        print("\nSystem Recommendations:")
        for rec in monitor.get_recommendations():
            print(f"[{rec.priority.upper()}] {rec.message} - Action: {rec.action}")
        
        # Generate system report
        print("\nSystem Health Report:")
        report = monitor.generate_report(period=60)
        for category, data in report.items():
            if isinstance(data, dict):
                print(f"\n{category.upper()}:")
                for key, value in data.items():
                    print(f"  {key}: {value}")
            else:
                print(f"{category}: {data}")
        
        # Clean shutdown
        monitor.stop_monitoring()
        print("\nMonitoring stopped.")
        
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user.")
        # Ensure clean shutdown
        monitor.stop_monitoring() if 'monitor' in locals() else None