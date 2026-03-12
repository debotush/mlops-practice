"""
MLOps Practice 2 - Production Monitoring Implementation
This module provides monitoring functions for production model deployment
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor:
    """
    Production monitoring system for ML model
    Tracks data drift, model performance, and system metrics
    """
    
    def __init__(self, baseline_metrics, training_data_stats):
        """
        Initialize monitor with baseline metrics and training statistics
        
        Args:
            baseline_metrics: dict with 'rmse', 'mae', 'r2_score' from production model
            training_data_stats: dict with feature statistics from training data
        """
        self.baseline_metrics = baseline_metrics
        self.training_stats = training_data_stats
        self.alerts = []
        
    def check_feature_drift(self, train_feature, prod_feature, feature_name, 
                           threshold_warning=0.05, threshold_critical=0.10):
        """
        Detect data drift using Kolmogorov-Smirnov test
        
        Args:
            train_feature: array-like, feature values from training data
            prod_feature: array-like, feature values from production data
            feature_name: str, name of the feature
            threshold_warning: float, KS statistic threshold for warning
            threshold_critical: float, KS statistic threshold for critical alert
            
        Returns:
            dict with drift status and statistics
        """
        # Perform KS test
        statistic, pvalue = ks_2samp(train_feature, prod_feature)
        
        status = "OK"
        recommendation = None
        
        if statistic > threshold_critical:
            status = "CRITICAL"
            recommendation = "RETRAIN_IMMEDIATELY"
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'DATA_DRIFT',
                'severity': 'CRITICAL',
                'feature': feature_name,
                'message': f'Critical drift detected in {feature_name}',
                'ks_statistic': statistic,
                'recommendation': recommendation
            })
        elif statistic > threshold_warning:
            status = "WARNING"
            recommendation = "INVESTIGATE"
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'DATA_DRIFT',
                'severity': 'WARNING',
                'feature': feature_name,
                'message': f'Warning: drift detected in {feature_name}',
                'ks_statistic': statistic,
                'recommendation': recommendation
            })
            
        return {
            'feature': feature_name,
            'ks_statistic': statistic,
            'p_value': pvalue,
            'status': status,
            'recommendation': recommendation
        }
    
    def check_model_performance(self, y_true, y_pred, 
                               threshold_warning=0.10, threshold_critical=0.20):
        """
        Monitor model performance degradation
        
        Args:
            y_true: array-like, actual values
            y_pred: array-like, predicted values
            threshold_warning: float, % increase in RMSE for warning
            threshold_critical: float, % increase in RMSE for critical
            
        Returns:
            dict with performance status and metrics
        """
        current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        current_mae = mean_absolute_error(y_true, y_pred)
        
        baseline_rmse = self.baseline_metrics['rmse']
        rmse_degradation = (current_rmse / baseline_rmse - 1)
        
        status = "OK"
        recommendation = None
        
        if rmse_degradation > threshold_critical:
            status = "CRITICAL"
            recommendation = "ROLLBACK_AND_RETRAIN"
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'PERFORMANCE_DEGRADATION',
                'severity': 'CRITICAL',
                'message': 'Critical performance degradation detected',
                'rmse_increase_pct': rmse_degradation * 100,
                'recommendation': recommendation
            })
        elif rmse_degradation > threshold_warning:
            status = "WARNING"
            recommendation = "MONITOR_CLOSELY"
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'PERFORMANCE_DEGRADATION',
                'severity': 'WARNING',
                'message': 'Performance degradation warning',
                'rmse_increase_pct': rmse_degradation * 100,
                'recommendation': recommendation
            })
        
        # Calculate error statistics
        errors = y_true - y_pred
        
        return {
            'current_rmse': current_rmse,
            'baseline_rmse': baseline_rmse,
            'rmse_degradation_pct': rmse_degradation * 100,
            'current_mae': current_mae,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'status': status,
            'recommendation': recommendation
        }
    
    def check_system_metrics(self, latency_p95, cpu_usage, memory_usage,
                            latency_threshold_warning=75, 
                            latency_threshold_critical=100,
                            cpu_threshold_warning=70,
                            cpu_threshold_critical=85):
        """
        Monitor system performance metrics
        
        Args:
            latency_p95: float, 95th percentile latency in milliseconds
            cpu_usage: float, CPU utilization percentage
            memory_usage: float, memory utilization percentage
            
        Returns:
            dict with system status
        """
        status = "OK"
        recommendations = []
        
        if latency_p95 > latency_threshold_critical:
            status = "CRITICAL"
            recommendations.append("SCALE_INFRASTRUCTURE")
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'SYSTEM_PERFORMANCE',
                'severity': 'CRITICAL',
                'message': f'High latency: {latency_p95}ms (threshold: {latency_threshold_critical}ms)',
                'recommendation': 'SCALE_INFRASTRUCTURE'
            })
        elif latency_p95 > latency_threshold_warning:
            status = "WARNING"
            recommendations.append("REVIEW_INFRASTRUCTURE")
            
        if cpu_usage > cpu_threshold_critical:
            status = "CRITICAL"
            recommendations.append("ADD_INSTANCES")
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'SYSTEM_PERFORMANCE',
                'severity': 'CRITICAL',
                'message': f'High CPU usage: {cpu_usage}% (threshold: {cpu_threshold_critical}%)',
                'recommendation': 'ADD_INSTANCES'
            })
        elif cpu_usage > cpu_threshold_warning:
            if status == "OK":
                status = "WARNING"
            recommendations.append("OPTIMIZE_MODEL")
            
        return {
            'latency_p95_ms': latency_p95,
            'cpu_usage_pct': cpu_usage,
            'memory_usage_pct': memory_usage,
            'status': status,
            'recommendations': recommendations
        }
    
    def generate_monitoring_report(self):
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if a['severity'] == 'CRITICAL']),
            'warning_alerts': len([a for a in self.alerts if a['severity'] == 'WARNING']),
            'alerts': self.alerts
        }
        return report
    
    def save_monitoring_report(self, filepath):
        """Save monitoring report to JSON file"""
        report = self.generate_monitoring_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Monitoring report saved to {filepath}")


# Example usage and demonstration
if __name__ == "__main__":
    print("="*60)
    print("PRODUCTION MONITORING SYSTEM - DEMONSTRATION")
    print("="*60)
    
    # Baseline metrics from Version 3
    baseline_metrics = {
        'rmse': 6.3,
        'mae': 4.1,
        'r2_score': 0.80
    }
    
    # Simulated training data statistics
    training_stats = {
        'trip_distance': {'mean': 3.5, 'std': 4.2},
        'passenger_count': {'mean': 1.4, 'std': 0.9},
        'pickup_hour': {'mean': 14.5, 'std': 6.8}
    }
    
    # Initialize monitor
    monitor = ModelMonitor(baseline_metrics, training_stats)
    
    # Simulate production data
    np.random.seed(42)
    
    # Scenario 1: No drift, good performance
    print("\n" + "="*60)
    print("SCENARIO 1: Normal Operation (No Drift)")
    print("="*60)
    
    train_distance = np.random.normal(3.5, 4.2, 10000)
    prod_distance = np.random.normal(3.5, 4.2, 1000)
    
    drift_result = monitor.check_feature_drift(
        train_distance, prod_distance, 'trip_distance'
    )
    print(f"\nFeature Drift Check:")
    print(f"  Feature: {drift_result['feature']}")
    print(f"  KS Statistic: {drift_result['ks_statistic']:.4f}")
    print(f"  Status: {drift_result['status']}")
    
    # Simulated predictions with good performance
    y_true = np.random.normal(15, 5, 1000)
    y_pred = y_true + np.random.normal(0, 6.3, 1000)
    
    perf_result = monitor.check_model_performance(y_true, y_pred)
    print(f"\nModel Performance Check:")
    print(f"  Current RMSE: {perf_result['current_rmse']:.2f} minutes")
    print(f"  Baseline RMSE: {perf_result['baseline_rmse']:.2f} minutes")
    print(f"  Degradation: {perf_result['rmse_degradation_pct']:.2f}%")
    print(f"  Status: {perf_result['status']}")
    
    # System metrics - normal
    sys_result = monitor.check_system_metrics(
        latency_p95=45.0,
        cpu_usage=55.0,
        memory_usage=60.0
    )
    print(f"\nSystem Metrics Check:")
    print(f"  P95 Latency: {sys_result['latency_p95_ms']:.1f}ms")
    print(f"  CPU Usage: {sys_result['cpu_usage_pct']:.1f}%")
    print(f"  Status: {sys_result['status']}")
    
    # Scenario 2: Data drift detected
    print("\n" + "="*60)
    print("SCENARIO 2: Data Drift Detected")
    print("="*60)
    
    # Simulate drift - distribution shift
    prod_distance_drift = np.random.normal(5.0, 5.5, 1000)  # Different mean and std
    
    drift_result = monitor.check_feature_drift(
        train_distance, prod_distance_drift, 'trip_distance'
    )
    print(f"\nFeature Drift Check:")
    print(f"  Feature: {drift_result['feature']}")
    print(f"  KS Statistic: {drift_result['ks_statistic']:.4f}")
    print(f"  Status: {drift_result['status']}")
    print(f"  Recommendation: {drift_result['recommendation']}")
    
    # Scenario 3: Performance degradation
    print("\n" + "="*60)
    print("SCENARIO 3: Performance Degradation")
    print("="*60)
    
    # Simulate worse predictions
    y_pred_bad = y_true + np.random.normal(0, 9.0, 1000)  # Higher error
    
    perf_result = monitor.check_model_performance(y_true, y_pred_bad)
    print(f"\nModel Performance Check:")
    print(f"  Current RMSE: {perf_result['current_rmse']:.2f} minutes")
    print(f"  Baseline RMSE: {perf_result['baseline_rmse']:.2f} minutes")
    print(f"  Degradation: {perf_result['rmse_degradation_pct']:+.2f}%")
    print(f"  Status: {perf_result['status']}")
    print(f"  Recommendation: {perf_result['recommendation']}")
    
    # Scenario 4: System issues
    print("\n" + "="*60)
    print("SCENARIO 4: System Performance Issues")
    print("="*60)
    
    sys_result = monitor.check_system_metrics(
        latency_p95=110.0,  # High latency
        cpu_usage=88.0,      # High CPU
        memory_usage=75.0
    )
    print(f"\nSystem Metrics Check:")
    print(f"  P95 Latency: {sys_result['latency_p95_ms']:.1f}ms")
    print(f"  CPU Usage: {sys_result['cpu_usage_pct']:.1f}%")
    print(f"  Status: {sys_result['status']}")
    print(f"  Recommendations: {', '.join(sys_result['recommendations'])}")
    
    # Generate final report
    print("\n" + "="*60)
    print("MONITORING SUMMARY")
    print("="*60)
    
    report = monitor.generate_monitoring_report()
    print(f"\nTotal Alerts: {report['total_alerts']}")
    print(f"Critical: {report['critical_alerts']}")
    print(f"Warning: {report['warning_alerts']}")
    
    if report['alerts']:
        print("\nAlert Details:")
        for i, alert in enumerate(report['alerts'], 1):
            print(f"\n{i}. [{alert['severity']}] {alert['type']}")
            print(f"   Message: {alert['message']}")
            if 'recommendation' in alert:
                print(f"   Action: {alert['recommendation']}")
    
    # Save report
    monitor.save_monitoring_report('reports/monitoring_demo.json')
    
    print("\n" + "="*60)
    print("END OF MONITORING DEMONSTRATION")
    print("="*60)
