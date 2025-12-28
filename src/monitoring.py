"""
ML-Specific Monitoring Module
PDF ZORUNLULUĞU: Monitoring must track ML-specific metrics related to features and predictions
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from collections import defaultdict
import mlflow
from mlflow.tracking import MlflowClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONITORING_DIR = os.path.join(BASE_DIR, 'data', 'monitoring')
MODEL_NAME = "IBM_Attrition_Model"

class MLMonitor:
    """ML-specific metrics monitoring sınıfı"""
    
    def __init__(self):
        os.makedirs(MONITORING_DIR, exist_ok=True)
        self.prediction_history = []
        self.feature_history = []
        
    def log_prediction(self, input_features, prediction, probability, timestamp=None):
        """Prediction'ı logla"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "prediction": int(prediction),
            "probability": float(probability),
            "features": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                       for k, v in input_features.items()}
        }
        
        self.prediction_history.append(log_entry)
        
        # Her 100 prediction'da bir dosyaya kaydet
        if len(self.prediction_history) % 100 == 0:
            self.save_history()
    
    def calculate_prediction_statistics(self, window_size=1000):
        """Son N prediction'ın istatistiklerini hesapla"""
        if len(self.prediction_history) == 0:
            return None
        
        recent = self.prediction_history[-window_size:]
        
        predictions = [p['prediction'] for p in recent]
        probabilities = [p['probability'] for p in recent]
        
        stats = {
            "total_predictions": len(recent),
            "positive_rate": sum(predictions) / len(predictions) if predictions else 0,
            "avg_confidence": np.mean(probabilities) if probabilities else 0,
            "min_confidence": np.min(probabilities) if probabilities else 0,
            "max_confidence": np.max(probabilities) if probabilities else 0,
            "std_confidence": np.std(probabilities) if probabilities else 0,
            "high_confidence_count": sum(1 for p in probabilities if p > 0.8),
            "low_confidence_count": sum(1 for p in probabilities if p < 0.5)
        }
        
        return stats
    
    def calculate_feature_statistics(self, window_size=1000):
        """Feature'ların dağılım istatistiklerini hesapla"""
        if len(self.prediction_history) == 0:
            return None
        
        recent = self.prediction_history[-window_size:]
        
        # Tüm feature'ları topla
        all_features = defaultdict(list)
        for entry in recent:
            for key, value in entry['features'].items():
                if isinstance(value, (int, float, np.number)):
                    all_features[key].append(float(value))
        
        feature_stats = {}
        for feature_name, values in all_features.items():
            if len(values) > 0:
                feature_stats[feature_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75))
                }
        
        return feature_stats
    
    def detect_prediction_shift(self, baseline_positive_rate=0.16, threshold=0.05):
        """Prediction distribution shift tespit et"""
        stats = self.calculate_prediction_statistics()
        if stats is None:
            return None
        
        current_positive_rate = stats['positive_rate']
        shift = abs(current_positive_rate - baseline_positive_rate)
        
        if shift > threshold:
            return {
                "detected": True,
                "baseline_positive_rate": baseline_positive_rate,
                "current_positive_rate": current_positive_rate,
                "shift": shift,
                "severity": "high" if shift > threshold * 2 else "medium"
            }
        
        return {"detected": False}
    
    def detect_feature_skew(self, baseline_stats=None, threshold=0.2):
        """Feature skew detection (Great Expectations benzeri)"""
        if baseline_stats is None:
            return None
        
        current_stats = self.calculate_feature_statistics()
        if current_stats is None:
            return None
        
        skew_detected = {}
        for feature_name in baseline_stats.keys():
            if feature_name not in current_stats:
                continue
            
            baseline_mean = baseline_stats[feature_name].get('mean', 0)
            current_mean = current_stats[feature_name].get('mean', 0)
            baseline_std = baseline_stats[feature_name].get('std', 1)
            
            # Z-score benzeri kontrol
            if baseline_std > 0:
                z_score = abs(current_mean - baseline_mean) / baseline_std
                if z_score > threshold:
                    skew_detected[feature_name] = {
                        "z_score": float(z_score),
                        "baseline_mean": baseline_mean,
                        "current_mean": current_mean,
                        "severity": "high" if z_score > threshold * 2 else "medium"
                    }
        
        return skew_detected if skew_detected else None
    
    def save_history(self):
        """Prediction history'yi dosyaya kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = os.path.join(MONITORING_DIR, f"predictions_{timestamp}.json")
        
        with open(history_path, 'w') as f:
            json.dump(self.prediction_history, f, indent=2, default=str)
        
        print(f"[OK] Prediction history kaydedildi: {history_path}")
    
    def get_monitoring_report(self, baseline_stats_path=None):
        """Kapsamlı monitoring raporu oluştur"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "prediction_stats": self.calculate_prediction_statistics(),
            "feature_stats": self.calculate_feature_statistics(),
            "prediction_shift": self.detect_prediction_shift(),
        }
        
        # Feature skew detection
        if baseline_stats_path and os.path.exists(baseline_stats_path):
            with open(baseline_stats_path, 'r') as f:
                baseline_stats = json.load(f)
            report["feature_skew"] = self.detect_feature_skew(baseline_stats)
        
        return report
    
    def log_to_mlflow(self, report):
        """Monitoring metriklerini MLflow'a logla"""
        try:
            mlflow.set_experiment("IBM_Attrition_Monitoring")
            with mlflow.start_run():
                if report.get("prediction_stats"):
                    stats = report["prediction_stats"]
                    mlflow.log_metric("monitoring_positive_rate", stats.get("positive_rate", 0))
                    mlflow.log_metric("monitoring_avg_confidence", stats.get("avg_confidence", 0))
                    mlflow.log_metric("monitoring_high_confidence_count", stats.get("high_confidence_count", 0))
                    mlflow.log_metric("monitoring_low_confidence_count", stats.get("low_confidence_count", 0))
                
                if report.get("prediction_shift") and report["prediction_shift"].get("detected"):
                    mlflow.log_metric("monitoring_prediction_shift", report["prediction_shift"]["shift"])
                
                print("[OK] Monitoring metrikleri MLflow'a loglandı.")
        except Exception as e:
            print(f"[WARNING] MLflow logging hatası: {e}")

# Global monitor instance
_global_monitor = None

def get_monitor():
    """Global monitor instance'ı al"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MLMonitor()
    return _global_monitor

