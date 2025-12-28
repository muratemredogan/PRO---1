"""
Continuous Model Evaluation (CME) Module
PDF ZORUNLULUĞU: CONTINUED MODEL EVALUATION design pattern
Model degradation ve data distribution shift'leri tespit eder.
"""
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import json
from datetime import datetime
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CME_RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'cme_results')
MODEL_NAME = "IBM_Attrition_Model"
THRESHOLD_ACCURACY = 0.70  # Minimum kabul edilebilir accuracy
THRESHOLD_F1 = 0.60  # Minimum kabul edilebilir F1 score

def load_production_model():
    """Production'daki en son modeli yükle"""
    client = MlflowClient()
    try:
        # Production stage'deki modeli al
        model_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not model_version:
            # Production yoksa staging'den al
            model_version = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        
        if not model_version:
            raise ValueError("Model Registry'de model bulunamadı!")
        
        model_uri = f"models:/{MODEL_NAME}/{model_version[0].version}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"[OK] Production model yüklendi: Version {model_version[0].version}")
        return model, model_version[0].version
    except Exception as e:
        print(f"[ERROR] Model yükleme hatası: {e}")
        return None, None

def calculate_feature_statistics(df, feature_cols):
    """Feature'ların istatistiksel özelliklerini hesapla (drift detection için)"""
    stats = {}
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
        else:
            # Kategorik için value counts
            stats[col] = {
                'value_counts': df[col].value_counts().to_dict()
            }
    return stats

def detect_feature_drift(baseline_stats, current_stats, threshold=0.2):
    """Feature drift tespit et (baseline ile karşılaştır)"""
    drift_detected = {}
    
    for col in baseline_stats.keys():
        if col not in current_stats:
            drift_detected[col] = {"type": "missing", "severity": "high"}
            continue
        
        if 'mean' in baseline_stats[col]:  # Numeric feature
            baseline_mean = baseline_stats[col]['mean']
            current_mean = current_stats[col]['mean']
            
            # Mean shift detection
            mean_shift = abs(baseline_mean - current_mean) / (baseline_stats[col]['std'] + 1e-6)
            if mean_shift > threshold:
                drift_detected[col] = {
                    "type": "mean_shift",
                    "severity": "high" if mean_shift > threshold * 2 else "medium",
                    "baseline_mean": baseline_mean,
                    "current_mean": current_mean,
                    "shift_ratio": mean_shift
                }
        else:  # Categorical feature
            baseline_dist = baseline_stats[col].get('value_counts', {})
            current_dist = current_stats[col].get('value_counts', {})
            
            # Distribution shift detection (basit versiyon)
            total_baseline = sum(baseline_dist.values())
            total_current = sum(current_dist.values())
            
            for key in set(list(baseline_dist.keys()) + list(current_dist.keys())):
                baseline_pct = baseline_dist.get(key, 0) / total_baseline if total_baseline > 0 else 0
                current_pct = current_dist.get(key, 0) / total_current if total_current > 0 else 0
                
                if abs(baseline_pct - current_pct) > threshold:
                    drift_detected[col] = {
                        "type": "distribution_shift",
                        "severity": "medium",
                        "baseline_dist": baseline_dist,
                        "current_dist": current_dist
                    }
                    break
    
    return drift_detected

def evaluate_model_performance(model, X_test, y_test, feature_cols):
    """Model performansını değerlendir"""
    X_test_ordered = X_test[feature_cols]
    y_pred = model.predict(X_test_ordered)
    y_pred_proba = model.predict_proba(X_test_ordered)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    return metrics

def run_cme(test_data_path=None, baseline_stats_path=None):
    """
    Continuous Model Evaluation çalıştır
    
    Args:
        test_data_path: Test verisi yolu (None ise data/test.csv kullanır)
        baseline_stats_path: Baseline feature statistics yolu
    """
    print("=" * 60)
    print("CONTINUOUS MODEL EVALUATION (CME) BAŞLATILIYOR...")
    print("=" * 60)
    
    os.makedirs(CME_RESULTS_DIR, exist_ok=True)
    
    # 1. Production modelini yükle
    model, model_version = load_production_model()
    if model is None:
        print("[ERROR] Model yüklenemedi, CME çalıştırılamıyor!")
        return None
    
    # 2. Test verisini yükle
    if test_data_path is None:
        test_data_path = os.path.join(BASE_DIR, 'data', 'test.csv')
    
    if not os.path.exists(test_data_path):
        print(f"[ERROR] Test verisi bulunamadı: {test_data_path}")
        return None
    
    test_df = pd.read_csv(test_data_path)
    y_test = test_df['Attrition']
    feature_cols = [col for col in test_df.columns if col != 'Attrition']
    X_test = test_df[feature_cols]
    
    # 3. Model performansını değerlendir
    print("\n[1/4] Model performansı değerlendiriliyor...")
    metrics = evaluate_model_performance(model, X_test, y_test, feature_cols)
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   F1 Score: {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # 4. Performance degradation kontrolü
    print("\n[2/4] Performance degradation kontrol ediliyor...")
    performance_issues = []
    if metrics['accuracy'] < THRESHOLD_ACCURACY:
        performance_issues.append({
            "metric": "accuracy",
            "value": metrics['accuracy'],
            "threshold": THRESHOLD_ACCURACY,
            "severity": "high"
        })
    
    if metrics['f1_score'] < THRESHOLD_F1:
        performance_issues.append({
            "metric": "f1_score",
            "value": metrics['f1_score'],
            "threshold": THRESHOLD_F1,
            "severity": "high"
        })
    
    # 5. Feature drift detection
    print("\n[3/4] Feature drift tespiti yapılıyor...")
    current_stats = calculate_feature_statistics(X_test, feature_cols)
    
    drift_detected = {}
    if baseline_stats_path and os.path.exists(baseline_stats_path):
        with open(baseline_stats_path, 'r') as f:
            baseline_stats = json.load(f)
        drift_detected = detect_feature_drift(baseline_stats, current_stats)
        print(f"   {len(drift_detected)} feature'da drift tespit edildi.")
    else:
        print("   [WARNING] Baseline statistics bulunamadı, drift detection atlandı.")
        print("   [INFO] İlk çalıştırma - baseline statistics kaydediliyor...")
        baseline_stats = current_stats
    
    # 6. Sonuçları kaydet
    print("\n[4/4] Sonuçlar kaydediliyor...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "model_version": model_version,
        "metrics": metrics,
        "performance_issues": performance_issues,
        "drift_detected": drift_detected,
        "thresholds": {
            "accuracy": THRESHOLD_ACCURACY,
            "f1_score": THRESHOLD_F1
        }
    }
    
    result_path = os.path.join(CME_RESULTS_DIR, f"cme_result_{timestamp}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    # Baseline statistics'i kaydet (ilk çalıştırmada)
    if not os.path.exists(baseline_stats_path or ""):
        baseline_path = os.path.join(BASE_DIR, 'data', 'baseline_stats.json')
        with open(baseline_path, 'w') as f:
            json.dump(baseline_stats, f, indent=2, default=str)
        print(f"   Baseline statistics kaydedildi: {baseline_path}")
    
    # 7. Özet rapor
    print("\n" + "=" * 60)
    print("CME SONUÇ ÖZETİ")
    print("=" * 60)
    print(f"Model Version: {model_version}")
    print(f"Accuracy: {metrics['accuracy']:.4f} (Threshold: {THRESHOLD_ACCURACY})")
    print(f"F1 Score: {metrics['f1_score']:.4f} (Threshold: {THRESHOLD_F1})")
    
    if performance_issues:
        print(f"\n⚠️  PERFORMANCE DEGRADATION TESPİT EDİLDİ!")
        for issue in performance_issues:
            print(f"   - {issue['metric']}: {issue['value']:.4f} < {issue['threshold']}")
    else:
        print("\n✅ Model performansı kabul edilebilir seviyede.")
    
    if drift_detected:
        print(f"\n⚠️  FEATURE DRIFT TESPİT EDİLDİ: {len(drift_detected)} feature")
        for col, info in list(drift_detected.items())[:5]:  # İlk 5'ini göster
            print(f"   - {col}: {info.get('type', 'unknown')} ({info.get('severity', 'unknown')})")
    else:
        print("\n✅ Feature drift tespit edilmedi.")
    
    print(f"\nDetaylı sonuç: {result_path}")
    print("=" * 60)
    
    return result

if __name__ == "__main__":
    # Baseline stats path
    baseline_path = os.path.join(BASE_DIR, 'data', 'baseline_stats.json')
    run_cme(baseline_stats_path=baseline_path)

