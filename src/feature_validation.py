"""
Feature Validation Module
PDF ZORUNLULUĞU: Monitoring should incorporate statistical checks (like Great Expectations)
to perform feature validation and detect skew
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATION_DIR = os.path.join(BASE_DIR, 'data', 'validation')

class FeatureValidator:
    """Great Expectations benzeri feature validation sınıfı"""
    
    def __init__(self, baseline_path=None):
        os.makedirs(VALIDATION_DIR, exist_ok=True)
        self.baseline_path = baseline_path
        self.baseline_stats = None
        
        if baseline_path and os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                self.baseline_stats = json.load(f)
    
    def validate_numeric_feature(self, feature_name, values, baseline_stats=None):
        """Numeric feature validation"""
        if baseline_stats is None:
            baseline_stats = self.baseline_stats.get(feature_name, {}) if self.baseline_stats else {}
        
        if not baseline_stats:
            return {"valid": True, "reason": "no_baseline"}
        
        violations = []
        
        # Mean check (z-score)
        baseline_mean = baseline_stats.get('mean', 0)
        baseline_std = baseline_stats.get('std', 1)
        current_mean = np.mean(values)
        
        if baseline_std > 0:
            z_score = abs(current_mean - baseline_mean) / baseline_std
            if z_score > 2.0:  # 2 sigma rule
                violations.append({
                    "type": "mean_shift",
                    "severity": "high" if z_score > 3.0 else "medium",
                    "z_score": float(z_score),
                    "baseline_mean": baseline_mean,
                    "current_mean": float(current_mean)
                })
        
        # Range check
        baseline_min = baseline_stats.get('min', float('-inf'))
        baseline_max = baseline_stats.get('max', float('inf'))
        current_min = float(np.min(values))
        current_max = float(np.max(values))
        
        if current_min < baseline_min * 0.8 or current_max > baseline_max * 1.2:
            violations.append({
                "type": "range_violation",
                "severity": "medium",
                "baseline_range": [baseline_min, baseline_max],
                "current_range": [current_min, current_max]
            })
        
        # Null check
        null_count = pd.Series(values).isna().sum()
        if null_count > len(values) * 0.1:  # %10'dan fazla null
            violations.append({
                "type": "high_null_rate",
                "severity": "high",
                "null_count": int(null_count),
                "null_rate": float(null_count / len(values))
            })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations
        }
    
    def validate_categorical_feature(self, feature_name, values, baseline_dist=None):
        """Categorical feature validation"""
        if baseline_dist is None:
            baseline_dist = self.baseline_stats.get(feature_name, {}).get('value_counts', {}) if self.baseline_stats else {}
        
        if not baseline_dist:
            return {"valid": True, "reason": "no_baseline"}
        
        violations = []
        current_dist = pd.Series(values).value_counts().to_dict()
        
        # Distribution shift check
        total_baseline = sum(baseline_dist.values())
        total_current = sum(current_dist.values())
        
        for key in set(list(baseline_dist.keys()) + list(current_dist.keys())):
            baseline_pct = baseline_dist.get(key, 0) / total_baseline if total_baseline > 0 else 0
            current_pct = current_dist.get(key, 0) / total_current if total_current > 0 else 0
            
            if abs(baseline_pct - current_pct) > 0.15:  # %15'ten fazla değişim
                violations.append({
                    "type": "distribution_shift",
                    "severity": "medium",
                    "category": str(key),
                    "baseline_pct": float(baseline_pct),
                    "current_pct": float(current_pct)
                })
        
        # Unknown categories check
        baseline_categories = set(baseline_dist.keys())
        current_categories = set(current_dist.keys())
        unknown_categories = current_categories - baseline_categories
        
        if unknown_categories:
            violations.append({
                "type": "unknown_categories",
                "severity": "high",
                "categories": list(unknown_categories)
            })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations
        }
    
    def validate_dataframe(self, df, feature_cols=None):
        """Tüm DataFrame'i validate et"""
        if feature_cols is None:
            feature_cols = df.columns.tolist()
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "total_features": len(feature_cols),
            "validated_features": {},
            "overall_valid": True,
            "violations_summary": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            values = df[col].values
            
            # Numeric veya categorical kontrolü
            if df[col].dtype in ['int64', 'float64']:
                result = self.validate_numeric_feature(col, values)
            else:
                result = self.validate_categorical_feature(col, values)
            
            validation_results["validated_features"][col] = result
            
            # Violations sayısını topla
            if not result.get("valid", True):
                for violation in result.get("violations", []):
                    severity = violation.get("severity", "low")
                    validation_results["violations_summary"][severity] += 1
                    validation_results["overall_valid"] = False
        
        return validation_results
    
    def save_validation_result(self, result, filename=None):
        """Validation sonucunu kaydet"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_{timestamp}.json"
        
        filepath = os.path.join(VALIDATION_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return filepath

def validate_features(df, baseline_path=None):
    """Feature validation fonksiyonu"""
    validator = FeatureValidator(baseline_path)
    result = validator.validate_dataframe(df)
    filepath = validator.save_validation_result(result)
    
    print("=" * 60)
    print("FEATURE VALIDATION SONUCLARI")
    print("=" * 60)
    print(f"Toplam Feature: {result['total_features']}")
    print(f"Genel Durum: {'[OK] VALID' if result['overall_valid'] else '[WARNING] VIOLATIONS DETECTED'}")
    print(f"Yuksek Oncelikli: {result['violations_summary']['high']}")
    print(f"Orta Oncelikli: {result['violations_summary']['medium']}")
    print(f"Sonuc dosyasi: {filepath}")
    print("=" * 60)
    
    return result

if __name__ == "__main__":
    # Test için
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        df = pd.read_csv(test_file)
        baseline_path = os.path.join(BASE_DIR, 'data', 'baseline_stats.json')
        validate_features(df, baseline_path)

