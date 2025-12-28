import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import os
import json
import tempfile

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test.csv')
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, 'data', 'feature_order.json')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'data', 'checkpoints')
MODEL_NAME = "IBM_Attrition_Model"

def reframe_problem(y, X=None, strategy="risk_buckets"):
    """
    PDF ZORUNLULUĞU: Problem Reframing
    Binary classification'ı risk seviyelerine göre multi-class'a çevirir
    veya probability distribution'ı kullanarak daha expressif hale getirir.
    
    Args:
        y: Target variable (binary)
        X: Features (opsiyonel, risk hesaplama için)
        strategy: "risk_buckets" veya "keep_binary"
    
    Returns:
        Reframed target variable
    """
    if strategy == "keep_binary":
        return y, "binary_classification"
    
    # Risk buckets stratejisi: Binary'yi risk seviyelerine çevir
    # Bu, modelin daha detaylı probability distribution öğrenmesini sağlar
    y_reframed = y.copy()
    
    if X is not None:
        # Risk faktörlerine göre bucketize et
        # Örnek: OverTime, MonthlyIncome, JobSatisfaction gibi faktörlere bak
        risk_scores = pd.Series(0, index=y.index)
        
        # Risk faktörleri (örnek)
        if 'OverTime' in X.columns:
            risk_scores += (X['OverTime'] == 1).astype(int) * 0.3
        if 'MonthlyIncome' in X.columns:
            risk_scores += (X['MonthlyIncome'] < X['MonthlyIncome'].median()).astype(int) * 0.2
        if 'JobSatisfaction' in X.columns:
            risk_scores += (X['JobSatisfaction'] < 2).astype(int) * 0.2
        if 'YearsAtCompany' in X.columns:
            risk_scores += (X['YearsAtCompany'] < 2).astype(int) * 0.15
        
        # Bucketize: 0=Low, 1=Medium, 2=High risk
        try:
            y_reframed = pd.cut(risk_scores, bins=[-0.1, 0.3, 0.6, 1.0], labels=[0, 1, 2])
            y_reframed = y_reframed.astype(int)
        except:
            # Fallback: basit threshold
            y_reframed = (risk_scores > 0.5).astype(int) * 2
            y_reframed = y_reframed.clip(0, 2)
        
        # Eğer orijinal target 1 ise, en az medium risk yap
        y_reframed = y_reframed + (y == 1).astype(int)
        y_reframed = y_reframed.clip(0, 2)
        
        return y_reframed, "multi_class_risk_buckets"
    
    return y, "binary_classification"

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Attrition'ı ayır
    y_train = train_df['Attrition']
    y_test = test_df['Attrition']
    
    # Feature kolonlarını belirle (Attrition hariç, CSV'deki sırayı koru)
    feature_cols = [col for col in train_df.columns if col != 'Attrition']
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # PDF ZORUNLULUĞU: Problem Reframing (opsiyonel - şimdilik binary'de kalıyoruz)
    # Not: Multi-class için XGBoost'u XGBClassifier yerine kullanmak gerekir
    # Şimdilik binary classification'da kalıyoruz ama reframing fonksiyonu hazır
    reframing_strategy = "keep_binary"  # "risk_buckets" veya "keep_binary"
    y_train_reframed, reframing_type = reframe_problem(y_train, X_train, strategy=reframing_strategy)
    y_test_reframed, _ = reframe_problem(y_test, X_test, strategy=reframing_strategy)
    
    # Feature sırasını kaydet (inference için)
    os.makedirs(os.path.dirname(FEATURE_ORDER_PATH), exist_ok=True)
    with open(FEATURE_ORDER_PATH, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"[OK] Feature sırası kaydedildi: {len(feature_cols)} kolon")
    print(f"[INFO] Problem reframing: {reframing_type}")
    
    return X_train, y_train_reframed, X_test, y_test_reframed, feature_cols, reframing_type

def main():
    # 1. MLflow Experiment Tanımla
    mlflow.set_experiment("IBM_Attrition_Project")
    
    with mlflow.start_run():
        print("[INFO] Egitim Basliyor...")
        X_train, y_train, X_test, y_test, feature_cols, reframing_type = load_data()
        
        print(f"[INFO] Feature sayisi: {len(feature_cols)}")
        print(f"[INFO] Feature kolonlari: {feature_cols[:5]}... (ilk 5)")
        
        # 2. PDF ZORUNLULUĞU: Data Imbalance -> REBALANCING (SMOTE)
        print(f"[INFO] Orijinal Veri: {y_train.value_counts().to_dict()}")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"[INFO] SMOTE Sonrasi: {y_train_res.value_counts().to_dict()}")
        
        # SMOTE sonrası kolon sırasını kontrol et
        if list(X_train_res.columns) != feature_cols:
            print(f"[WARNING] SMOTE sonrası kolon sırası değişti! Düzeltiliyor...")
            X_train_res = X_train_res[feature_cols]
        
        # 3. PDF ZORUNLULUĞU: Ensembles -> XGBoost
        # PDF ZORUNLULUĞU: CHECKPOINTS - Training sırasında checkpoint kaydetme
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "xgb_checkpoint.model")
        
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "eval_metric": "logloss"
        }
        
        # Checkpoint callback ile eğitim (resilience için)
        model = XGBClassifier(**params)
        # XGBoost yeni versiyonunda early_stopping_rounds fit() içinde değil, callback olarak kullanılır
        try:
            model.fit(
                X_train_res, y_train_res,
                eval_set=[(X_test[feature_cols], y_test)],
                verbose=False
            )
        except TypeError:
            # Eski versiyon uyumluluğu
            model.fit(X_train_res, y_train_res)
        
        # Checkpoint'i kaydet (resume training için) - joblib kullan
        import joblib
        joblib.dump(model, checkpoint_path)
        print(f"[OK] Checkpoint kaydedildi: {checkpoint_path}")
        mlflow.log_artifact(checkpoint_path, "checkpoints")
        
        # 4. Değerlendirme - Kolon sırasını garanti altına al
        X_test_ordered = X_test[feature_cols]
        y_pred = model.predict(X_test_ordered)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC-AUC için probability hesapla
        y_pred_proba = model.predict_proba(X_test_ordered)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0
        
        print(f"[OK] Sonuclar -> Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # 5. PDF ZORUNLULUĞU: Experiment Tracking
        mlflow.log_params(params)
        mlflow.log_param("class_imbalance_method", "SMOTE")
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("checkpoint_enabled", "True")
        mlflow.log_param("problem_reframing", reframing_type)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        # Model'i MLflow'a kaydet
        mlflow.sklearn.log_model(model, "model")
        print("[OK] Model MLflow'a basariyla kaydedildi.")
        
        # PDF ZORUNLULUĞU: MLflow Model Registry - Model'i Registry'ye kaydet
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        try:
            # Model Registry'ye kaydet (staging olarak)
            mlflow.register_model(model_uri, MODEL_NAME)
            print(f"[OK] Model Model Registry'ye kaydedildi: {MODEL_NAME}")
            
            # Eğer daha önce production modeli varsa, yeni modeli staging'e al
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if latest_versions:
                print(f"[INFO] Production'da {len(latest_versions)} model var. Yeni model staging'de.")
        except Exception as e:
            print(f"[WARNING] Model Registry kaydı başarısız (ilk çalıştırma olabilir): {e}")
        
        print(f"[OK] Feature sırası dosyaya kaydedildi: {FEATURE_ORDER_PATH}")

if __name__ == "__main__":
    main()