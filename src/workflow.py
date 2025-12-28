from prefect import task, flow
import os

# --- GÖREV 1: Veri Hazırlığı ---
@task(name="1. Data Preprocessing", log_prints=True)
def run_preprocessing():
    print("🔄 Preprocessing başlatılıyor...")
    exit_code = os.system("python src/preprocess.py")
    
    if exit_code != 0:
        raise Exception("❌ Preprocessing adımında hata oluştu!")
    print("✅ Veri işleme tamamlandı.")

# --- GÖREV 2: Feature Validation ---
@task(name="2. Feature Validation", log_prints=True)
def run_feature_validation():
    print("🔄 Feature validation başlatılıyor...")
    exit_code = os.system("python -c \"import sys; sys.path.insert(0, 'src'); from feature_validation import validate_features; import pandas as pd; df = pd.read_csv('data/test.csv'); validate_features(df)\"")
    
    if exit_code != 0:
        print("⚠️ Feature validation uyarıları var, devam ediliyor...")
    print("✅ Feature validation tamamlandı.")

# --- GÖREV 3: Model Eğitimi ---
@task(name="3. Model Training", log_prints=True)
def run_training():
    print("🔄 Model eğitimi başlatılıyor...")
    exit_code = os.system("python src/train.py")
    
    if exit_code != 0:
        raise Exception("❌ Training adımında hata oluştu!")
    print("✅ Model eğitimi ve MLflow kaydı tamamlandı.")

# --- GÖREV 4: Continuous Model Evaluation ---
@task(name="4. Continuous Model Evaluation", log_prints=True)
def run_cme():
    print("🔄 Continuous Model Evaluation başlatılıyor...")
    exit_code = os.system("python src/cme.py")
    
    if exit_code != 0:
        print("⚠️ CME uyarıları var, devam ediliyor...")
    print("✅ CME tamamlandı.")

# --- AKIŞ YÖNETİCİSİ (FLOW) ---
@flow(name="IBM Attrition MLOps Pipeline")
def main_flow():
    print("🚀 MLOps Pipeline Tetiklendi!")
    print("=" * 60)
    
    # Adımları sırayla yap
    run_preprocessing()
    run_feature_validation()
    run_training()
    run_cme()
    
    print("=" * 60)
    print("✅ Tüm pipeline adımları tamamlandı!")

if __name__ == "__main__":
    main_flow()