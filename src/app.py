import pandas as pd
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import glob
import joblib
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
try:
    from .monitoring import get_monitor
except ImportError:
    from monitoring import get_monitor

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLRUNS_DIR = os.path.join(BASE_DIR, 'mlruns')
ENCODERS_DIR = os.path.join(BASE_DIR, 'data', 'encoders')
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, 'data', 'feature_order.json')
MODEL_NAME = "IBM_Attrition_Model"

app = FastAPI(title="IBM Attrition Prediction API", version="1.0")

# Monitoring instance
monitor = get_monitor()

def get_latest_model_physically():
    """
    V4: Önce en son run'ı bul, sonra onun modelini yükle.
    Bu, yeni eğitilen modellerin öncelikli olarak yüklenmesini sağlar.
    """
    print(f"[INFO] En son run araniyor...")
    
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        # En son run'ı bul
        runs = client.search_runs(experiment_ids=['1'], order_by=['start_time desc'], max_results=1)
        if runs:
            latest_run = runs[0]
            model_uri = f"runs:/{latest_run.info.run_id}/model"
            print(f"[OK] En son run bulundu: {latest_run.info.run_id}")
            return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"[WARNING] Run'dan model yuklenemedi, fiziksel arama yapiliyor: {e}")
    
    # Fallback: Fiziksel arama
    print(f"[INFO] Derinlemesine taranıyor: {MLRUNS_DIR}")
    
    # mlruns altındaki TÜM klasörlerde 'MLmodel' dosyasını ara
    # ** operatörü alt klasörlere inmeyi sağlar
    search_pattern = os.path.join(MLRUNS_DIR, "**", "MLmodel")
    found_files = glob.glob(search_pattern, recursive=True)
    
    if not found_files:
        raise FileNotFoundError(f"[ERROR] '{MLRUNS_DIR}' icinde hicbir MLmodel dosyasi bulunamadi. train.py calistirildi mi?")

    # En son değiştirilen MLmodel dosyasını bul
    latest_file = max(found_files, key=os.path.getmtime)
    
    # MLmodel dosyasının bulunduğu klasör, modelin URI adresidir
    model_dir = os.path.dirname(latest_file)
    
    # Windows path düzeltmesi
    model_uri = "file:///" + model_dir.replace("\\", "/")
    
    print(f"[OK] Model bulundu: {model_dir}")
    return mlflow.sklearn.load_model(model_uri)

# Encoder'ları yükle
def load_encoders():
    """Tüm encoder'ları yükle"""
    encoders = {}
    if not os.path.exists(ENCODERS_DIR):
        print(f"[WARNING] Encoder klasoru bulunamadi: {ENCODERS_DIR}")
        return encoders
    
    # PDF ZORUNLULUĞU: High Cardinality -> HASHED FEATURE Pattern
    # JobRole için FeatureHasher yükle
    hasher_path = os.path.join(ENCODERS_DIR, 'jobrole_hasher.pkl')
    if os.path.exists(hasher_path):
        encoders['JobRole'] = joblib.load(hasher_path)
        print("[OK] JobRole FeatureHasher yuklendi (Hashed Features pattern).")
    else:
        print("[WARNING] JobRole FeatureHasher bulunamadi, LabelEncoder deneniyor...")
        # Geriye dönük uyumluluk için LabelEncoder'ı dene
        encoder_path = os.path.join(ENCODERS_DIR, 'JobRole_encoder.pkl')
        if os.path.exists(encoder_path):
            encoders['JobRole'] = joblib.load(encoder_path)
            print("[OK] JobRole LabelEncoder yuklendi (geriye donuk uyumluluk).")
    
    # Diğer kategorik feature'lar için LabelEncoder
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'MaritalStatus', 'OverTime']
    for col in categorical_cols:
        encoder_path = os.path.join(ENCODERS_DIR, f'{col}_encoder.pkl')
        if os.path.exists(encoder_path):
            encoders[col] = joblib.load(encoder_path)
            print(f"[OK] {col} LabelEncoder yuklendi.")
        else:
            print(f"[WARNING] {col} encoder bulunamadi: {encoder_path}")
    
    return encoders

# Feature sırasını yükle
def load_feature_order():
    """Model eğitilirken kullanılan feature sırasını yükle"""
    if not os.path.exists(FEATURE_ORDER_PATH):
        print(f"[WARNING] Feature sırası dosyasi bulunamadi: {FEATURE_ORDER_PATH}")
        return None
    
    with open(FEATURE_ORDER_PATH, 'r') as f:
        feature_order = json.load(f)
    print(f"[OK] Feature sırası yuklendi: {len(feature_order)} kolon")
    return feature_order

# --- PREPROCESSING FONKSİYONU ---
def prepare_features(input_data: Dict[str, Any], encoders: Dict, feature_order: list, model) -> pd.DataFrame:
    """
    Çalışan verisini model için hazırlar (feature engineering + encoding + sıralama)
    Hem orijinal hem simülasyon için kullanılabilir.
    """
    df = pd.DataFrame([input_data])
    
    # 1. Feature Engineering: Model'in beklediği tüm feature'ları oluştur
    # Age_DailyRate
    df['Age_DailyRate'] = df['Age'] * df['DailyRate']
    
    # MonthlyIncome_JobLevel
    if 'MonthlyIncome' in df.columns and 'JobLevel' in df.columns:
        df['MonthlyIncome_JobLevel'] = df['MonthlyIncome'] * df['JobLevel']
    
    # YearsAtCompany_TotalWorkingYears
    if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
        df['YearsAtCompany_TotalWorkingYears'] = df['YearsAtCompany'] * df['TotalWorkingYears']
    
    # Age_Binned (0-30: 0, 31-40: 1, 41-50: 2, 51+: 3)
    if 'Age' in df.columns:
        age = df['Age'].iloc[0]
        if age <= 30:
            df['Age_Binned'] = 0
        elif age <= 40:
            df['Age_Binned'] = 1
        elif age <= 50:
            df['Age_Binned'] = 2
        else:
            df['Age_Binned'] = 3
    
    # MonthlyIncome_Binned (basitleştirilmiş: median ve q75 değerlerini kullan)
    if 'MonthlyIncome' in df.columns:
        income = df['MonthlyIncome'].iloc[0]
        # Eğitim verisinden alınan değerler (yaklaşık)
        income_median = 4919
        income_q75 = 8379
        if income <= income_median:
            df['MonthlyIncome_Binned'] = 0
        elif income <= income_q75:
            df['MonthlyIncome_Binned'] = 1
        else:
            df['MonthlyIncome_Binned'] = 2
    
    # Income_to_Level_Ratio
    if 'MonthlyIncome' in df.columns and 'JobLevel' in df.columns:
        df['Income_to_Level_Ratio'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
    
    # Company_Loyalty
    if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
        df['Company_Loyalty'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    
    # WorkLife_OverTime_Stress (OverTime henüz encode edilmedi, string olarak kontrol et)
    if 'WorkLifeBalance' in df.columns and 'OverTime' in df.columns:
        overtime_numeric = 1 if df['OverTime'].iloc[0] == 'Yes' else 0
        df['WorkLife_OverTime_Stress'] = df['WorkLifeBalance'] * overtime_numeric
    
    # 2. Kategorik feature'ları encode et
    # PDF ZORUNLULUĞU: JobRole için Hashed Features pattern
    if 'JobRole' in df.columns and 'JobRole' in encoders:
        hasher = encoders['JobRole']
        # FeatureHasher mı yoksa LabelEncoder mı kontrol et
        if hasattr(hasher, 'transform'):  # FeatureHasher
            # JobRole değerini string olarak hash'le
            jobrole_value = str(df['JobRole'].iloc[0])
            jobrole_hashed = hasher.transform([[jobrole_value]]).toarray()[0]
            # Hash'lenmiş feature'ları ekle (JobRole_Hash_0, JobRole_Hash_1, ...)
            for i in range(len(jobrole_hashed)):
                df[f'JobRole_Hash_{i}'] = jobrole_hashed[i]
            # Orijinal JobRole kolonunu sil
            df = df.drop(columns=['JobRole'])
        else:  # LabelEncoder (geriye dönük uyumluluk)
            le = hasher
            try:
                df['JobRole'] = le.transform([df['JobRole'].iloc[0]])[0]
            except ValueError:
                print(f"[WARNING] JobRole icin bilinmeyen deger: {df['JobRole'].iloc[0]}, 0 kullaniliyor.")
                df['JobRole'] = 0
    
    # Diğer kategorik feature'lar için Label Encoding
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'MaritalStatus', 'OverTime']
    for col in categorical_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            # Eğer değer encoder'da yoksa, en yaygın değeri kullan
            try:
                df[col] = le.transform([df[col].iloc[0]])[0]
            except ValueError:
                # Bilinmeyen değer için ilk kategoriyi kullan (0)
                print(f"[WARNING] {col} icin bilinmeyen deger: {df[col].iloc[0]}, 0 kullaniliyor.")
                df[col] = 0
    
    # 3. Modelin beklediği kolonları kontrol et ve sırala
    # ÖNCE feature_order.json'a bak (train.csv'den alınmış, en güvenilir)
    # Sonra model'in feature_names_in_ özelliğini kontrol et
    if feature_order:
        required_cols = feature_order.copy()
        # feature_order.json zaten JobRole_Hash_* içeriyor olmalı
        # Ama eğer JobRole varsa ve biz JobRole_Hash_* oluşturduysak, güncelle
        if 'JobRole' in required_cols and 'JobRole_Hash_0' in df.columns:
            # JobRole'u JobRole_Hash_* ile değiştir
            jobrole_index = required_cols.index('JobRole')
            required_cols = required_cols[:jobrole_index] + [f'JobRole_Hash_{i}' for i in range(8)] + required_cols[jobrole_index+1:]
            print("[INFO] feature_order.json guncellendi: JobRole -> JobRole_Hash_0...7")
    elif hasattr(model, "feature_names_in_"):
        required_cols = list(model.feature_names_in_)
        # Model JobRole bekliyorsa ama biz JobRole_Hash_* oluşturduysak, model'in beklediği listeyi güncelle
        if 'JobRole' in required_cols and 'JobRole_Hash_0' in df.columns:
            # JobRole'u JobRole_Hash_* ile değiştir
            jobrole_index = required_cols.index('JobRole')
            required_cols = required_cols[:jobrole_index] + [f'JobRole_Hash_{i}' for i in range(8)] + required_cols[jobrole_index+1:]
            print("[INFO] Model feature_names guncellendi: JobRole -> JobRole_Hash_0...7")
    else:
        # Son çare: DataFrame'deki mevcut kolonları kullan (sıralı)
        required_cols = sorted(df.columns.tolist())
    
    # Eksik kolonları 0 ile doldur
    for col in required_cols:
        if col not in df.columns:
            print(f"[WARNING] Eksik kolon: {col}, 0 ile dolduruluyor.")
            df[col] = 0
    
    # Kolonları modelin beklediği sıraya göre sırala
    df = df[required_cols]
    
    # Son kontrol: Kolon sayısı ve sırası
    if len(df.columns) != len(required_cols):
        raise ValueError(f"Kolon sayisi uyumsuz! Beklenen: {len(required_cols)}, Mevcut: {len(df.columns)}")
    
    return df

def predict_attrition(df: pd.DataFrame, input_data: Dict[str, Any], model) -> tuple:
    """
    Model ile tahmin yapar ve olasılık döndürür.
    Returns: (prediction_class, probability, model_source)
    """
    use_fallback = False
    model_source = "primary"
    
    if model is None:
        use_fallback = True
        model_source = "fallback_rule_based"
    else:
        try:
            prediction = model.predict(df)
            proba = model.predict_proba(df)
            attrition_probability = proba[0][1]  # Class 1 (attrition) olasılığı
            return prediction[0], attrition_probability, model_source
        except Exception as e:
            print(f"[WARNING] Model prediction hatası, fallback kullanılıyor: {e}")
            use_fallback = True
            model_source = "fallback_error"
    
    # Fallback kullanılıyorsa
    if use_fallback:
        prediction_val, confidence_val = fallback_prediction(input_data)
        return prediction_val, confidence_val, "fallback_rule_based"

# PDF ZORUNLULUĞU: Algorithmic Fallback - Basit rule-based fallback modeli
def fallback_prediction(input_data):
    """
    Basit rule-based fallback modeli
    Model başarısız olursa veya performans düşerse kullanılır
    """
    # Basit kurallar: OverTime, MonthlyIncome, YearsAtCompany gibi faktörlere bak
    risk_score = 0
    
    if input_data.get('OverTime', 'No') == 'Yes':
        risk_score += 0.3
    if input_data.get('MonthlyIncome', 5000) < 3000:
        risk_score += 0.2
    if input_data.get('YearsAtCompany', 5) < 2:
        risk_score += 0.2
    if input_data.get('JobSatisfaction', 3) < 2:
        risk_score += 0.15
    if input_data.get('WorkLifeBalance', 3) < 2:
        risk_score += 0.15
    
    prediction = 1 if risk_score > 0.5 else 0
    confidence = min(risk_score, 0.95) if prediction == 1 else min(1 - risk_score, 0.95)
    
    return prediction, confidence

# Uygulama başlarken modeli ve encoder'ları yükle
try:
    model = get_latest_model_physically()
    fallback_model = None  # Production modeli varsa fallback'e gerek yok
    print("[OK] Model bellege yuklendi, API hazir!")
except Exception as e:
    print(f"[WARNING] Model yukleme hatasi: {e}")
    model = None
    fallback_model = "rule_based"  # Fallback kullanılacak

encoders = load_encoders()
feature_order = load_feature_order()

# Model Registry'den production modelini yüklemeyi dene
def load_production_model():
    """Model Registry'den production modelini yükle"""
    try:
        client = MlflowClient()
        # FutureWarning düzeltmesi: get_latest_versions yerine search_model_versions kullan
        try:
            # Yeni API: search_model_versions kullan
            production_versions = client.search_model_versions(
                f"name='{MODEL_NAME}' AND status='Production'"
            )
            if production_versions:
                # En son versiyonu al
                latest_prod = max(production_versions, key=lambda x: x.version)
                model_uri = f"models:/{MODEL_NAME}/{latest_prod.version}"
                return mlflow.sklearn.load_model(model_uri), "production"
            
            # Staging'den dene
            staging_versions = client.search_model_versions(
                f"name='{MODEL_NAME}' AND status='Staging'"
            )
            if staging_versions:
                # En son versiyonu al
                latest_staging = max(staging_versions, key=lambda x: x.version)
                model_uri = f"models:/{MODEL_NAME}/{latest_staging.version}"
                return mlflow.sklearn.load_model(model_uri), "staging"
        except Exception as search_error:
            # Geriye dönük uyumluluk için eski API'yi dene
            try:
                model_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])
                if model_version:
                    model_uri = f"models:/{MODEL_NAME}/Production"
                    return mlflow.sklearn.load_model(model_uri), "production"
                else:
                    # Staging'den dene
                    model_version = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
                    if model_version:
                        model_uri = f"models:/{MODEL_NAME}/Staging"
                        return mlflow.sklearn.load_model(model_uri), "staging"
            except:
                pass
    except Exception as e:
        print(f"[WARNING] Model Registry'den yükleme hatası: {e}")
    return None, None

# Production modelini yüklemeyi dene
production_model, model_stage = load_production_model()
if production_model is not None:
    model = production_model
    print(f"[OK] Production model yüklendi (Stage: {model_stage})")

# --- İSTEK ŞABLONU ---
class EmployeeData(BaseModel):
    Age: int = 30
    DailyRate: int = 200
    DistanceFromHome: int = 5
    Education: int = 3
    EnvironmentSatisfaction: int = 4
    HourlyRate: int = 50
    JobInvolvement: int = 3
    JobLevel: int = 2
    JobSatisfaction: int = 4
    MonthlyIncome: int = 5000
    MonthlyRate: int = 10000
    NumCompaniesWorked: int = 1
    PercentSalaryHike: int = 15
    PerformanceRating: int = 3
    RelationshipSatisfaction: int = 4
    StockOptionLevel: int = 1
    TotalWorkingYears: int = 10
    TrainingTimesLastYear: int = 3
    WorkLifeBalance: int = 3
    YearsAtCompany: int = 5
    YearsInCurrentRole: int = 2
    YearsSinceLastPromotion: int = 1
    YearsWithCurrManager: int = 2
    # Kategorik feature'lar (string olarak gelecek, API'de encode edilecek)
    BusinessTravel: str = "Travel_Rarely"  # Travel_Rarely, Travel_Frequently, Non-Travel
    Department: str = "Sales"  # Sales, Research & Development, Human Resources
    EducationField: str = "Life Sciences"  # Life Sciences, Medical, Marketing, vb.
    Gender: str = "Male"  # Male, Female
    MaritalStatus: str = "Single"  # Single, Married, Divorced
    OverTime: str = "No"  # Yes, No
    JobRole: str = "Sales Executive"  # Sales Executive, Research Scientist, vb.

# --- WHAT-IF ANALİZİ İÇİN YENİ MODELLER ---
class SimulationChanges(BaseModel):
    """Değiştirilecek parametreler için model"""
    MonthlyIncome: Optional[int] = None
    OverTime: Optional[str] = None
    DistanceFromHome: Optional[int] = None
    JobSatisfaction: Optional[int] = None
    WorkLifeBalance: Optional[int] = None
    JobLevel: Optional[int] = None
    PercentSalaryHike: Optional[int] = None
    YearsAtCompany: Optional[int] = None
    # Diğer değiştirilebilir parametreler
    Age: Optional[int] = None
    DailyRate: Optional[int] = None
    Education: Optional[int] = None
    EnvironmentSatisfaction: Optional[int] = None
    HourlyRate: Optional[int] = None
    JobInvolvement: Optional[int] = None
    MonthlyRate: Optional[int] = None
    NumCompaniesWorked: Optional[int] = None
    PerformanceRating: Optional[int] = None
    RelationshipSatisfaction: Optional[int] = None
    StockOptionLevel: Optional[int] = None
    TotalWorkingYears: Optional[int] = None
    TrainingTimesLastYear: Optional[int] = None
    YearsInCurrentRole: Optional[int] = None
    YearsSinceLastPromotion: Optional[int] = None
    YearsWithCurrManager: Optional[int] = None
    BusinessTravel: Optional[str] = None
    Department: Optional[str] = None
    EducationField: Optional[str] = None
    Gender: Optional[str] = None
    MaritalStatus: Optional[str] = None
    JobRole: Optional[str] = None

class SimulationRequest(BaseModel):
    """What-If analizi için istek modeli"""
    employee: EmployeeData = Field(..., description="Orijinal çalışan verisi")
    changes: SimulationChanges = Field(..., description="Değiştirilecek parametreler")

class SimulationResponse(BaseModel):
    """What-If analizi için yanıt modeli"""
    original_risk: float = Field(..., description="Orijinal ayrılma riski (0-1)")
    new_risk: float = Field(..., description="Değişiklikler sonrası ayrılma riski (0-1)")
    risk_reduction: float = Field(..., description="Risk azalması (negatif ise risk artmış demektir)")
    original_prediction: str = Field(..., description="Orijinal tahmin (YES/NO)")
    new_prediction: str = Field(..., description="Yeni tahmin (YES/NO)")
    changes_applied: Dict[str, Any] = Field(..., description="Uygulanan değişiklikler")
    status: str = Field(default="Success", description="İşlem durumu")
    model_source: str = Field(..., description="Kullanılan model kaynağı")

@app.get("/")
def home():
    return {
        "message": "IBM HR Attrition Prediction API is Live!",
        "model_status": "production" if model is not None else "fallback",
        "monitoring": "enabled"
    }

@app.get("/monitoring/report")
def get_monitoring_report():
    """Monitoring raporunu al"""
    baseline_path = os.path.join(BASE_DIR, 'data', 'baseline_stats.json')
    report = monitor.get_monitoring_report(baseline_path)
    return report

@app.post("/predict")
def predict(data: EmployeeData):
    """
    Standart tahmin endpoint'i - tek bir çalışan için ayrılma riski tahmini.
    Geriye dönük uyumluluk için korunuyor.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model sunucuda yüklü değil.")
    
    if not encoders:
        raise HTTPException(status_code=500, detail="Encoder'lar yüklenemedi. Preprocessing çalıştırıldı mı?")

    try:
        input_data = data.dict()
        
        # Preprocessing
        df = prepare_features(input_data, encoders, feature_order, model)
        print(f"[INFO] Tahmin yapiliyor... Kolon sayisi: {len(df.columns)}")
        
        # Tahmin yap
        prediction_class, probability, model_source = predict_attrition(df, input_data, model)
        
        result = "YES (Will Leave)" if prediction_class == 1 else "NO (Will Stay)"
        
        # PDF ZORUNLULUĞU: Monitoring - Prediction'ı logla
        monitor.log_prediction(input_data, prediction_class, probability)
        
        return {
            "prediction": result,
            "confidence": f"{probability:.4f}",
            "status": "Success",
            "model_source": model_source
        }
    
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/predict/simulate", response_model=SimulationResponse)
def predict_simulate(request: SimulationRequest):
    """
    What-If Analizi Endpoint'i
    
    Orijinal çalışan verisi ile değişiklikler sonrası risk karşılaştırması yapar.
    Örnek kullanım:
    {
        "employee": { ... tüm çalışan verisi ... },
        "changes": {
            "OverTime": "No",
            "MonthlyIncome": 7000
        }
    }
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model sunucuda yüklü değil.")
    
    if not encoders:
        raise HTTPException(status_code=500, detail="Encoder'lar yüklenemedi. Preprocessing çalıştırıldı mı?")

    try:
        # 1. Orijinal veriyi hazırla
        original_data = request.employee.dict()
        original_df = prepare_features(original_data, encoders, feature_order, model)
        original_class, original_prob, original_source = predict_attrition(original_df, original_data, model)
        
        # 2. Değişiklikleri uygula
        modified_data = original_data.copy()
        changes_applied = {}
        
        changes_dict = request.changes.dict(exclude_none=True)
        for key, value in changes_dict.items():
            if key in modified_data:
                old_value = modified_data[key]
                modified_data[key] = value
                changes_applied[key] = {
                    "old": old_value,
                    "new": value
                }
        
        # 3. Değiştirilmiş veriyi hazırla ve tahmin yap
        modified_df = prepare_features(modified_data, encoders, feature_order, model)
        new_class, new_prob, new_source = predict_attrition(modified_df, modified_data, model)
        
        # 4. Delta hesapla (risk azalması/artması)
        risk_reduction = original_prob - new_prob  # Pozitif = risk azaldı, Negatif = risk arttı
        
        # 5. Sonuçları formatla
        original_prediction = "YES (Will Leave)" if original_class == 1 else "NO (Will Stay)"
        new_prediction = "YES (Will Leave)" if new_class == 1 else "NO (Will Stay)"
        
        # Model kaynağı (her iki tahmin için aynı olmalı, ama yine de kontrol edelim)
        model_source_used = original_source if original_source == new_source else "mixed"
        
        return SimulationResponse(
            original_risk=round(original_prob, 4),
            new_risk=round(new_prob, 4),
            risk_reduction=round(risk_reduction, 4),
            original_prediction=original_prediction,
            new_prediction=new_prediction,
            changes_applied=changes_applied,
            status="Success",
            model_source=model_source_used
        )
    
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)