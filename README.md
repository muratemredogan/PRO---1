# MLOps Projesi - IBM HR Attrition Prediction

## Proje Özeti

Bu proje, MLOps Level 2 (CI/CD Pipeline Automation) gereksinimlerini karşılayan end-to-end bir Machine Learning sistemidir. IBM HR Employee Attrition veri seti üzerinde çalışan, production-ready bir ML pipeline'ı içerir.

## Proje Yönergelerine Göre Uygulanan Gereksinimler

### ✅ I. Zorunlu Araçlar ve Altyapı

#### 1. Experiment Tracking ve Model Governance
- **MLflow** kullanılıyor
  - Her parametre, metrik ve model versiyonu loglanıyor
  - **Model Registry** entegrasyonu yapıldı (staging/production yönetimi)
  - Model versiyonlama ve stage yönetimi aktif

#### 2. Workflow Orchestration
- **Prefect** seçildi ve implementasyonu yapıldı
  - `src/workflow.py` dosyasında pipeline workflow tanımlı
  - Preprocessing → Feature Validation → Training → CME adımları otomatik

#### 3. CI/CD Pipeline
- **GitLab CI/CD** pipeline dosyası (`.gitlab-ci.yml`)
- **Jenkins** pipeline dosyası (`Jenkinsfile`)
  - Commit Stage: Kod kalitesi kontrolü
  - Acceptance Test Stage: Model eğitimi ve validasyon
  - Deploy Stage: Docker build

#### 4. Containerization
- **Docker** desteği (`Dockerfile` mevcut)
- FastAPI servisi containerize edilebilir

### ✅ II. Teknik Implementasyon Gereksinimleri

#### 1. Data Representation (High-Cardinality Handling)
- ✅ **Hashed Feature Pattern**: `JobRole` için FeatureHasher kullanılıyor (8 bucket)
  - PDF ZORUNLULUĞU: High Cardinality için Hashed Features pattern implementasyonu
  - Trade-off: Bucket collision kabul ediliyor, ancak model boyutu sabit tutuluyor
  - `preprocess.py` ve `app.py` içinde implementasyonu yapıldı
- ✅ **Feature Cross**: `Age_DailyRate`, `MonthlyIncome_JobLevel`, `YearsAtCompany_TotalWorkingYears` feature interaction implementasyonu
- ✅ **Label Encoding**: Diğer kategorik feature'lar için (BusinessTravel, Department, EducationField, Gender, MaritalStatus, OverTime)

#### 2. Problem Representation ve Training
- ✅ **Problem Reframing**: `reframe_problem()` fonksiyonu eklendi (risk buckets stratejisi)
- ✅ **Ensembles**: XGBoost kullanılıyor
- ✅ **Rebalancing**: SMOTE ile class imbalance çözülüyor
- ✅ **Checkpoints**: Training sırasında checkpoint kaydediliyor (`data/checkpoints/`)

#### 3. Resilient Serving ve Continuous Evaluation
- ✅ **Stateless Serving Function**: FastAPI REST endpoint (`src/app.py`)
- ✅ **Continuous Model Evaluation (CME)**: `src/cme.py` modülü
  - Model degradation detection
  - Feature drift detection
  - Performance monitoring
- ✅ **ML-Specific Monitoring**: `src/monitoring.py` modülü
  - Prediction distribution tracking
  - Feature statistics monitoring
  - Prediction shift detection
  - Feature skew detection
- ✅ **Algorithmic Fallback**: Rule-based fallback modeli implementasyonu
  - Model başarısız olursa veya performans düşerse kullanılır

#### 4. Feature Validation
- ✅ **Great Expectations benzeri validation**: `src/feature_validation.py`
  - Numeric feature validation (mean shift, range check, null check)
  - Categorical feature validation (distribution shift, unknown categories)
  - Statistical checks

## Proje Yapısı

```
deneme1/
├── src/
│   ├── preprocess.py          # Veri ön işleme (Feature Cross, Hashing)
│   ├── train.py               # Model eğitimi (SMOTE, XGBoost, Checkpoints, Registry)
│   ├── app.py                 # FastAPI serving (Monitoring, Fallback)
│   ├── workflow.py            # Prefect pipeline
│   ├── cme.py                 # Continuous Model Evaluation
│   ├── monitoring.py          # ML-specific monitoring
│   └── feature_validation.py  # Feature validation
├── data/
│   ├── train.csv              # Eğitim verisi
│   ├── test.csv               # Test verisi
│   ├── encoders/              # Encoder'lar (LabelEncoder, FeatureHasher)
│   ├── checkpoints/           # Model checkpoints
│   ├── cme_results/           # CME sonuçları
│   ├── monitoring/            # Monitoring logları
│   └── validation/            # Validation sonuçları
├── mlruns/                    # MLflow experiment tracking
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Containerization
├── .gitlab-ci.yml            # GitLab CI/CD pipeline
├── Jenkinsfile                # Jenkins pipeline
└── test_pipeline.py          # Pipeline test scripti
```

## Kullanım

### 1. Preprocessing
```bash
python src/preprocess.py
```

### 2. Model Training
```bash
python src/train.py
```

### 3. Continuous Model Evaluation
```bash
python src/cme.py
```

### 4. Feature Validation
```bash
python -c "import sys; sys.path.insert(0, 'src'); from feature_validation import validate_features; import pandas as pd; df = pd.read_csv('data/test.csv'); validate_features(df)"
```

### 5. Prefect Pipeline (Tüm Adımlar)
```bash
python src/workflow.py
```

### 6. API Servisi
```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

### 7. Monitoring Dashboard
```bash
streamlit run src/dashboard.py
```

## Test

Tüm pipeline'ı test etmek için:
```bash
python test_pipeline.py
```

## Model Performansı

- **Accuracy**: ~0.85
- **F1 Score**: ~0.40
- **ROC-AUC**: ~0.74
- **Precision**: ~0.54
- **Recall**: ~0.32

## Önemli Notlar

1. **Model Registry**: İlk çalıştırmada model otomatik olarak Registry'ye kaydedilir (staging stage)
2. **Monitoring**: API her prediction'ı loglar ve MLflow'a metrikleri gönderir
3. **Fallback**: Model yüklenemezse veya hata olursa rule-based fallback kullanılır
4. **CI/CD**: GitLab CI/CD veya Jenkins pipeline'ları kullanılabilir

## Gereksinimler

Tüm gereksinimler `requirements.txt` dosyasında listelenmiştir:
- pandas, numpy, scikit-learn
- xgboost
- mlflow
- fastapi, uvicorn
- prefect
- great-expectations
- imbalanced-learn
- streamlit

## Sonuç

Proje, MLOps Level 2 gereksinimlerini karşılayan, production-ready bir ML sistemidir. Tüm zorunlu design pattern'ler ve tool'lar implementasyonu yapılmıştır.
