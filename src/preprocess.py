import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher
import joblib
import os

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
OUTPUT_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
OUTPUT_TEST_PATH = os.path.join(BASE_DIR, 'data', 'test.csv')
ENCODERS_DIR = os.path.join(BASE_DIR, 'data', 'encoders')

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Ham veri yuklendi. Boyut: {df.shape}")
    return df

def feature_engineering(df):
    # 1. Target Mapping (Attrition: Yes/No -> 1/0)
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # 2. PDF ZORUNLULUĞU[cite: 32]: Feature Interactions (Feature Cross)
    # Yaş ve Günlük Ücret etkileşimi
    df['Age_DailyRate'] = df['Age'] * df['DailyRate']
    
    # 3. Model'in beklediği ek feature'ları oluştur
    # MonthlyIncome ve JobLevel etkileşimi
    if 'MonthlyIncome' in df.columns and 'JobLevel' in df.columns:
        df['MonthlyIncome_JobLevel'] = df['MonthlyIncome'] * df['JobLevel']
    
    # YearsAtCompany ve TotalWorkingYears etkileşimi
    if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
        df['YearsAtCompany_TotalWorkingYears'] = df['YearsAtCompany'] * df['TotalWorkingYears']
    
    # Age binning (0-30: 0, 31-40: 1, 41-50: 2, 51+: 3)
    if 'Age' in df.columns:
        df['Age_Binned'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=[0, 1, 2, 3]).astype(int)
    
    # MonthlyIncome binning (düşük, orta, yüksek)
    if 'MonthlyIncome' in df.columns:
        income_median = df['MonthlyIncome'].median()
        income_q75 = df['MonthlyIncome'].quantile(0.75)
        df['MonthlyIncome_Binned'] = pd.cut(
            df['MonthlyIncome'], 
            bins=[0, income_median, income_q75, float('inf')], 
            labels=[0, 1, 2]
        ).astype(int)
    
    # Income to Level Ratio
    if 'MonthlyIncome' in df.columns and 'JobLevel' in df.columns:
        df['Income_to_Level_Ratio'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)  # +1 to avoid division by zero
    
    # Company Loyalty (YearsAtCompany / TotalWorkingYears)
    if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
        df['Company_Loyalty'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)  # +1 to avoid division by zero
    
    # WorkLife_OverTime_Stress (WorkLifeBalance * OverTime)
    if 'WorkLifeBalance' in df.columns and 'OverTime' in df.columns:
        # OverTime'ı sayısal yap (Yes=1, No=0)
        overtime_numeric = (df['OverTime'] == 'Yes').astype(int)
        df['WorkLife_OverTime_Stress'] = df['WorkLifeBalance'] * overtime_numeric
    
    # 4. Gereksiz kolonları at
    drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    print("[OK] Feature Cross ve Temizlik tamamlandi.")
    return df

def handle_categorical(df):
    # Encoder'ları saklamak için dictionary
    encoders = {}
    
    # Encoder klasörünü oluştur
    os.makedirs(ENCODERS_DIR, exist_ok=True)
    
    # Kategorik kolonları ayır
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # PDF ZORUNLULUĞU: High Cardinality -> HASHED FEATURE Pattern
    # JobRole için FeatureHasher kullanıyoruz (yüksek cardinality için)
    # Trade-off: Bucket collision kabul ediyoruz, ancak model boyutunu sabit tutuyoruz
    if 'JobRole' in cat_cols:
        print("[INFO] PDF ZORUNLULUĞU: JobRole için Hashed Features pattern uygulanıyor...")
        # FeatureHasher: n_features=8 (8 bucket'a hash'liyoruz)
        # Bu, yüksek cardinality'yi sabit boyutlu feature'lara dönüştürür
        hasher = FeatureHasher(n_features=8, input_type='string')
        
        # JobRole değerlerini string listesi olarak hazırla
        jobrole_values = df['JobRole'].astype(str).values.reshape(-1, 1)
        
        # Hash'le ve sparse matrix'i dense'e çevir
        jobrole_hashed = hasher.transform(jobrole_values).toarray()
        
        # Hash'lenmiş feature'ları DataFrame'e ekle (JobRole_Hash_0, JobRole_Hash_1, ...)
        for i in range(8):
            df[f'JobRole_Hash_{i}'] = jobrole_hashed[:, i]
        
        # Orijinal JobRole kolonunu sil (artık hash'lenmiş versiyonlar var)
        df = df.drop(columns=['JobRole'])
        cat_cols.remove('JobRole')
        
        # Hasher'ı kaydet (inference için)
        hasher_path = os.path.join(ENCODERS_DIR, 'jobrole_hasher.pkl')
        joblib.dump(hasher, hasher_path)
        encoders['JobRole'] = hasher
        print(f"[OK] JobRole FeatureHasher kaydedildi (8 features).")

    # Geri kalan standart kategorikler için Label Encoding
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        # Her kolon için ayrı encoder kaydet
        encoder_path = os.path.join(ENCODERS_DIR, f'{col}_encoder.pkl')
        joblib.dump(le, encoder_path)
        encoders[col] = le
        print(f"[OK] {col} LabelEncoder kaydedildi.")
        
    print(f"[OK] Kategorik donusumler tamamlandi. Toplam {len(encoders)} encoder kaydedildi.")
    return df, encoders

def main():
    print("--- Preprocessing Başlıyor ---")
    df = load_data()
    df = feature_engineering(df)
    df, encoders = handle_categorical(df)
    
    # Train/Test Split (Stratified)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Attrition'])
    
    # Ayrı ayrı kaydet (MLflow aşamasında kolaylık olsun diye)
    train.to_csv(OUTPUT_TRAIN_PATH, index=False)
    test.to_csv(OUTPUT_TEST_PATH, index=False)
    
    print(f"[OK] Veri islendi ve ayrıstirildi:")
    print(f"   Train: {OUTPUT_TRAIN_PATH} ({train.shape})")
    print(f"   Test:  {OUTPUT_TEST_PATH} ({test.shape})")
    print(f"   Encoders: {ENCODERS_DIR}")

if __name__ == "__main__":
    main()