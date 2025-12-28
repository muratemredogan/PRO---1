import streamlit as st
import requests
import json
import pandas as pd
import hashlib

# Sayfa Ayarları
st.set_page_config(
    page_title="IBM HR Attrition Analysis",
    page_icon="🤖",
    layout="wide"
)

# --- BAŞLIK ---
st.title("🤖 IBM HR Employee Attrition Prediction System")
st.markdown("""
**MLOps Pipeline Dashboard** | Model: XGBoost | Serving: FastAPI
""")
st.divider()

# --- SIDEBAR (KRİTİK GİRİŞLER) ---
st.sidebar.header("📋 Kritik Faktörler")

def user_input_features():
    # 1. EN ÖNEMLİ FAKTÖRLER (Modeli Tetikleyenler)
    st.sidebar.markdown("### 🎯 Kritik Kategorik Özellikler")
    
    over_time = st.sidebar.radio("Fazla Mesai Yapıyor mu? (OverTime)", ["No", "Yes"])
    
    marital = st.sidebar.selectbox("Medeni Durum (MaritalStatus)", ["Single", "Married", "Divorced"])
    
    business_travel = st.sidebar.selectbox("İş Seyahati (BusinessTravel)", 
                                           ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    
    department = st.sidebar.selectbox("Departman (Department)", 
                                     ["Sales", "Research & Development", "Human Resources"])
    
    education_field = st.sidebar.selectbox("Eğitim Alanı (EducationField)", 
                                          ["Life Sciences", "Medical", "Marketing", 
                                           "Technical Degree", "Other", "Human Resources"])
    
    gender = st.sidebar.radio("Cinsiyet (Gender)", ["Male", "Female"])
    
    job_role = st.sidebar.selectbox("İş Rolü (JobRole)",
                                   ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                    "Manufacturing Director", "Healthcare Representative", "Manager",
                                    "Sales Representative", "Research Director", "Human Resources"])

    # 2. Sayısal Değerler - Demografik & İş
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Demografik & İş")
    
    age = st.sidebar.slider("Yaş (Age)", 18, 60, 30)
    income = st.sidebar.slider("Aylık Gelir (MonthlyIncome)", 1000, 20000, 5000)
    distance = st.sidebar.slider("Evden Uzaklık (km)", 1, 30, 10)
    years_at_company = st.sidebar.slider("Şirketteki Yılı (YearsAtCompany)", 0, 40, 2)
    daily_rate = st.sidebar.slider("Günlük Ücret (DailyRate)", 100, 1500, 500)
    
    satisfaction = st.sidebar.select_slider("İş Memnuniyeti (JobSatisfaction) (1-4)", options=[1, 2, 3, 4], value=2)
    environment = st.sidebar.select_slider("Ortam Memnuniyeti (EnvironmentSatisfaction) (1-4)", options=[1, 2, 3, 4], value=2)
    
    # 3. Ek Sayısal Parametreler
    st.sidebar.markdown("---")
    st.sidebar.header("📈 Ek İş Detayları")
    
    education = st.sidebar.select_slider("Eğitim Seviyesi (Education) (1-5)", options=[1, 2, 3, 4, 5], value=2)
    hourly_rate = st.sidebar.slider("Saatlik Ücret (HourlyRate)", 30, 100, 50)
    job_involvement = st.sidebar.select_slider("İş Katılımı (JobInvolvement) (1-4)", options=[1, 2, 3, 4], value=2)
    job_level = st.sidebar.select_slider("İş Seviyesi (JobLevel) (1-5)", options=[1, 2, 3, 4, 5], value=1)
    monthly_rate = st.sidebar.slider("Aylık Ücret Oranı (MonthlyRate)", 2000, 27000, 10000)
    num_companies_worked = st.sidebar.slider("Çalışılan Şirket Sayısı (NumCompaniesWorked)", 0, 10, 5)
    percent_salary_hike = st.sidebar.slider("Maaş Artış Yüzdesi (PercentSalaryHike)", 11, 25, 15)
    performance_rating = st.sidebar.select_slider("Performans Değerlendirmesi (PerformanceRating) (1-4)", options=[1, 2, 3, 4], value=3)
    relationship_satisfaction = st.sidebar.select_slider("İlişki Memnuniyeti (RelationshipSatisfaction) (1-4)", options=[1, 2, 3, 4], value=2)
    stock_option_level = st.sidebar.select_slider("Hisse Senedi Seviyesi (StockOptionLevel) (0-3)", options=[0, 1, 2, 3], value=0)
    total_working_years = st.sidebar.slider("Toplam Çalışma Yılı (TotalWorkingYears)", 0, 40, 10)
    training_times_last_year = st.sidebar.slider("Geçen Yıl Eğitim Sayısı (TrainingTimesLastYear)", 0, 6, 2)
    work_life_balance = st.sidebar.select_slider("İş-Hayat Dengesi (WorkLifeBalance) (1-4)", options=[1, 2, 3, 4], value=2)
    years_in_current_role = st.sidebar.slider("Mevcut Roldeki Yılı (YearsInCurrentRole)", 0, 20, 1)
    years_since_last_promotion = st.sidebar.slider("Son Terfiden Beri Yıl (YearsSinceLastPromotion)", 0, 15, 2)
    years_with_curr_manager = st.sidebar.slider("Mevcut Yöneticiyle Yıl (YearsWithCurrManager)", 0, 20, 1)
    
    # API İÇİN VERİ PAKETİ - Tüm feature'ları dinamik olarak dahil et
    data = {
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance,
        "Education": education,
        "EnvironmentSatisfaction": environment,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": satisfaction,
        "MaritalStatus": marital,  # String olarak gönderilecek
        "MonthlyIncome": income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": over_time,  # String olarak gönderilecek
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
        # Yeni eklenen kategorik feature'lar
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role
    }
    return data

input_data = user_input_features()

# Otomatik tahmin seçeneği
auto_predict = st.sidebar.checkbox("🔄 Parametreler değiştiğinde otomatik tahmin yap", value=True)

# --- GÖRSELLEŞTİRME ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Profil Özeti")
    # Kritik verileri vurgula
    st.info(f"Fazla Mesai: {input_data['OverTime']}")
    st.info(f"Medeni Durum: {input_data['MaritalStatus']}")
    st.info(f"Departman: {input_data['Department']}")
    st.info(f"İş Rolü: {input_data['JobRole']}")
    st.json(input_data)

with col2:
    st.subheader("Tahmin Sonucu")
    
    # Parametrelerin hash'ini alarak değişip değişmediğini kontrol et
    current_params_hash = hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
    
    # Session state'te önceki hash'i kontrol et
    if 'last_params_hash' not in st.session_state:
        st.session_state.last_params_hash = None
        st.session_state.last_prediction = None
    
    # Parametreler değişti mi?
    params_changed = st.session_state.last_params_hash != current_params_hash
    
    # Otomatik tahmin veya buton kontrolü
    should_predict = False
    
    if auto_predict and params_changed:
        should_predict = True
    elif st.button("🚀 Analiz Et (Predict)", type="primary"):
        should_predict = True
    
    if should_predict:
        api_url = "http://127.0.0.1:8000/predict"
        
        try:
            with st.spinner("Model karar veriyor..."):
                response = requests.post(api_url, json=input_data)
                
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                # DÜZELTME: confidence artık her zaman ayrılma ihtimali (class 1 olasılığı)
                attrition_probability = float(result["confidence"])
                
                # Session state'e kaydet
                st.session_state.last_params_hash = current_params_hash
                st.session_state.last_prediction = {
                    "prediction": prediction,
                    "attrition_probability": attrition_probability
                }
                
                if "YES" in prediction:
                    st.error(f"🚨 TAHMİN: {prediction}")
                    st.progress(attrition_probability, text=f"Ayrılma İhtimali: %{attrition_probability*100:.1f}")
                    st.warning("⚠️ Bu çalışan yüksek risk grubunda! Fazla mesai ve düşük gelir tetikleyici olabilir.")
                else:
                    st.success(f"✅ TAHMİN: {prediction}")
                    # Kalma ihtimali = 1 - ayrılma ihtimali
                    stay_probability = 1 - attrition_probability
                    st.progress(attrition_probability, text=f"Ayrılma İhtimali: %{attrition_probability*100:.1f} | Kalma İhtimali: %{stay_probability*100:.1f}")
            else:
                st.error("API Hatası!")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Bağlantı Hatası: {e}")
            st.caption("Docker veya Uvicorn çalışıyor mu?")
    elif st.session_state.last_prediction is not None and not params_changed:
        # Parametreler değişmediyse önceki sonucu göster
        last_pred = st.session_state.last_prediction
        prediction = last_pred["prediction"]
        attrition_probability = last_pred["attrition_probability"]
        
        if "YES" in prediction:
            st.error(f"🚨 TAHMİN: {prediction}")
            st.progress(attrition_probability, text=f"Ayrılma İhtimali: %{attrition_probability*100:.1f}")
            st.warning("⚠️ Bu çalışan yüksek risk grubunda! Fazla mesai ve düşük gelir tetikleyici olabilir.")
        else:
            st.success(f"✅ TAHMİN: {prediction}")
            stay_probability = 1 - attrition_probability
            st.progress(attrition_probability, text=f"Ayrılma İhtimali: %{attrition_probability*100:.1f} | Kalma İhtimali: %{stay_probability*100:.1f}")
    else:
        st.info("👆 Parametreleri ayarlayın ve 'Analiz Et' butonuna tıklayın veya otomatik tahmin seçeneğini aktif edin.")

st.markdown("---")
st.caption("MLOps Pipeline Demo v2.0")