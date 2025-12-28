import streamlit as st
import requests
import json
import pandas as pd

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

    # 2. Sayısal Değerler
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Demografik & İş")
    
    age = st.sidebar.slider("Yaş (Age)", 18, 60, 30)
    income = st.sidebar.slider("Aylık Gelir (MonthlyIncome)", 1000, 20000, 5000)
    distance = st.sidebar.slider("Evden Uzaklık (km)", 1, 30, 10)
    years_at_company = st.sidebar.slider("Şirketteki Yılı", 0, 40, 2)
    daily_rate = st.sidebar.slider("Günlük Ücret (DailyRate)", 100, 1500, 500)
    
    satisfaction = st.sidebar.select_slider("İş Memnuniyeti (1-4)", options=[1, 2, 3, 4], value=2)
    environment = st.sidebar.select_slider("Ortam Memnuniyeti (1-4)", options=[1, 2, 3, 4], value=2)
    
    # API İÇİN VERİ PAKETİ - Tüm feature'ları dahil et
    data = {
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance,
        "Education": 2,
        "EnvironmentSatisfaction": environment,
        "HourlyRate": 50,
        "JobInvolvement": 2,
        "JobLevel": 1,
        "JobSatisfaction": satisfaction,
        "MaritalStatus": marital,  # String olarak gönderilecek
        "MonthlyIncome": income,
        "MonthlyRate": 10000,
        "NumCompaniesWorked": 5,
        "OverTime": over_time,  # String olarak gönderilecek
        "PercentSalaryHike": 10,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StockOptionLevel": 0,
        "TotalWorkingYears": years_at_company + 2,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 2,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 1,
        # Yeni eklenen kategorik feature'lar
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role
    }
    return data

input_data = user_input_features()

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
    
    if st.button("🚀 Analiz Et (Predict)", type="primary"):
        api_url = "http://127.0.0.1:8000/predict"
        
        try:
            with st.spinner("Model karar veriyor..."):
                response = requests.post(api_url, json=input_data)
                
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                # DÜZELTME: confidence artık her zaman ayrılma ihtimali (class 1 olasılığı)
                attrition_probability = float(result["confidence"])
                
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

st.markdown("---")
st.caption("MLOps Pipeline Demo v2.0")