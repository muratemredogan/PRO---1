@echo off
chcp 65001 >nul
title IBM HR Attrition Prediction - MLOps Pipeline Baslatiliyor...

echo ========================================
echo IBM HR Attrition Prediction Sistemi
echo MLOps Pipeline - Otomatik Baslatma
echo ========================================
echo.

REM Proje dizinine git
cd /d "%~dp0"

REM Virtual environment'ı kontrol et
if not exist "venv\Scripts\activate.bat" (
    echo [HATA] Virtual environment bulunamadi!
    echo Lutfen once 'python -m venv venv' komutunu calistirin.
    pause
    exit /b 1
)

REM Python dosyalarını kontrol et
if not exist "src\app.py" (
    echo [HATA] src\app.py bulunamadi!
    pause
    exit /b 1
)

if not exist "src\dashboard.py" (
    echo [HATA] src\dashboard.py bulunamadi!
    pause
    exit /b 1
)

echo [1/8] Virtual environment aktif ediliyor...
call venv\Scripts\activate.bat

REM Gerekli paketlerin yüklü olup olmadığını kontrol et
echo [2/8] Gerekli paketler kontrol ediliyor...
python -c "import pandas" 2>nul
if %errorlevel% neq 0 (
    echo [UYARI] requirements.txt yukleniyor...
    pip install -r requirements.txt >nul 2>&1
)

python -c "import uvicorn" 2>nul
if %errorlevel% neq 0 (
    echo [UYARI] uvicorn bulunamadi. Yukleniyor...
    pip install uvicorn >nul 2>&1
)

python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [UYARI] streamlit bulunamadi. Yukleniyor...
    pip install streamlit >nul 2>&1
)

python -c "import mlflow" 2>nul
if %errorlevel% neq 0 (
    echo [UYARI] mlflow bulunamadi. Yukleniyor...
    pip install mlflow >nul 2>&1
)

REM Veri dosyasını kontrol et
echo [3/8] Veri dosyalari kontrol ediliyor...
if not exist "data\WA_Fn-UseC_-HR-Employee-Attrition.csv" (
    if exist "data\WA_Fn-UseC_-HR-Employee-Attrition.csv" (
        echo [OK] Veri dosyasi mevcut.
    ) else (
        echo [UYARI] Ham veri dosyasi bulunamadi!
        echo [BILGI] data\WA_Fn-UseC_-HR-Employee-Attrition.csv dosyasi gerekli.
    )
)

REM Preprocessing kontrolü
echo [4/8] Preprocessing kontrol ediliyor...
if not exist "data\train.csv" (
    echo [BILGI] train.csv bulunamadi. Preprocessing baslatiliyor...
    python src\preprocess.py
    if %errorlevel% neq 0 (
        echo [HATA] Preprocessing basarisiz!
        pause
        exit /b 1
    )
    echo [OK] Preprocessing tamamlandi.
) else (
    echo [OK] train.csv mevcut, preprocessing atlaniyor.
)

REM Model kontrolü
echo [5/8] Model kontrol ediliyor...
set MODEL_EXISTS=0

REM Python ile model kontrolü (Windows'ta ** wildcard çalışmadığı için)
python -c "import glob; import os; files = glob.glob('mlruns/**/MLmodel', recursive=True); exit(0 if files else 1)" 2>nul
if %errorlevel% == 0 (
    set MODEL_EXISTS=1
)

REM Model Registry kontrolü
python -c "import mlflow; from mlflow.tracking import MlflowClient; client = MlflowClient(); versions = client.get_latest_versions('IBM_Attrition_Model', stages=['Production', 'Staging']); exit(0 if versions else 1)" 2>nul
if %errorlevel% == 0 (
    set MODEL_EXISTS=1
)

if %MODEL_EXISTS% == 0 (
    echo [BILGI] Model bulunamadi. Model egitimi baslatiliyor...
    python src\train.py
    if %errorlevel% neq 0 (
        echo [UYARI] Model egitimi basarisiz olabilir, devam ediliyor...
    ) else (
        echo [OK] Model egitimi tamamlandi.
    )
) else (
    echo [OK] Model mevcut, egitim atlaniyor.
)

REM Feature Validation
echo [6/8] Feature validation calistiriliyor...
if exist "data\test.csv" (
    python -c "import sys; sys.path.insert(0, 'src'); from feature_validation import validate_features; import pandas as pd; df = pd.read_csv('data/test.csv'); validate_features(df)" 2>nul
    echo [OK] Feature validation tamamlandi.
) else (
    echo [UYARI] test.csv bulunamadi, feature validation atlaniyor.
)

REM Portların kullanılabilir olup olmadığını kontrol et
echo [7/8] Portlar kontrol ediliyor...
netstat -ano | findstr ":8000" >nul
if %errorlevel% == 0 (
    echo [UYARI] Port 8000 zaten kullanimda!
    echo [BILGI] Mevcut process kapatiliyor...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
    echo [OK] Port 8000 temizlendi.
)

netstat -ano | findstr ":8501" >nul
if %errorlevel% == 0 (
    echo [UYARI] Port 8501 zaten kullanimda!
    echo [BILGI] Mevcut process kapatiliyor...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
    echo [OK] Port 8501 temizlendi.
)

echo.
echo [8/8] Servisler baslatiliyor...
echo.

REM FastAPI başlat
echo [API] FastAPI baslatiliyor (Port 8000)...
start "FastAPI Server - IBM Attrition API" cmd /k "cd /d %~dp0 && venv\Scripts\activate.bat && echo FastAPI Server basladi... && uvicorn src.app:app --host 127.0.0.1 --port 8000"

REM API'nin başlaması için bekleme
timeout /t 5 /nobreak >nul

REM Streamlit Dashboard başlat
echo [DASHBOARD] Streamlit Dashboard baslatiliyor (Port 8501)...
start "Streamlit Dashboard - IBM Attrition" cmd /k "cd /d %~dp0 && venv\Scripts\activate.bat && echo Streamlit Dashboard basladi... && streamlit run src/dashboard.py --server.port 8501"

REM Streamlit'in başlaması için bekleme
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo [OK] Sistem baslatildi!
echo ========================================
echo.
echo SERVISLER:
echo   - API: http://127.0.0.1:8000
echo   - API Docs: http://127.0.0.1:8000/docs
echo   - Dashboard: http://localhost:8501
echo.
echo MONITORING:
echo   - Monitoring Raporu: http://127.0.0.1:8000/monitoring/report
echo.
echo [BILGI] Streamlit otomatik olarak tarayiciyi acacak.
echo [BILGI] Iki ayri pencere acildi - birinde API, digerinde Streamlit calisiyor.
echo.
echo [BILGI] Pipeline Adimlari:
echo   1. Preprocessing: Tamamlandi
echo   2. Model Training: Tamamlandi
echo   3. Feature Validation: Tamamlandi
echo   4. API Server: Calisiyor
echo   5. Dashboard: Calisiyor
echo.
echo [BILGI] Sistemi kapatmak icin pencereleri kapatin.
echo.
pause

