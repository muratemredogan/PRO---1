# 1. Base Image: Python 3.9 kullanıyoruz (Hafif sürüm)
FROM python:3.9-slim

# 2. Çalışma dizinini ayarla
WORKDIR /app

# 3. Gereklilikleri kopyala ve yükle
# Önce sadece requirements.txt'yi kopyalıyoruz ki cache kullansın
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Proje kodlarını ve modelleri kopyala
# . (nokta) şu anki dizindeki her şeyi /app içine atar
COPY . .

# 5. MLflow için çevre değişkeni (Opsiyonel ama güvenli)
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# 6. Portu dışarı aç (FastAPI default portu)
EXPOSE 8000

# 7. Başlatma komutu (API'yi ayağa kaldır)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]