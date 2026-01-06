import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ayarlar
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'correlation_heatmap.png')
OUTPUT_ATTRITION_PATH = os.path.join(BASE_DIR, 'data', 'attrition_correlation.png')

# Veriyi yükle
print("[INFO] Veri yükleniyor...")
df = pd.read_csv(TRAIN_PATH)

# Attrition kolonunu çıkar
feature_cols = [col for col in df.columns if col != 'Attrition']
df_features = df[feature_cols]

# Sayısal kolonları al
numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = df_features[numeric_cols]

print(f"[INFO] {len(numeric_cols)} sayısal değişken bulundu.")

# 1. TÜM DEĞİŞKENLER ARASI KORELASYON
print("[INFO] Tüm değişkenler arası korelasyon matrisi hesaplanıyor...")
correlation_matrix = df_numeric.corr()

# Figür boyutu
plt.figure(figsize=(22, 18))

# Üst üçgeni gizle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Heatmap
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=False,
    cmap='RdBu_r',
    center=0,
    square=True,
    linewidths=0.3,
    cbar_kws={"shrink": 0.8, "label": "Korelasyon Katsayısı"},
    fmt='.2f'
)

plt.title('Tüm Değişkenler Arası Korelasyon Isı Haritası\n(Employee Attrition Model - 45 Değişken)', 
          fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"[OK] Isı haritası kaydedildi: {OUTPUT_PATH}")

# 2. ATTRITION İLE KORELASYON (Özel görselleştirme)
if 'Attrition' in df.columns:
    print("\n[INFO] Attrition ile korelasyon analizi...")
    
    # Attrition ile korelasyonları hesapla
    all_numeric = df[numeric_cols + ['Attrition']]
    attrition_corr = all_numeric.corr()['Attrition'].drop('Attrition').sort_values(ascending=False)
    
    # En yüksek ve en düşük 15 değişkeni al
    top_features = pd.concat([attrition_corr.head(15), attrition_corr.tail(15)])
    
    # Görselleştir
    plt.figure(figsize=(10, 12))
    colors = ['red' if x > 0 else 'blue' for x in top_features.values]
    
    plt.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Korelasyon Katsayısı', fontsize=12)
    plt.title('Attrition ile En Yüksek/Düşük Korelasyona Sahip Değişkenler', 
              fontsize=14, fontweight='bold', pad=15)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_ATTRITION_PATH, dpi=300, bbox_inches='tight')
    print(f"[OK] Attrition korelasyon grafiği kaydedildi: {OUTPUT_ATTRITION_PATH}")
    
    # Konsola yazdır
    print("\n" + "="*60)
    print("ATTRITION İLE EN YÜKSEK POZİTİF KORELASYON:")
    print("="*60)
    for idx, (feature, corr) in enumerate(attrition_corr.head(10).items(), 1):
        print(f"{idx:2d}. {feature:35s} : {corr:+.4f}")
    
    print("\n" + "="*60)
    print("ATTRITION İLE EN YÜKSEK NEGATİF KORELASYON:")
    print("="*60)
    for idx, (feature, corr) in enumerate(attrition_corr.tail(10).items(), 1):
        print(f"{idx:2d}. {feature:35s} : {corr:+.4f}")

print("\n[OK] Tüm analizler tamamlandı!")

