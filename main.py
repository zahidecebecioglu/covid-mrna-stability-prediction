"""
COVID-19 mRNA Aşı Stabilite Tahmini

Bu script, mRNA aşılarının stabilite özelliklerini tahmin etmek için geliştirilmiş bir makine öğrenmesi modelini içerir.
Veri analizi, özellik mühendisliği, model eğitimi ve tahmin aşamalarını gerçekleştirir.

Yazar: [İsminiz]
Tarih: [Tarih]
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')

def load_json_data(file_path):
    """
    JSON Lines formatındaki veriyi yükler.
    
    Args:
        file_path (str): JSON dosyasının yolu
        
    Returns:
        list: Yüklenen veri listesi
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Veri yükleme ve analiz
print("Eğitim verisi yükleniyor...")
train_data = load_json_data('train.json')
print("Test verisi yükleniyor...")
test_data = load_json_data('test.json')

# Veri analizi
print("\n=== VERİ ANALİZİ ===")
print(f"Eğitim seti boyutu: {len(train_data)}")
print(f"Test seti boyutu: {len(test_data)}")

# İlk örneğin yapısını incele
print("\nÖrnek veri yapısı:")
for key, value in train_data[0].items():
    if isinstance(value, (list, np.ndarray)):
        print(f"{key}: {type(value)}, uzunluk: {len(value)}")
    else:
        print(f"{key}: {type(value)}, değer: {value}")

# Hedef değişkenlerin analizi
target_columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
print("\n=== HEDEF DEĞİŞKENLERİN ANALİZİ ===")

# Her hedef için detaylı istatistikler
for col in target_columns:
    values = []
    for item in train_data:
        if col in item:
            values.extend(item[col])
    values = np.array(values)
    
    print(f"\n{col} için istatistikler:")
    print(f"Ortalama: {np.mean(values):.4f}")
    print(f"Medyan: {np.median(values):.4f}")
    print(f"Standart Sapma: {np.std(values):.4f}")
    print(f"Min: {np.min(values):.4f}")
    print(f"Max: {np.max(values):.4f}")
    print(f"Çarpıklık: {pd.Series(values).skew():.4f}")
    print(f"Basıklık: {pd.Series(values).kurtosis():.4f}")
    
    # Aykırı değer analizi
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = values[(values < lower_bound) | (values > upper_bound)]
    print(f"Aykırı değer sayısı: {len(outliers)}")
    print(f"Aykırı değer yüzdesi: {(len(outliers)/len(values))*100:.2f}%")

# Korelasyon analizi
print("\n=== KORELASYON ANALİZİ ===")
correlation_data = []
for item in train_data:
    row = {}
    for col in target_columns:
        if col in item:
            row[col] = np.mean(item[col])
    correlation_data.append(row)

correlation_df = pd.DataFrame(correlation_data)
correlation_matrix = correlation_df.corr()
print("\nHedef değişkenler arasındaki korelasyon:")
print(correlation_matrix)

# Korelasyon matrisini görselleştir
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Hedef Değişkenler Arasındaki Korelasyon')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Özellik mühendisliği
print("\n=== ÖZELLİK MÜHENDİSLİĞİ ===")

def extract_features(data, is_train=True):
    """
    RNA dizisi ve yapısından özellikler çıkarır.
    
    Args:
        data (list): Ham veri listesi
        is_train (bool): Eğitim verisi mi test verisi mi
        
    Returns:
        pandas.DataFrame: Çıkarılan özellikler
    """
    features = []
    for item in tqdm(data, desc="Özellik çıkarılıyor"):
        feature_dict = {
            'id': item['id'],
            'seq_length': item['seq_length'],
            'seq_scored': item['seq_scored']
        }
        
        if is_train:
            feature_dict['signal_to_noise'] = item['signal_to_noise']
            feature_dict['SN_filter'] = item['SN_filter']
        
        # RNA dizisi özellikleri
        sequence = item['sequence']
        feature_dict['A_count'] = sequence.count('A')
        feature_dict['U_count'] = sequence.count('U')
        feature_dict['G_count'] = sequence.count('G')
        feature_dict['C_count'] = sequence.count('C')
        
        # Nükleotid oranları
        total = len(sequence)
        feature_dict['A_ratio'] = feature_dict['A_count'] / total
        feature_dict['U_ratio'] = feature_dict['U_count'] / total
        feature_dict['G_ratio'] = feature_dict['G_count'] / total
        feature_dict['C_ratio'] = feature_dict['C_count'] / total
        
        # Yapı özellikleri
        structure = item['structure']
        feature_dict['dot_count'] = structure.count('.')
        feature_dict['bracket_count'] = structure.count('(') + structure.count(')')
        feature_dict['dot_ratio'] = feature_dict['dot_count'] / len(structure)
        feature_dict['bracket_ratio'] = feature_dict['bracket_count'] / len(structure)
        
        features.append(feature_dict)
    return pd.DataFrame(features)

# Eğitim ve test verileri için özellik çıkar
print("\nEğitim verisi için özellikler çıkarılıyor...")
train_features = extract_features(train_data, is_train=True)
print("\nTest verisi için özellikler çıkarılıyor...")
test_features = extract_features(test_data, is_train=False)

# Özellik istatistiklerini göster
print("\nÖzellik istatistikleri:")
print(train_features.describe())

# Özellik korelasyonlarını görselleştir
numeric_cols = train_features.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(15, 12))
sns.heatmap(train_features[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Özellikler Arasındaki Korelasyon')
plt.tight_layout()
plt.savefig('feature_correlations.png')
plt.close()

print("\nAnaliz tamamlandı. Görselleştirmeler kaydedildi.")

# Basit model: Her hedef için eğitimdeki ortalamayı tahmin olarak kullan
mean_targets = {}
for col in target_columns:
    values = []
    for item in train_data:
        if col in item:
            values.extend(item[col])
    mean_targets[col] = float(np.mean(values))

print("\nBasit model: Ortalama değerler ile tahmin yapılıyor...")

# Submission dosyası için satırları oluştur
submission_rows = []
for sample in test_data:
    sample_id = sample['id']
    seq_length = sample['seq_length']
    for seqpos in range(seq_length):
        row = {
            'id_seqpos': f'{sample_id}_{seqpos}',
            'reactivity': mean_targets['reactivity'],
            'deg_Mg_pH10': mean_targets['deg_Mg_pH10'],
            'deg_pH10': mean_targets['deg_pH10'],
            'deg_Mg_50C': mean_targets['deg_Mg_50C'],
            'deg_50C': mean_targets['deg_50C'],
        }
        submission_rows.append(row)

# DataFrame'e çevir ve kaydet
submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission dosyası 'submission.csv' olarak kaydedildi.")

def mcrmse(y_true, y_pred):
    """
    Mean Columnwise Root Mean Squared Error hesaplar.
    
    Args:
        y_true (numpy.ndarray): Gerçek değerler
        y_pred (numpy.ndarray): Tahmin edilen değerler
        
    Returns:
        tuple: (Ortalama MCRMSE, Her sütun için RMSE listesi)
    """
    rmses = []
    for i in range(y_true.shape[1]):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        rmses.append(rmse)
    return np.mean(rmses), rmses

# Ortak sayısal sütunları belirle
common_numeric_cols = [col for col in train_features.columns if col in test_features.columns and train_features[col].dtype != 'O']

# Eğitimde kullanılacak sütunlar
train_feature_cols = common_numeric_cols + ['seqpos']

# --- POZİSYON BAZINDA VERİYİ HAZIRLA ---
print("\n=== POZİSYON BAZINDA VERİ HAZIRLANIYOR ===")

# Sadece skorlanan hedefler
scored_targets = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']

X_rows = []
y_rows = {target: [] for target in scored_targets}

for item in train_data:
    feats = train_features[train_features['id'] == item['id']].iloc[0]
    for i in range(item['seq_scored']):
        row = feats[common_numeric_cols].copy()
        row['seqpos'] = i
        X_rows.append(row)
        for target in scored_targets:
            y_rows[target].append(item[target][i])

X = pd.DataFrame(X_rows).reset_index(drop=True)
Y = np.stack([y_rows[target] for target in scored_targets], axis=1)

print(f"Toplam pozisyon sayısı: {len(X)}")

# --- EĞİTİM/VALIDASYON AYRIMI ---
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- MODEL EĞİTİMİ ve TAHMİN ---
print("\n=== XGBOOST REGRESYON MODELİ EĞİTİLİYOR ===")
models = {}
y_val_preds = np.zeros_like(y_val)
for i, target in enumerate(scored_targets):
    print(f"\nModel eğitiliyor: {target}")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train[:, i])
    y_pred = model.predict(X_val)
    y_val_preds[:, i] = y_pred
    rmse = np.sqrt(mean_squared_error(y_val[:, i], y_pred))
    print(f"{target} için RMSE: {rmse:.4f}")
    models[target] = model

# --- MCRMSE HESAPLAMA ---
mcrmse_score, rmse_list = mcrmse(y_val, y_val_preds)
print(f"\nMCRMSE (Validation): {mcrmse_score:.4f}")
for i, target in enumerate(scored_targets):
    print(f"{target} için RMSE: {rmse_list[i]:.4f}")

# --- SONUÇLARIN RAPORLANMASI ---
with open('model_report.txt', 'w') as f:
    f.write(f"MCRMSE (Validation): {mcrmse_score:.4f}\n")
    for i, target in enumerate(scored_targets):
        f.write(f"{target} için RMSE: {rmse_list[i]:.4f}\n")
print("\nModel raporu 'model_report.txt' olarak kaydedildi.")

# --- TEST SETİ İÇİN TAHMİN VE SUBMISSION DOSYASI ---
print("\n=== TEST SETİ İÇİN TAHMİN VE SUBMISSION OLUŞTURULUYOR ===")

deg_pH10_mean = np.mean([v for item in train_data for v in item['deg_pH10']])
deg_50C_mean = np.mean([v for item in train_data for v in item['deg_50C']])

test_submission_rows = []
for idx, item in enumerate(test_data):
    feats = test_features.iloc[idx][common_numeric_cols]
    seq_length = item['seq_length']
    for seqpos in range(seq_length):
        row_feats = feats.copy()
        row_feats['seqpos'] = seqpos
        X_row = row_feats.values.reshape(1, -1)
        preds = []
        for i, target in enumerate(scored_targets):
            preds.append(models[target].predict(X_row)[0])
        preds.append(deg_pH10_mean)
        preds.append(deg_50C_mean)
        test_submission_rows.append({
            'id_seqpos': f"{item['id']}_{seqpos}",
            'reactivity': preds[0],
            'deg_Mg_pH10': preds[1],
            'deg_pH10': preds[3],
            'deg_Mg_50C': preds[2],
            'deg_50C': preds[4],
        })

submission_df = pd.DataFrame(test_submission_rows)
submission_df.to_csv('submission_xgb.csv', index=False)
print("Submission dosyası 'submission_xgb.csv' olarak kaydedildi.") 