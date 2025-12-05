import cv2
import numpy as np
import os
import glob
from skimage.feature import hog
import random
from sklearn.utils import shuffle

# ==============================================================================
# 1. SABİT TANIMLAMALARI
# ==============================================================================

# HOG ve SVM için temel sabitler (Bu değerleri projenize göre ayarlayabilirsiniz)
MIN_WDW_SZ = (64, 128) # HOG pencere boyutu (genişlik, yükseklik)
PPC = (8, 8)           # pixels_per_cell
CPB = (2, 2)           # cells_per_block
ORIENTATIONS = 9       # Yönlendirme sayısı

# HOG Parametre Sözlüğü
HOG_PARAMS = {
    'orientations': ORIENTATIONS,
    'pixels_per_cell': PPC,
    'cells_per_block': CPB,
    'visualize': False, 
    'channel_axis': None # Gri tonlama (tek kanal) kullanıldığı için None
}

# Veri ve Model Yolları (Kendi klasör yollarınıza göre düzenleyin)
POS_IM_PATH = '/content/drive/MyDrive/data_dog/1'  # Pozitif örneklerin yolu (örneğin kedi resimleri)
NEG_IM_PATH = '/content/drive/MyDrive/data_dog/0'  # Negatif örneklerin yolu (örneğin arka plan resimleri)
MODEL_SAVE_PATH = '/content/drive/MyDrive/svm_model_dog.xml'

# ==============================================================================
# 2. EĞİTİM FONKSİYONU
# ==============================================================================

def train_svm():
    """HOG özelliklerini çıkarır, veriyi karıştırır ve OpenCV SVM modelini eğitip kaydeder."""
    samples = []
    labels = [] 
    
    img_extensions = ['*.png', '*.jpg', '*.jpeg']
    
    # --- Beklenen HOG Vektör Uzunluğunu Hesaplama ---
    # Bu, tüm vektörlerin aynı boyutta olması için bir referans noktasıdır.
    try:
        # Örnek bir 64x128 siyah görüntüden HOG vektör uzunluğunu hesapla
        temp_img = np.zeros(MIN_WDW_SZ, dtype=np.uint8)
        # HOG'u yalnızca bir kere hesaplayıp uzunluğunu alıyoruz
        expected_len = len(hog(temp_img, **HOG_PARAMS))
        print(f"[DEBUG] Beklenen HOG Vektör Uzunluğu: {expected_len}")
    except Exception as e:
        print(f"[HATA] HOG Parametreleri veya MIN_WDW_SZ hatalı. Eğitim durduruluyor. Hata: {e}")
        return False
        
    
    # --- Pozitif Örnekler İşleniyor (Etiket: 1) ---
    print("\nPozitif örnekler işleniyor...")
    for ext in img_extensions:
        for filename in glob.glob(os.path.join(POS_IM_PATH, ext)):
            img = cv2.imread(filename, 0) # Görüntüyü gri tonlamalı oku
            if img is not None:
                # Görüntüyü HOG pencere boyutuna yeniden boyutlandır
                img = cv2.resize(img, MIN_WDW_SZ)
                try:
                    hist = hog(img, **HOG_PARAMS)
                    
                    # KRİTİK KONTROL: Vektör uzunluğunu kontrol et
                    if len(hist) == expected_len:
                        samples.append(hist)
                        labels.append(1)
                    else:
                        print(f"[UYARI] Pozitif Dosya Hatalı: {os.path.basename(filename)} - Beklenmeyen HOG boyutu ({len(hist)}). Atlanıyor.")
                except Exception as e:
                    print(f"[UYARI] Pozitif İşleme Hatası: {os.path.basename(filename)} atlanıyor. Hata: {e}")


    # --- Negatif Örnekler İşleniyor (Etiket: 0) ---
    print("\nNegatif örnekler işleniyor...")
    for ext in img_extensions:
        for filename in glob.glob(os.path.join(NEG_IM_PATH, ext)):
            img = cv2.imread(filename, 0) # Görüntüyü gri tonlamalı oku
            if img is not None:
                # Görüntüyü HOG pencere boyutuna yeniden boyutlandır
                img = cv2.resize(img, MIN_WDW_SZ)
                try:
                    hist = hog(img, **HOG_PARAMS)
                    
                    # KRİTİK KONTROL: Vektör uzunluğunu kontrol et
                    if len(hist) == expected_len:
                        samples.append(hist)
                        labels.append(0)
                    else:
                        print(f"[UYARI] Negatif Dosya Hatalı: {os.path.basename(filename)} - Beklenmeyen HOG boyutu ({len(hist)}). Atlanıyor.")
                except Exception as e:
                    print(f"[UYARI] Negatif İşleme Hatası: {os.path.basename(filename)} atlanıyor. Hata: {e}")
            
    
    # --- Veri Ön Kontrolleri ---
    if len(samples) == 0:
        print("\n[HATA] Eğitim için hiçbir görüntü yüklenmedi. Yolları ve uzantıları kontrol edin.")
        return False

    # Sınıf sayısı kontrolü
    pos_count = labels.count(1)
    neg_count = labels.count(0)
    if pos_count == 0 or neg_count == 0:
        print("\n[HATA] SVM eğitimi için her iki sınıftan da (pozitif ve negatif) örneklere ihtiyaç vardır.")
        print(f"[DEBUG] Toplanan Pozitif Örnek: {pos_count}, Negatif Örnek: {neg_count}")
        return False

    # --- NumPy'a Çevirme ve Karıştırma ---
    print(f"\nToplam {len(samples)} örnek ile eğitime hazırlanılıyor.")
    
    # ValueError'a neden olan satır: Şimdi samples listesindeki tüm HOG vektörleri
    # aynı uzunlukta olduğu için hata oluşmayacak.
    samples = np.float32(samples)
    labels = np.array(labels)

    # Veriyi Karıştır (Genellikle daha iyi genelleme için yapılır)
    samples, labels = shuffle(samples, labels, random_state=42)
    
    # --- SVM Eğitimi ---
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC) # Sınıflandırma için C-SVC
    svm.setKernel(cv2.ml.SVM_RBF) # Radial Basis Function (RBF) çekirdeği
    svm.setGamma(5.383)           # Bu parametreler optimizasyonla bulunmuştur
    svm.setC(2.67) 

    print("SVM eğitimi başlatılıyor...")
    # cv2.ml.ROW_SAMPLE: Her satırın bir veri örneği olduğunu belirtir.
    svm.train(samples, cv2.ml.ROW_SAMPLE, labels) 
    print("Eğitim tamamlandı.")
    
    # Modeli Kaydet
    svm.save(MODEL_SAVE_PATH)
    print(f"\nModel başarıyla kaydedildi: {MODEL_SAVE_PATH}")
    return True
import numpy as np
import cv2
import os
import glob
from skimage.feature import hog
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ====================================================
# I. KONFİGÜRASYON VE PARAMETRELER
# ====================================================

# --- TEST YOLLARI ---
# Kaydedilmiş modelinizin yolu (Önceki kodda MODEL_SAVE_PATH'a eşittir)
MODEL_PATH = r'/content/drive/MyDrive/svm_model_dog.xml' 
# Test verilerinizin ana klasörü (altında 1 ve 0 klasörleri olmalı)
TEST_DATA_DIR = r'/content/drive/MyDrive/data_dog_test'

# --- HOG PARAMETRELERİ (Eğitimde kullanılanlarla AYNI olmalıdır) ---
MIN_WDW_SZ = (64, 128) # HOG pencere boyutu
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'visualize': False,
    'channel_axis': None 
    # Not: Eğitimde kullandığınız HOG_PARAMS sözlüğünü buraya tamamen aktarın.
} 

# ====================================================
# II. VERİ YÜKLEME VE ÖZELLİK ÇIKARMA
# ====================================================

def load_test_data(data_dir, target_size, hog_params):
    """Test verilerini yükler, HOG özelliklerini çıkarır ve etiketler."""
    test_features = []
    true_labels = []
    
    img_extensions = ['*.png', '*.jpg', '*.jpeg']
    
    # 1. Beklenen HOG Vektör Uzunluğunu Hesapla (Hata Kontrolü İçin)
    temp_img = np.zeros(target_size, dtype=np.uint8)
    expected_len = len(hog(temp_img, **hog_params))
    
    print(f"Test verileri işleniyor: {data_dir}")
    
    # data_dog_test/1 ve data_dog_test/0 alt klasörlerini tara
    for class_name in ['1', '0']: 
        class_path = os.path.join(data_dir, class_name)
        label_value = int(class_name) # 1 veya 0
        
        print(f"-> Sınıf {class_name} işleniyor (Etiket: {label_value})...")
        
        for ext in img_extensions:
            for filename in glob.glob(os.path.join(class_path, ext)):
                img = cv2.imread(filename, 0)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    try:
                        features = hog(img, **hog_params)
                        
                        if len(features) == expected_len:
                            test_features.append(features)
                            true_labels.append(label_value)
                        # else: Hatalı vektörler atlanmıştır.
                        
                    except Exception as e:
                        print(f"HOG çıkarma hatası ({os.path.basename(filename)}): {e}. Atlanıyor.")

    return np.float32(test_features), np.array(true_labels)


# ====================================================
# III. MODEL YÜKLEME VE TAHMİN
# ====================================================

def test_model(X_test, y_test, model_path):
    """Kaydedilmiş SVM modelini yükler ve tahmin yapar."""
    
    # 1. Modeli Yükle
    try:
        clf = cv2.ml.SVM_load(model_path)
    except Exception as e:
        print(f"\n[FATAL HATA] Model yüklenemedi: {model_path}. Yolu ve dosya adını kontrol edin.")
        print(f"Hata detayı: {e}")
        return
    
    # 2. Tahmin Yap
    print("\nTahminler başlatılıyor...")
    
    # predict metodu, Raw Output (Güven Skoru) döndürür.
    # StatModel_RAW_OUTPUT bayrağı ile karar fonksiyonu çıktısını alırız.
    _, pred_raw = clf.predict(X_test, flags=cv2.ml.StatModel_RAW_OUTPUT)
    
    # Raw output genellikle pozitif değerler için pozitif, negatif değerler için negatif işaretlidir.
    # Bizim etiketlerimiz 1 ve 0 olduğu için, skoru etiketlere dönüştürmeliyiz:
    # Skor > 0 (veya CONFIDENCE_THRESHOLD) ise 1 (pozitif sınıf), aksi takdirde 0.
    
    # Not: Eğitimde kullandığınız confidence threshold (0.5) yerine
    # basitçe işaret (sign) kontrolü yapalım, bu SVM'de daha yaygındır.
    # Yani Karar Sınırı (Decision Boundary) 0'dır.
    y_pred = np.where(pred_raw.flatten() > 0, 1, 0)
    
    # 3. Metrikleri Hesapla
    print("Metrikler hesaplanıyor...")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # 4. Sonuçları Yazdır
    print("\n==============================================")
    print("SVM Test Sonuçları (HOG Özellikleri):")
    print("==============================================")
    print(f"Toplam Test Örneği: {len(y_test)}")
    print(f" Doğruluk (Accuracy): {accuracy:.4f}")
    print(f" Hassasiyet (Precision): {precision:.4f}")
    print(f" Geri Çağırma (Recall): {recall:.4f}")
    print(f" F1-Skoru: {f1:.4f}")
    print("==============================================")


# ====================================================
# IV. ANA ÇALIŞTIRMA BLOĞU
# ====================================================

if __name__ == '__main__':
    # 1. Veriyi Yükle ve HOG Özelliklerini Çıkar
    X_test, y_test = load_test_data(TEST_DATA_DIR, MIN_WDW_SZ, HOG_PARAMS)
    
    if len(X_test) > 0:
        # 2. Modeli Test Et ve Metrikleri Göster
        test_model(X_test, y_test, MODEL_PATH)
    else:
        print("\n[UYARI] Test verisi bulunamadı veya işlenemedi. Lütfen klasör yollarını kontrol edin.")