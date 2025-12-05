### sınıflandırma karşılaştırma

import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#parametre ve veri yolu ayarları
DATA_DIR = '/content/drive/MyDrive/data_dog'
TARGET_SIZE = (64,128) #hog pencere boyutu
HOG_PARAMS = {
    'orientations':9,
    'pixels_per_cell':(8,8),
    'cells_per_block':(2,2),
    'transform_sqrt' :True,
    'feature_vector':True
}

#özellik çıkarımı ve veri yükleme
def extract_hog_features(image_path, target_size, hog_params):
    """Görüntüyü yükler, işler ve HOG özelliklerini çıkarır."""
    try:
        # Hata kontrolü ve griye çevirme
        image = imread(image_path)
        if image.ndim == 3:
            image = rgb2gray(image)
            
        resized_image = resize(image, target_size, anti_aliasing=True)
        features = hog(resized_image, **hog_params)
        return features
        
    except Exception as e:
        print(f"Hata oluştu: {image_path} -> {e}")
        return None

def load_data_simple(data_dir):
    """
    Klasör adlarına göre etiketleme yaparak tüm görüntüleri yükler ve HOG özelliklerini çıkarır.
    (En kolay yöntem, 100 görsel için idealdir.)
    """
    all_features = []
    all_labels = []
    
    # 1. Sınıf İsimlerini ve Etiket Haritasını Oluştur
    class_names = sorted(os.listdir(data_dir))
    label_map = {name: idx for idx, name in enumerate(class_names) if os.path.isdir(os.path.join(data_dir, name))}
    
    # 2. Klasörleri Tara
    for class_name, label_idx in label_map.items():
        class_path = os.path.join(data_dir, class_name)
        print(f"Sınıf işleniyor: {class_name} (Etiket: {label_idx})")
        
        # 3. Resim Dosyalarını Bul
        for file_name in os.listdir(class_path):
            if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                file_path = os.path.join(class_path, file_name)
                
                features = extract_hog_features(file_path, TARGET_SIZE, HOG_PARAMS)
                
                if features is not None:
                    all_features.append(features)
                    all_labels.append(label_idx)
                        
    return np.array(all_features), np.array(all_labels)

# ----------------------------------------------------
# 4. ANA ÇALIŞTIRMA BLOĞU (load_data_simple'ı çağırın)
# ----------------------------------------------------


#sınıflandırma ve karşılaştırma
def classify_and_compare(X,y):
  """farklı sınıflandırıcılar karşılaştırılır"""
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

  classifiers = {
      "Linear SVM" : LinearSVC(random_state=42, max_iter=10000),
      "K-Nearest Neighbors (K=5)" : KNeighborsClassifier(n_neighbors=5)
  }

  results = {}

  for name, classifier in classifiers.items():
    print(f"\n--- eğitiliyor: {name}")

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1= f1_score(y_test, y_pred, average='macro', zero_division=0)

    results[name] = {
            "Doğruluk (Accuracy)": f"{acc:.4f}",
            "Hassasiyet (Precision)": f"{prec:.4f}",
            "Geri Çağırma (Recall)": f"{rec:.4f}",
            "F1-Skoru": f"{f1:.4f}"
        }

  return results

#main

if __name__ == '__main__':
  print("görüntü sınıflandırma eğitimi başlıypr")
  print(f"Hog parametreleri {HOG_PARAMS}")

  X,y=load_data_simple(DATA_DIR)

  if X.size == 0:
    print("\n fatal hata")

  else:
    print(f"\n toplam {len(X)} örnek için {X.shape[1]} boyutlu HOG özellik vektörleri .ççıkartıldı")

    # Sınıflandırma ve Karşılaştırma Yap
    comparison_results = classify_and_compare(X, y)

    print("\n==============================================")
    print("📊 Sınıflandırıcı Karşılaştırma Sonuçları (HOG):")
    print("==============================================")
    
    # Sonuçları Temiz Bir Şekilde Yazdır
    for classifier, metrics in comparison_results.items():
        print(f"\n[{classifier}]")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")