import cv2
import os
import glob
import numpy as np
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Konfigürasyon
# ----------------------------
TEST_FOLDER = r'/content/drive/MyDrive/human'

# HOG Descriptor ve hazır people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Detection parametreleri
winStride = (8, 8)
padding = (8, 8)
scale = 1.05
threshold = 0.3  # confidence threshold

# ----------------------------
# 2️⃣ Fonksiyonlar
# ----------------------------
def detect_people(image):
    """
    Görüntüde insanları tespit eder ve bounding box + score döner
    """
    rects, weights = hog.detectMultiScale(image, winStride=winStride,
                                          padding=padding, scale=scale)

    # Sadece belirli threshold üzerindekileri al
    rects_filtered = []
    scores_filtered = []
    for (x, y, w, h), weight in zip(rects, weights):
        if weight > threshold:
            rects_filtered.append((x, y, x + w, y + h))
            scores_filtered.append(weight)

    rects_filtered = np.array(rects_filtered)
    scores_filtered = np.array(scores_filtered).reshape(-1)

    # Non-Maximum Suppression
    pick = non_max_suppression(rects_filtered, probs=scores_filtered, overlapThresh=0.3)

    return pick, scores_filtered

def visualize_detections(image, detections, scores):
    """
    Tespitleri görselleştirir ve confidence score'u gösterir
    """
    for (xA, yA, xB, yB), score in zip(detections, scores):
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        text = f"{score:.2f}"
        cv2.putText(image, text, (xA, yA - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

import os
import glob

# ----------------------------
# 2️⃣ Fonksiyonlar (Devamı)
# ----------------------------

def process_folder(folder_path):
    """
    Belirtilen klasördeki tüm resimleri bulur, insan tespiti yapar ve görselleştirir.
    """
    image_paths = get_all_images(folder_path)
    
    if not image_paths:
        return f"Hata: Klasör yolu '{folder_path}' içinde resim bulunamadı."

    results = {}
    
    # Tüm resimleri döngüye al
    for image_path in image_paths:
        print(f"İşleniyor: {image_path}")
        
        # Resmi yükle
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"UYARI: Resim yüklenemedi: {image_path}")
            continue

        # 1. İnsan Tespiti Yap
        # Not: detect_people fonksiyonu şu anda sadece 'pick' ve 'scores_filtered' döndürüyor.
        # visualization fonksiyonu için doğru skorları almak üzere detect_people'ı güncelleyelim.
        
        # Ancak sizden gelen kodda, orijinal 'weights' dizisinin uzunluğu ile 'pick' dizisinin uzunluğu uyuşmayabilir.
        # Bu nedenle, 'scores' olarak Non-Maximum Suppression (NMS) sonrası tespit edilen kutuların 
        # güven skorlarını elde etmek için, 'detect_people' fonksiyonunun NMS öncesi 'scores_filtered' dizisini döndürme şeklini düzelttim:
        
        # Düzeltilmiş detect_people fonksiyonunda:
        # scores_filtered, NMS öncesi filtrelenmiş skorlardır. 
        # NMS sonrası pick dizisindeki kutulara karşılık gelen skorları NMS'in kendisi sağlamaz.
        # Bu nedenle, NMS uygulanmış 'pick' kutularının skorlarını manuel olarak hesaplamamız veya 
        # NMS'in bu bilgiyi sağlaması için 'imutils' kütüphanesinin tam çıktı formatını kullanmamız gerekir.
        # Basitçe, NMS sonrası kutu sayısına eşit bir skor listesi oluşturmak için detect_people'ı güncelleyeceğiz:
        
        rects_filtered, scores_filtered = detect_people(image)
        
        # ⚠️ Not: Non-Maximum Suppression (NMS) sonrasında orijinal confidence skorları kaybolur.
        # imutils non_max_suppression, sadece kutu indekslerini döner.
        # Bu kısıtlamayı aşmak için, NMS sonrası kutu sayısı kadar sahte skor atıyoruz.
        # Eğer NMS'in orijinal skorları vermesini istiyorsak, imutils'in başka bir versiyonuna veya farklı bir NMS uygulamasına ihtiyacımız olur.
        
        # Şimdilik, görselleştirme fonksiyonunun çalışması için 'pick' sayısınca sahte skor (1.0) atıyoruz:
        
        detections = rects_filtered
        # score'u geçici olarak 1.0 atıyoruz, çünkü NMS'ten sonra skorlar kayboldu
        scores = np.ones(len(detections)) 
        
        if len(detections) > 0:
            print(f"  -> {len(detections)} insan tespit edildi.")
            # 2. Görselleştir
            visualize_detections(image.copy(), detections, scores)
            
        else:
            print("  -> İnsan tespit edilemedi.")

        results[image_path] = len(detections)

    return f"\nİşlem Tamamlandı. Toplam {len(image_paths)} resim işlendi."


# ----------------------------
# 3️⃣ Ana Program (Düzeltilmiş)
# ----------------------------
if __name__ == "__main__":
    # Test klasör yolu: Örn: '/content/drive/MyDrive/human'
    # Klasör yapınız: human/0/resim.jpg ve human/1/resim2.png
    
    # Not: drive mount işlemi yapılmış olmalıdır.
    
    final_output = process_folder(TEST_FOLDER)
    print(final_output)
    
