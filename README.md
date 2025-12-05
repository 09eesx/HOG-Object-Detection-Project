---

#  HOG Temelli Nesne Tespiti ve Sınıflandırma

Bu proje, **Histogram of Oriented Gradients (HOG)** algoritmasını kullanarak hem **insan tespiti** hem de **özel nesne sınıflandırması** (köpek / köpek değil) uygulamalarını gerçekleştirmektedir. HOG, nesnenin şekil bilgisini öne çıkaran güçlü bir özellik çıkarma yöntemidir. Bu proje kapsamında HOG; **SVM**, **KNN** gibi geleneksel makine öğrenimi yöntemleriyle entegre edilerek performans karşılaştırması yapılmıştır.

---

##  1. Proje Özeti

Bu çalışma:

* OpenCV’nin önceden eğitilmiş **HOG + SVM insan tespit modeli** ile pedestrian (yaya) tespitini,
* Kullanıcı tarafından oluşturulan köpek veri setiyle **HOG + SVM/KNN ikili sınıflandırma** modelini,
* Çok ölçekli nesne tespiti (multi-scale detection),
* Non-Maximum Suppression (NMS),
* Confidence score analizi

gibi teknikleri içermektedir.

---

##  2. Görev 2.1: Önceden Eğitilmiş Model ile İnsan Tespiti

###  Kullanılan Model

OpenCV'nin yerleşik:

```python
cv2.HOGDescriptor_getDefaultPeopleDetector()
```

modeli kullanılmıştır. Bu model, Dalal & Triggs (2005) tarafından geliştirilen klasik **HOG + Linear SVM pedestrian detector** modelidir.

###  Kullanılan Teknikler

| Teknik                              | Açıklama                                                  |
| ----------------------------------- | --------------------------------------------------------- |
| **Multi-Scale Detection**           | Farklı boyutlardaki insanları yakalamak için `scale=1.05` |
| **Thresholding**                    | Zayıf tespitleri elemek için `threshold = 0.3`            |
| **Non-Maximum Suppression (NMS)**   | Çakışan kutuları filtrelemek için `overlapThresh = 0.3`   |
| **Bounding Box + Confidence Score** | Her tespit skor ile işaretlenir                           |

###  Beklenen Çıktılar

* NMS uygulanmış insan tespitli görüntüler
* Her görüntüde tespit edilen kişi sayısı
* Farklı eşik değerlerinin (0.3, 0.5) tespit başarına etkisi

---

##  3. Görev 2.2: Köpek Sınıflandırması (HOG + Makine Öğrenimi)

Bu görevde, özel bir veri setiyle köpek / köpek değil sınıflandırması yapılmıştır.

###  Veri Seti

| Sınıf                 | Adet |
| --------------------- | ---- |
| Köpek (Pozitif)       | 50   |
| Köpek Değil (Negatif) | 50   |

Toplam: **100 görüntü**

###  İşleme Adımları

####  1. HOG Özellik Çıkarımı

Tüm görüntüler:

* **64×128 boyutuna** yeniden ölçeklendi
* HOG parametreleri:

  * 9 yönelim (orientations)
  * 8×8 hücre boyutu
  * 2×2 blok
* Ortalama **3780** boyutlu HOG vektörü elde edildi.

####  2. Model Eğitimi

İki model karşılaştırıldı:

* **Linear SVM (LinearSVC)**
* **KNN (K=5)**

#### ✔️ 3. Değerlendirme

Veri seti:
 **%70 Eğitim — %30 Test**

---

##  3.2 Test Sonuçları

| Model          | Accuracy | Precision  | Recall | F1-Score |
| -------------- | -------- | ---------- | ------ | -------- |
| **Linear SVM** | 0.7000   | 0.6944     | 0.6900 | 0.6914   |
| **KNN (K=5)**  | 0.7000   | **0.7500** | 0.7262 | 0.6970   |

###  Sonuç Analizi

* Her iki model de **%70 doğruluk** elde etmiştir.
* HOG'un, nesnelerin karakteristik şekil bilgisini iyi temsil ettiği bir kez daha görülmüştür.
* KNN, **Precision=0.75** ile pozitif tahminlerde daha güvenilir davranmıştır.
* SVM, karar sınırlarını daha agresif öğrendiği için dengeli performans göstermiştir.

---

## ▶ 4. Kullanım Talimatları

###  Gerekli Kütüphaneler

```bash
pip install opencv-python numpy imutils scikit-image scikit-learn matplotlib joblib
```

###  Girdi / Çıktı Ayarları

Kodun başındaki:

```python
TEST_FOLDER = "/content/drive/MyDrive/human"
OUTPUT_FOLDER = "/content/human_detections"
```

kısımlarını kendi dizinlerinize göre düzenleyin.

###  Çalıştırmak İçin:

```python
if __name__ == "__main__":
    main()
```

komutunu çalıştırmanız yeterlidir.

Çıktılar, otomatik olarak **OUTPUT_FOLDER** içine kaydedilecektir.

---

##  Sonuç

Bu proje, HOG'un:

* İnsan tespitinde,
* Köpek sınıflandırmasında,
* Makine öğrenimi ile birlikte kullanımında

ne kadar güçlü bir özellik çıkarıcı olduğunu göstermektedir.

Hem geleneksel bilgisayarlı görü tekniklerini hem de modern değerlendirme yöntemlerini bir arada kullanarak sağlam bir görüntü işleme altyapısı geliştirilmiştir.

---
