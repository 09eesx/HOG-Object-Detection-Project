import numpy as np
import matplotlib.pyplot as plt
import math
import skimage.io, skimage.color # Görüntü yükleme için gerekli

def compute_gradients(image):
    """Görüntünün x (yatay) ve y (dikey) yönündeki gradyanlarını hesaplar.
    girdi: gri görüntü
    çıktı: gradient_magnitude, gradient_angle"""

    Kx = np.array([[-1,0,1]])
    Ky = np.array([[-1], [0], [1]]) 

    g_col = np.empty_like(image, dtype=float) #görüntünün x ve y yönlerindeki türevlerini saklamak için boş float tipinde diziler oluşturur.
    g_row = np.empty_like(image, dtype=float)
    #dikey gradyan
    g_row[0,:] = 0
    g_row[-1,:] = 0
    g_row[1:-1,:] = image[2:,:] - image[:-2,:]

    #yatay gradyan
    g_col[0,:] = 0
    g_col[-1,:] = 0
    g_col[:, 1:-1] = image[:,2:] - image[:,:-2]

    #büyüklük
    gradient_magnitude = np.sqrt(g_row**2 + g_col**2)
    
    #arctan2 ile yön
    gradient_angle = np.arctan2(g_row,g_col)
    gradient_angle = np.rad2deg(gradient_angle)

    #0-180 derece arası indirgeme
    gradient_angle = gradient_angle % 180

    return gradient_magnitude, gradient_angle

## HÜCRE İÇİN HİSTOGRAM HESAPLAMA

def create_cell_histogram(cell_magnitude, cell_angle, num_bins=9):
    """bir hücre için yönelim histogramı oluşturulur (trilinear interpolation)"""

    HOG_hist = np.zeros(shape=(num_bins))
    bins_size = 180.0 / num_bins

    #tüm piksel çiftlerinde döngü
    for angle, magnitude in zip(cell_angle.flatten(), cell_magnitude.flatten()):
        angle = angle % 180.0

        #bin indeksini hesapla
        bin_float_index = angle / bins_size #48 derecelik açı 20'ye bölünüyor. 2.4 geldi. floor ile 2'ye yuvarlanıyor ve ekleniyor.(0,1,2,3,4,5,6,7,8,9)
        """Bir pikselin gradyan açısı ve büyüklüğü var.
           Açısı tam olarak histogramdaki bir kutuya denk gelmeyebilir.
           O zaman iki komşu kutuya paylaştırılır"""
        #birinci (daha düşük açılı) bin'in indeksi
        first_bin_idx = np.uint8(np.floor(bin_float_index)) #2.4 -floor-> 2
        #daha yüksek açılı bin'in indeksi
        second_bin_idx = (first_bin_idx + 1) %num_bins # 2+1 % 9 = 3

        weight_for_second_bin = bin_float_index - first_bin_idx  #2.4 - 2 = 0.4
        weight_for_first_bin = 1.0 - weight_for_second_bin #1 - 0.4 = 0.6

        #ağırlıklı oylama (#magnitude ile çarp)
        HOG_hist[first_bin_idx] += weight_for_first_bin * magnitude #ters işlemle 0.6 first bini
        HOG_hist[second_bin_idx] += weight_for_second_bin * magnitude #0.4 second binin

def normalize_block(block_histogram, method='L2', epsilon = 1e-5):
    """Blok histogramını normalize eder. 
    Çıktı: Normalize edilmiş histogram"""

    if method == 'L2':
        l2_norm_sq = np.sum(block_histogram ** 2)
        denominator = np.sqrt(l2_norm_sq + epsilon**2)
        normalized_hist = block_histogram / denominator

    elif method == 'L2-Hys':
        # 1. L2 normalizasyonu
        l2_norm_sq = np.sum(block_histogram ** 2)
        denominator = np.sqrt(l2_norm_sq + epsilon**2)
        normalized_hist = block_histogram / denominator
        
        # 2. Kırpma (Hysteresis)
        normalized_hist = np.minimum(normalized_hist, 0.2)
        
        # 3. Yeniden L2 normalizasyonu
        l2_norm_sq_hys = np.sum(normalized_hist ** 2)
        denominator_hys = np.sqrt(l2_norm_sq_hys + epsilon**2)
        normalized_hist = normalized_hist / denominator_hys

    else:
        # L1 ve L1-sqrt opsiyonları da burada eklenebilir.
        raise ValueError(f"Bilinmeyen normalizasyon metodu: {method}")
        
    return normalized_hist
    
#4. TAM HOG DESCRİPTOR HESAPLAMA
def compute_hog_descriptor(image, cell_size=(8,8) ,block_size=(2,2), num_bins=9):
    """sliding window ile blokları tara"""
    mag, angle = compute_gradients(image)
    img_h, img_w = mag.shape
    cell_h, cell_w = cell_size
    block_h, block_w = block_size

    #hücre ızgarası boyutları
    n_cells_row = img_h // cell_h
    n_cells_col = img_w // cell_w 

    #1 hücre kaymalı sliding window
    n_blocks_row = n_cells_row - block_h +1
    n_blocks_col = n_cells_col - block_w +1

    if n_blocks_row <= 0 or n_blocks_col <=0:
        raise ValueError("görüntü göccük")

    block_descriptor_size = block_h * block_w * num_bins
    hog_descriptor = []

    cell_histograms = np.zeros((n_cells_row, n_cells_col, num_bins))

    
    for r in range(n_cells_row):
        for c in range(n_cells_col):
            r_start, r_end = r * cell_h, (r + 1) * cell_h
            c_start, c_end = c * cell_w, (c + 1) * cell_w
            
            cell_mag = mag[r_start:r_end, c_start:c_end]
            cell_angle = angle[r_start:r_end, c_start:c_end]
            
            cell_histograms[r, c, :] = create_cell_histogram(cell_mag, cell_angle, num_bins)
        # Blokları tara ve normalize et (Sliding Window)
    for br in range(n_blocks_row):
        for bc in range(n_blocks_col):
            
            # Bloğun kapsadığı hücrelerin histogramlarını seç (Örn: 2x2 hücre)
            block_hists = cell_histograms[br:br + block_h, bc:bc + block_w, :]
            
            # 1D vektöre birleştir (Örn: 2*2*9 = 36 boyutlu)
            block_vector = block_hists.flatten()
            
            # Normalize et (Varsayılan: L2-Hys)
            normalized_block = normalize_block(block_vector, method="L2-Hys")
            
            # Ana HOG listesine ekle
            hog_descriptor.extend(normalized_block)
            
    return np.array(hog_descriptor)
    
    #HOG görselleştirme
def visualize_hog(image, cell_size=(8,8), num_bins = 9):
    """difüzyon okları, quiver kullandık"""
    mag, angle = compute_gradients(image)
    img_h, img_w = mag.shape
    cell_h, cell_w = cell_size
    
    n_cells_row = img_h // cell_h
    n_cells_col = img_w // cell_w    
    bin_size = 180.0 / num_bins
    hist_bins = np.arange(num_bins) * bin_size + (bin_size / 2) # Merkezi açılar

    # 1. Hücre histogramlarını hesapla
    cell_histograms = np.zeros((n_cells_row, n_cells_col, num_bins))
    for r in range(n_cells_row):
        for c in range(n_cells_col):
            r_start, r_end = r * cell_h, (r + 1) * cell_h
            c_start, c_end = c * cell_w, (c + 1) * cell_w
            
            cell_mag = mag[r_start:r_end, c_start:c_end]
            cell_angle = angle[r_start:r_end, c_start:c_end]
            cell_histograms[r, c, :] = create_cell_histogram(cell_mag, cell_angle, num_bins)
            
    # 2. Ok çizimi için verileri hazırla
    X_pos, Y_pos, U, V = [], [], [], []
    max_hist_val = np.max(cell_histograms)
    
    # Ölçeklendirme faktörü: Ok uzunluğunu hücre boyutunun yarısı civarında tutar
    # Bu, aşırı büyük oklar çizilmesini engeller
    scale_factor = 0.4 * cell_h / (max_hist_val + 1e-6) 

    for r in range(n_cells_row):
        for c in range(n_cells_col):
            center_x = c * cell_w + cell_w / 2
            center_y = r * cell_h + cell_h / 2
            
            hist = cell_histograms[r, c, :]
            
            for bin_idx in range(num_bins):
                magnitude = hist[bin_idx]
                if magnitude > 0.01 * max_hist_val: # Çok zayıf oyları ihmal et
                    
                    angle_deg = hist_bins[bin_idx]
                    angle_rad = np.deg2rad(angle_deg)
                    
                    length = magnitude * scale_factor
                    
                    # Okun X ve Y bileşenleri (dx, dy)
                    u_comp = length * np.cos(angle_rad)
                    v_comp = length * np.sin(angle_rad)
                    
                    # Merkezden yayılma (çift çizgi, HOG imzasız yön)
                    X_pos.extend([center_x, center_x])
                    Y_pos.extend([center_y, center_y])
                    U.extend([u_comp, -u_comp])
                    V.extend([v_comp, -v_comp])

    # 3. Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Orijinal Görüntü
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Orijinal Görüntü")
    axes[0].axis('off')
    
    # HOG Görselleştirmesi
    axes[1].imshow(image, cmap='gray')
    axes[1].set_title("HOG Difüzyon Okları")
    
    axes[1].quiver(X_pos, Y_pos, U, V, 
                   color='red', 
                   scale=1.0, 
                   units='xy', 
                   pivot='middle',
                   width=1.0, 
                   headwidth=0, headlength=0) # Çizgi (oksuz) çizer
    
    # Koordinat sistemini görüntüye uydur
    axes[1].set_ylim(img_h, 0)
    axes[1].set_xlim(0, img_w)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

#çalıştırma bloğu


def run_hog_test(image_path, cell_size=(8, 8), block_size=(2, 2), num_bins=9):
    """Verilen parametrelerle HOG'u çalıştırır ve sonuçları yazdırır."""
    try:
        # 1. Görüntüyü Yükle ve Griye Çevir
        img = skimage.io.imread(image_path)
        if img.ndim == 3:
            img_gray = skimage.color.rgb2gray(img)
        else:
            img_gray = img
            
        # 2. HOG Hesabı
        hog_vector = compute_hog_descriptor(img_gray, 
                                            cell_size=cell_size, 
                                            block_size=block_size, 
                                            num_bins=num_bins)

        # 3. Sonuçları Yazdır
        print("--- Test Sonuçları ---")
        print(f"Görüntü: {image_path}")
        print(f"Parametreler: Hücre={cell_size}, Blok={block_size}, Bin={num_bins}")
        print(f"Çıkarılan HOG Vektör Boyutu: {hog_vector.shape}")

        # 4. Görselleştirme
        visualize_hog(img_gray, cell_size=cell_size, num_bins=num_bins)

    except FileNotFoundError:
        print(f"Hata: Dosya bulunamadı: {image_path}")
    except ValueError as e:
        print(f"Hata: {e}")

#### TEST ETME
import cv2 # OpenCV for image processing (reading, HoG)
import matplotlib.pyplot as plt # For visualization
import os # For file system operations

# Eğer Python dosyanız src/ içinde ise, data/images klasörüne ulaşmak için bir üst klasöre (..) çıkmalısınız.
# ====================================================================
# DÜZELTİLMİŞ TEST ETME BLOĞU
# ====================================================================

import os 
# cv2 artık gerekli değil, çünkü skimage yükleme yapıyor ve cv2'nin sonucu kullanılmıyor.

def run_multiple_hog_tests(image_path):
    """
    Farklı parametrelerle HOG testlerini otomatik olarak çalıştırır.
    Her bir sonucu ayrı PNG olarak kaydeder.
    """
    tests = [
        # (cell_size, block_size, num_bins)
        ((8, 8),   (2, 2), 9),
        ((16, 16), (2, 2), 9),
        ((8, 8),   (3, 3), 9),
        ((8, 8),   (2, 2), 6),
        ((4, 4),   (2, 2), 9),
    ]

    for i, (cell, block, bins) in enumerate(tests):
        print(f"\n--- Test {i+1} ---")
        print(f"cell={cell}, block={block}, num_bins={bins}")

        # Çıktıyı figür olarak al
        fig = run_hog_test(image_path,
                           cell_size=cell,
                           block_size=block,
                           num_bins=bins)

        # Kaydet
        filename = f"hog_output_test_{i+1}_cell{cell}_block{block}_bins{bins}.png"
        fig.savefig(filename)
        print(f"KAYDEDİLDİ → {filename}")
        plt.close(fig)
run_multiple_hog_tests("/content/images.jpeg")
