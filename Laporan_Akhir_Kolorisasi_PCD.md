# Laporan Akhir – Sistem Kolorisasi Foto Makro Hitam Putih | PCD

## KERANGKA ACUAN KERJA (KAK)
**Sistem Kolorisasi Foto Makro Hitam Putih Berbasis Pengolahan Citra Digital dan Kecerdasan Buatan**

Harlan Luthi Permana – 32230033, Mohammad Bintang Indy Taura Putra – 32230064, Ruben Wijaya – 32230048

---

## Uraian Pendahuluan

### 1. Latar Belakang
Fotografi pada masa lampau sebagian besar didominasi oleh format hitam putih (grayscale), yang sering kali menyulitkan kita untuk merasakan suasana yang sebenarnya dari momen tersebut. Mengembalikan warna pada foto-foto historis (kolorisasi) secara manual merupakan tugas yang membutuhkan waktu yang lama, keterampilan artistik, dan pemahaman mendalam tentang konteks sejarah.

Kemajuan di bidang Pengolahan Citra Digital (PCD) dan Kecerdasan Buatan (AI), khususnya penggunaan arsitektur Convolutional Neural Network (CNN), telah menawarkan solusi otomatis untuk tugas kolorisasi ini. Model deep learning mampu "mempelajari" korelasi antara kecerahan (luminance) dan warna (chrominance) dari jutaan gambar, sehingga memungkinkan proses pemberian warna yang lebih realistis secara otomatis dan dalam waktu singkat.

Atas dasar kebutuhan ini, dikembangkanlah "Sistem Kolorisasi Foto Makro Hitam Putih Berbasis Pengolahan Citra Digital dan Kecerdasan Buatan" sebagai bagian dari Ujian Akhir Semester (UAS) Mata Kuliah Pengolahan Citra Digital (PCD). Sistem ini memadukan teknik pemrosesan citra dasar dengan framework AI *Zhang et al. Colorization CNN (ECCV 2016)* untuk merekonstruksi warna secara meyakinkan, sekaligus mengimplementasikan pipeline lengkap dari tahap prapemrosesan hingga pascapemrosesan citra.

### 2. Maksud dan Tujuan
**a. Maksud**
Maksud dari kegiatan pengembangan sistem ini adalah merancang, mengimplementasikan, dan mengevaluasi pipeline kolorisasi foto hitam putih berbasis AI yang mengintegrasikan teknik Pengolahan Citra Digital, sebagai alat otomatis untuk merestorasi warna pada gambar klasik atau dokumentasi sejarah.

**b. Tujuan**
1. Mengimplementasikan teknik pengolahan ruang warna (Color Space Conversion) dari BGR ke LAB untuk memisahkan informasi iluminasi dan warna.
2. Menerapkan model deep learning *Zhang Colorization CNN* untuk memprediksi saluran chrominance (A dan B) dari citra input.
3. Melakukan tahapan pascapemrosesan (Post-processing) seperti Bilateral Filtering dan peningkatan saturasi untuk memperhalus dan mengoptimalkan kualitas warna yang dihasilkan.
4. Menyediakan antarmuka Command Line (CLI) dan Graphical User Interface (GUI) yang interaktif untuk memudahkan proses restorasi oleh pengguna akhir.

### 3. Sasaran
Sasaran dari pengembangan sistem ini adalah:
- **Mahasiswa dan Akademisi:** Sebagai referensi implementasi integrasi antara model Deep Learning (CNN) dan algoritma Pengolahan Citra Digital klasik.
- **Sejarawan dan Arsiparis:** Sebagai alat bantu untuk memberikan warna pada arsip-arsip foto bersejarah agar terlihat lebih hidup dan relevan dengan audiens modern.
- **Masyarakat Umum:** Untuk memulihkan foto-foto keluarga masa lampau atau eksperimen fotografi makro hitam putih menjadi berwarna.

### 4. Lokasi Pengerjaan
Pengembangan sistem dilaksanakan di Laboratorium Pengolahan Citra Digital.

### 5. Sumber Pendanaan
Kegiatan ini merupakan bagian dari Evaluasi UAS Mata Kuliah Pengolahan Citra Digital, dibiayai secara mandiri/akademis.

### 6. Nama dan Organisasi Pengembang
| | |
| --- | --- |
| **Nama Pengembang** | Harlan Luthi Permana (32230033)<br>Mohammad Bintang Indy Taura Putra (32230064)<br>Ruben Wijaya (32230048) |
| **Mata Kuliah** | Pengolahan Citra Digital (PCD) |
| **Program Studi** | Informatika |

---

## Data Penunjang

### 7. Data Dasar
1. Citra digital grayscale (hitam putih) atau citra berwarna yang dikonversi ke grayscale sebagai input utama pengujian sistem.
2. Model bobot (weights) pre-trained dari *Zhang et al. Colorization CNN (ECCV 2016)* berbasis Caffe Framework.

### 8. Standar Teknis
Dalam pengembangan sistem ini, diterapkan standar dan referensi teknis sebagai berikut:
1. Ruang Warna CIE LAB sebagai representasi citra untuk pemisahan L (Lightness) dan komponen warna A dan B.
2. Arsitektur Zhang Colorization CNN yang dijalankan menggunakan modul OpenCV DNN.
3. Bilateral Filter untuk penghalusan warna yang tetap mempertahankan tepi objek (edge-preserving).
4. Python 3.8+ sebagai bahasa pemrograman utama.
5. OpenCV (Open Source Computer Vision Library) dan NumPy sebagai komponen komputasi matriks dan pemrosesan citra.

### 9. Referensi Ilmiah dan Hukum
- Zhang, R., Isola, P., & Efros, A. A. (2016). Colorful Image Colorization. In European Conference on Computer Vision (ECCV).
- Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and color images. Sixth International Conference on Computer Vision.

---

## Ruang Lingkup

### 10. Lingkup Pekerjaan
Lingkup kegiatan pengembangan sistem ini terbagi menjadi 5 modul utama:
1. **Modul A (Preprocessing):** Memuat gambar, validasi, konversi dari BGR ke grayscale, konversi ke ruang warna LAB, serta ekstraksi channel L (luminance).
2. **Modul B (Persiapan Input):** Meresize channel L ke resolusi standar (224x224), dan melakukan normalisasi (mean centering) sebelum citra dikirim ke model CNN sebagai *blob* tensor 4D.
3. **Modul C (Inferensi AI / Kolorisasi):** Menginisialisasi dan mengeksekusi model Zhang CNN untuk memprediksi channel A dan B dari representasi L input.
4. **Modul D (Rekonstruksi Gambar):** Melakukan proses resize channel A dan B kembali ke ukuran resolusi awal, menggabungkan channel L dengan AB, serta mengonversinya kembali menjadi ruang warna BGR.
5. **Modul E (Post-processing):** Memodifikasi saturasi untuk menghasilkan warna yang lebih natural, menerapkan *Bilateral Filtering* untuk meminimalkan artefak prediksi warna, serta menerapkan teknik *blending* dengan citra asli.

### 11. Keluaran (Output)
**a. Perangkat Lunak (Software)**
1. File program utama: `main.py` dan antarmuka `gui.py`.
2. File modul pemrosesan: `modul_a_preprocessing.py`, `modul_b_persiapan.py`, `modul_c_kolorisasi.py`, `modul_d_rekonstruksi.py`, `modul_e_postprocessing.py`.
3. File requirements: `requirements.txt`.

**b. Dokumentasi Teknis**
Laporan Akhir berisi implementasi lengkap, pembagian tugas modul, dan hasil uji sistem.

**c. Hasil Visualisasi**
Sistem mengembalikan file *output* berupa citra kolorisasi hasil inferensi AI yang berformat `.jpg` atau `.png`. Selain itu, terdapat visualisasi antarmuka pengguna untuk membandingkan citra *Before* dan *After*.

### 12. Peralatan, Bahasa, dan Library yang Digunakan
**a. Perangkat Keras (Hardware)**
- Komputer/Laptop dengan RAM minimal 8 GB.
- Prosesor dengan kapabilitas komputasi standar (sistem berjalan optimal di CPU karena implementasi OpenCV DNN module).

**b. Perangkat Lunak dan Library**
| Library / Tools | Fungsi |
| --- | --- |
| Python 3.8+ | Bahasa pemrograman utama |
| OpenCV (opencv-python) | Pemrosesan citra digital, manipulasi matriks gambar, dan modul inferensi DNN |
| NumPy | Komputasi numerik dan operasi *array* multidimensi |
| Argparse | *Parsing parameter command-line interface* (CLI) |

### 13. Peralatan dan Material dari Pengembang
- Lingkungan pengembangan (IDE): VS Code, PyCharm.
- Akses internet untuk pengunduhan arsitektur Caffe Model dan dependensi.

### 14. Lingkup Kewenangan Pengembang
- Mengoptimalkan hyperparameter *blending* warna dan parameter operasi morfologis (Bilateral Filter).
- Melatih dan menala parameter (fine-tuning) antarmuka dan parameter *saturation boost*.
- Mengatur tata letak modul secara terstruktur untuk pengembangan lebih lanjut.

### 15. Personil Pengembang
Struktur tim pengembang proyek terdiri dari mahasiswa pengembang:
| Nama | Peran | Tanggung Jawab |
| --- | --- | --- |
| **Mohammad Bintang Indy Taura Putra** | Ketua Kelompok & Lead Integrator | Mengintegrasikan seluruh modul pada `main.py`, merancang parameter post-processing, dan mengawal alur implementasi. |
| **Harlan Luthi Permana** | Image Processing Specialist | Merancang fungsi preprocessing, segmentasi saluran warna, konversi LAB, dan persiapan *tensor blob* model AI. |
| **Ruben Wijaya** | AI Inferencing Specialist | Memuat konfigurasi CNN model, mengelola inferensi probabilitas warna, dan merekonstruksi keluaran *tensor* kembali ke resolusi asli (BGR). |

---

## Deskripsi Teknis Sistem

### A. Arsitektur Pipeline Sistem
Sistem ini menggunakan *pipelining* terstruktur dalam 5 fase (Modul A - E):
1. **Modul A (Prapemrosesan):** BGR → Grayscale → LAB. Diperoleh saluran L (*Lightness*).
2. **Modul B (Persiapan):** Saluran L diubah dimensinya menjadi 224x224, nilai rata-rata dikurangi (normalisasi mean 50) membentuk masukan AI.
3. **Modul C (Kolorisasi):** Eksekusi *Zhang et al. CNN* memanfaatkan OpenCV DNN. Menghasilkan prediksi A dan B (skala 56x56).
4. **Modul D (Rekonstruksi):** Skalakan matriks A dan B ke dimensi citra awal, gabung dengan salinan L asli, kemudian ubah ke ruang warna BGR.
5. **Modul E (Pascapemrosesan):** Tingkatkan intensitas warna menggunakan transformasi saturasi, dan haluskan luaran dengan *Bilateral Filter* untuk meratakan warna tanpa merusak garis tepi objek.

### B. Implementasi Modul Utama (main.py)
Program `main.py` berperan sebagai pengontrol alur (controller) dari berbagai modul terpisah. Logikanya berjalan berurutan sesuai perintah yang diterima melalui Command Line Argument (CLI) atau dialihkan otomatis ke antarmuka grafis (GUI) apabila parameter `input` tidak terdeteksi.

```python
# Integrasi Alur Pipeline pada main.py
img_original = load_and_validate(args.input)
img_bw = convert_to_grayscale(img_original)
img_lab = convert_to_lab(img_bw)
L_channel = extract_l_channel(img_lab)

L_resized = resize_for_model(L_channel, target_size=(224, 224))
L_normalized = normalize_l_channel(L_resized)
blob = prepare_blob(L_normalized)

net = load_model()
ab_predicted = predict_ab_channels(net, blob)

ab_resized = resize_ab_to_original(ab_predicted, original_shape=(h, w))
img_lab_result = combine_lab_channels(L_channel, ab_resized)
img_colored = convert_lab_to_bgr(img_lab_result)

img_saturated = boost_saturation(img_colored, factor=args.saturation)
img_filtered = apply_bilateral_filter(img_saturated, d=9, sigma_color=75, sigma_space=75)
img_final = blend_with_original(img_filtered, img_bw, alpha=args.blend)
```

### C. Analisis Teknik Pengolahan Citra Digital
Berbagai konsep fundamental PCD digunakan untuk mengoptimalkan proses restorasi ini:
1. **Color Space Transformation:** Penggunaan ruang warna *CIE LAB* sangat esensial karena memisahkan iluminasi (*Lightness*) dengan warna (*A* dan *B*). Hal ini membuat model AI cukup memprediksi nilai warna (saluran A dan B) tanpa harus merekonstruksi tingkat kecerahan yang sudah ada di citra aslinya.
2. **Mean-Centering:** Dilakukan pengurangan rerata konstan (nilai 50) pada saluran L. Ini adalah teknik normalisasi standar untuk input *neural networks* pada pengolahan citra guna mempercepat konvergensi dan keakuratan.
3. **Bilateral Filtering:** Penggunaan *Bilateral Filter* (d=9, sigma=75) dalam pascapemrosesan mengeliminasi derau warna buatan (*color bleeding* atau *halo artifacts*) akibat kesalahan regresi prediktif CNN di tepian gambar.

---

## Kesimpulan

Proyek Sistem Kolorisasi Foto Makro Hitam Putih Berbasis Pengolahan Citra Digital dan Kecerdasan Buatan telah berhasil diimplementasikan dengan memadukan paradigma pengolahan citra klasik dan modern. Beberapa kesimpulan kunci dari sistem ini adalah:
1. **Pemisahan dan Integrasi yang Terstruktur:** Pembagian alur kerja menjadi lima sub-modul (Prapemrosesan, Persiapan Input, Kolorisasi AI, Rekonstruksi, dan Pascapemrosesan) memberikan skalabilitas dan kejelasan tanggung jawab, yang terkoordinasi secara efektif oleh berkas `main.py`.
2. **Keunggulan Ruang Warna LAB:** Ruang warna LAB terbukti menjadi ruang warna yang ideal untuk tugas AI kolorisasi gambar, memastikan detail kecerahan tidak terpengaruh oleh inferensi prediksi warna.
3. **Pentingnya Pascapemrosesan:** Pendekatan *Bilateral Filtering* dan modifikasi saturasi mampu memperbaiki ketidakakuratan prediksi AI, menyamarkan transisi *color blobs*, dan membuat hasil restorasi terlihat jauh lebih tajam dan meyakinkan.
4. **Fleksibilitas Penggunaan:** Ketersediaan dukungan *Command Line Interface (CLI)* dan penanganan parameter opsional (`--saturation`, `--blend`) memberikan pengembang kendali halus (fine-tuning) terhadap bagaimana sistem bereaksi pada berbagai tipe masukan citra foto jadul.
