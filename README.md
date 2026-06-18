# Proyek UAS — Kolorisasi Foto Hitam-Putih Berbasis AI
**Mata Kuliah:** TIF23 - Pengolahan Citra Digital
**Framework AI:** Zhang et al. Colorization CNN (ECCV 2016)

---

## 👥 Anggota Kelompok
* **32230033** - Harlan Luthi Permana
* **32230048** - Ruben Wijaya
* **32230064** - Mohammad Bintang Indy Taura Putra

---

## 📂 Struktur File

```
colorization_uas/
├── main.py                      ← Program utama (entry point)
├── app.py                       ← Web GUI Server (Flask) & Logika Editor
├── download_model.py            ← Unduh model AI (jalankan sekali di awal)
├── requirements.txt             ← Daftar library yang dibutuhkan
├── modul_a_preprocessing.py     ← Modul 1: Grayscaling + LAB
├── modul_b_persiapan.py         ← Modul 2: Resize + Normalisasi
├── modul_c_kolorisasi.py        ← Modul 3: Zhang CNN (inti AI)
├── modul_d_rekonstruksi.py      ← Modul 4: Rekonstruksi gambar
├── modul_e_postprocessing.py    ← Modul 5: Saturasi + Filter + Blend
├── samples/                     ← Folder berisi gambar-gambar B&W contoh
└── models/                      ← Folder model AI (diisi otomatis)
    ├── colorization_deploy_v2.prototxt
    ├── colorization_release_v2.caffemodel  (~125 MB)
    └── pts_in_hull.npy
```

---

## 🚀 Fitur Unggulan Web GUI
1. **Real-time Colorization**: Proses pewarnaan memanfaatkan CNN via *OpenCV DNN Module* dan dikirim dengan *Server-Sent Events (SSE)* untuk *progress bar*.
2. **Built-in Photo Editor**: Pengaturan Exposure, Brilliance, Highlights, Shadows, Brightness, Contrast, Saturation, Vibrance, Warmth, Tint, Sharpness, Definition, Noise Reduction, dan Vignette secara dinamis.
3. **Gallery Samples**: Menyediakan beberapa gambar *hitam-putih* bawaan (Bird, Rose, Gears, Aurora, Eye) yang bisa langsung dicoba dengan satu klik.
4. **Before/After Split View**: Menampilkan perbandingan gambar asli (hitam-putih) dan hasil kolorisasi AI secara berdampingan.

---

## 💻 Cara Menjalankan

Langkah-langkah di bawah ini sedikit berbeda tergantung sistem operasi komputer yang Anda gunakan (Mac atau Windows).

### Langkah 1 — Install library
**🍎 Untuk macOS / Linux:**
```bash
pip3 install -r requirements.txt
```
**🪟 Untuk Windows:**
```cmd
pip install -r requirements.txt
```

### Langkah 2 — Download model AI (hanya sekali, ~125 MB)
**🍎 Untuk macOS / Linux:**
```bash
python3 download_model.py
```
**🪟 Untuk Windows:**
```cmd
python download_model.py
```

### Langkah 3 — Jalankan Web GUI
**🍎 Untuk macOS / Linux:**
```bash
python3 main.py
```
**🪟 Untuk Windows:**
```cmd
python main.py
```
Aplikasi akan secara otomatis membuka Web GUI (Flask) di browser pada alamat:
`http://localhost:5050`

---

## ⚙️ Proses Pengolahan Citra yang Digunakan

| # | Fungsi OpenCV | Modul | Keterangan |
|---|---|---|---|
| 1 | `cv2.cvtColor(BGR→GRAY)` | A | Grayscaling |
| 2 | `cv2.cvtColor(BGR→LAB)` | A | Konversi ruang warna |
| 3 | `cv2.resize()` | B | Resize ke 224×224 untuk CNN |
| 4 | `cv2.dnn.blobFromImage()` | B | Pembentukan tensor input |
| 5 | `cv2.dnn.readNetFromCaffe()` | C | Load model AI |
| 6 | `net.forward()` | C | Inferensi CNN (inti AI) |
| 7 | `cv2.resize()` | D | Resize AB output ke ukuran asli |
| 8 | `np.concatenate()` | D | Gabung channel L + A + B |
| 9 | `cv2.cvtColor(LAB→BGR)` | D | Konversi balik ke BGR |
| 10| `cv2.cvtColor(BGR→HSV)` | E | Untuk boost saturasi (versi dasar) |
| 11| `cv2.bilateralFilter()` | E | Filter tanpa blur tepi |
| 12| `cv2.addWeighted()` | E | Blending dua gambar |
| 13| *Manipulasi Numpy Matrix* | `app.py`| Operasi matematis untuk Photo Editor di web |

---

## 📚 Referensi

Zhang, R., Isola, P., & Efros, A. A. (2016).
*Colorful Image Colorization.*
European Conference on Computer Vision (ECCV 2016).
https://doi.org/10.1007/978-3-319-46487-9_40
