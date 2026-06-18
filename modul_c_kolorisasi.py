import cv2
import numpy as np
import os

# ── Konfigurasi Path File Model ────────────────────────────────────────────────
MODEL_DIR   = 'models'
PROTO_PATH  = os.path.join(MODEL_DIR, 'colorization_deploy_v2.prototxt')
WEIGHTS_PATH= os.path.join(MODEL_DIR, 'colorization_release_v2.caffemodel')
CLUSTER_PATH= os.path.join(MODEL_DIR, 'pts_in_hull.npy')


# ── FUNGSI 1: Memuat Model ke Memori ──────────────────────────────────────────
def load_model() -> cv2.dnn_Net:
    """
    Memuat model Zhang Colorization dari file .prototxt dan .caffemodel,
    kemudian menyuntikkan cluster center warna (pts_in_hull.npy) ke
    layer-layer yang tepat.

    File yang dibutuhkan (unduh via download_model.py):
        1. colorization_deploy_v2.prototxt   → arsitektur/struktur CNN
        2. colorization_release_v2.caffemodel → bobot model (≈125 MB)
        3. pts_in_hull.npy                   → 313 cluster warna AB

    Returns:
        cv2.dnn_Net: Objek model yang siap untuk inferensi

    Raises:
        FileNotFoundError: Jika salah satu file model tidak ada
    """
    # Periksa keberadaan semua file yang diperlukan
    missing = []
    for path in [PROTO_PATH, WEIGHTS_PATH, CLUSTER_PATH]:
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            f"[MODUL C] File model tidak ditemukan:\n"
            + "\n".join(f"  ✗ {p}" for p in missing) +
            "\n\n  Solusi: Jalankan 'python download_model.py' terlebih dahulu!"
        )

    # ── Step 1: Baca arsitektur dan bobot model ──────────────────────────────
    # readNetFromCaffe() memuat model Caffe (format yang digunakan Zhang et al.)
    # .prototxt  = file teks berisi definisi layer-layer CNN
    # .caffemodel = file biner berisi semua nilai bobot hasil training
    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, WEIGHTS_PATH)

    # ── Step 2: Muat cluster center warna AB ────────────────────────────────
    # pts_in_hull.npy berisi 313 titik cluster dalam ruang warna AB
    # yang merepresentasikan warna-warna paling umum ditemukan di foto
    pts = np.load(CLUSTER_PATH)

    # Reshape dari (313, 2) → (2, 313, 1, 1) agar sesuai format blob layer
    pts = pts.transpose().reshape(2, 313, 1, 1)

    # ── Step 3: Suntikkan cluster ke layer 'class8_ab' dan 'conv8_313_rh' ──
    # 'class8_ab'  : layer yang mengubah probabilitas 313 kelas → nilai AB
    # 'conv8_313_rh': layer rebalancing yang memberi bobot pada setiap kelas
    #                 (nilai 2.606 = faktor temperature annealing)
    class8 = net.getLayerId('class8_ab')
    conv8  = net.getLayerId('conv8_313_rh')

    # Masukkan cluster centers sebagai bobot layer
    net.getLayer(class8).blobs = [pts.astype(np.float32)]

    # Masukkan faktor rebalancing (temperature = 2.606)
    # Nilai ini membuat output lebih "colorful" daripada terlalu abu-abu
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]


    return net


# ── FUNGSI 2: Prediksi Channel A dan B ────────────────────────────────────────
def predict_ab_channels(net: cv2.dnn_Net,
                        blob: np.ndarray) -> np.ndarray:
    """
    Menjalankan forward pass model Zhang CNN untuk memprediksi
    channel warna A dan B dari channel L yang diberikan.

    Alur inferensi:
        1. Blob (1, 1, 224, 224) dimasukkan ke model sebagai input
        2. Model melakukan forward pass melalui ±30 layer konvolusi
        3. Output: probabilitas 313 kelas warna dengan ukuran spasial 56×56
        4. Layer class8_ab mengubah 313 kelas → 2 nilai (A, B) per piksel
        5. Hasil akhir: tensor (1, 2, 56, 56)

    Kenapa outputnya 56×56 bukan 224×224?
        CNN dengan operasi pooling mengurangi resolusi spasial.
        Ukuran 56×56 = 224/4 (karena ada 2 MaxPooling stride-2).
        Channel AB akan di-resize kembali ke ukuran asli di Modul D.

    Args:
        net  (cv2.dnn_Net): Model Zhang CNN yang sudah dimuat
        blob (np.ndarray) : Input blob, shape (1, 1, 224, 224)

    Returns:
        np.ndarray: Channel AB hasil prediksi, shape (56, 56, 2)
    """
    # ── Step 1: Set blob sebagai input model ────────────────────────────────
    net.setInput(blob)

    # ── Step 2: Forward pass (inferensi) ────────────────────────────────────
    # net.forward() menjalankan seluruh rangkaian layer CNN dari awal hingga akhir
    # Ini adalah bagian terberat komputasinya — memakan waktu 1–5 detik
    ab_predicted = net.forward()   # Output shape: (1, 2, 56, 56)

    # ── Step 3: Hapus dimensi batch (N=1 tidak diperlukan) ──────────────────
    # Dari (1, 2, 56, 56) → (2, 56, 56)
    ab_predicted = ab_predicted[0, :, :, :]

    # ── Step 4: Transpose dari NCHW ke HWC ──────────────────────────────────
    # Dari (2, 56, 56) → (56, 56, 2)
    # OpenCV dan numpy bekerja dalam format HWC (Height, Width, Channel)
    ab_predicted = ab_predicted.transpose((1, 2, 0))


    return ab_predicted
