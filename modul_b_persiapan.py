import cv2
import numpy as np


# ── FUNGSI 1: Resize ke Ukuran Input Model ────────────────────────────────────
def resize_for_model(L_channel: np.ndarray,
                     target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Mengubah ukuran channel L ke dimensi yang diterima oleh model Zhang CNN.
    Model Zhang dilatih menggunakan gambar berukuran 224×224 piksel,
    sehingga setiap input harus diubah ke ukuran tersebut terlebih dahulu.

    Kenapa 224×224?
        Ini adalah standar umum untuk model CNN berbasis ImageNet.
        Model Zhang CNN menggunakan arsitektur yang terinspirasi dari VGG,
        yang mengharapkan input 224×224.

    Metode interpolasi:
        cv2.INTER_CUBIC digunakan karena menghasilkan kualitas yang lebih
        halus dibandingkan INTER_LINEAR untuk operasi resize-up.

    Args:
        L_channel  (np.ndarray): Channel L, shape (H, W)
        target_size (tuple)    : Ukuran target (width, height), default (224, 224)

    Returns:
        np.ndarray: Channel L yang sudah diresize ke target_size
    """
    h_orig, w_orig = L_channel.shape
    L_resized = cv2.resize(L_channel, target_size, interpolation=cv2.INTER_CUBIC)


    return L_resized


# ── FUNGSI 2: Normalisasi Channel L ───────────────────────────────────────────
def normalize_l_channel(L_resized: np.ndarray) -> np.ndarray:
    """
    Menormalisasi nilai channel L dengan mengurangi nilai mean (50).
    Teknik ini disebut 'mean subtraction' atau 'zero-centering'.

    Mengapa dikurangi 50?
        Channel L memiliki range [0, 100].
        Nilai tengah idealnya adalah 50 (kecerahan sedang).
        Dengan mengurangi 50, distribusi bergeser ke [-50, 50],
        sehingga nilai positif = lebih terang dari rata-rata,
        dan negatif = lebih gelap dari rata-rata.

        Ini membantu proses gradient descent saat model melakukan
        inferensi lebih stabil karena input tersebar di sekitar nol.

    Args:
        L_resized (np.ndarray): Channel L yang sudah diresize

    Returns:
        np.ndarray: Channel L yang sudah dinormalisasi, range [-50, 50]
    """
    # Kurangi nilai mean 50 (nilai tengah range L = [0, 100])
    L_normalized = L_resized - 50


    return L_normalized


# ── FUNGSI 3: Pembentukan Blob untuk DNN ──────────────────────────────────────
def prepare_blob(L_normalized: np.ndarray) -> np.ndarray:
    """
    Mengubah array numpy menjadi 'blob' yang siap diproses oleh
    modul cv2.dnn (Deep Neural Network) OpenCV.

    Apa itu blob?
        Blob (Binary Large OBject) dalam konteks DNN adalah tensor 4D
        dengan format NCHW:
            N = jumlah gambar dalam satu batch (= 1 untuk inferensi tunggal)
            C = jumlah channel (= 1 karena hanya channel L)
            H = tinggi gambar (= 224)
            W = lebar gambar (= 224)

        Jadi shape blob: (1, 1, 224, 224)

    Args:
        L_normalized (np.ndarray): Channel L yang sudah dinormalisasi

    Returns:
        np.ndarray: Blob 4D siap untuk input ke model DNN
    """
    # cv2.dnn.blobFromImage() mengkonversi gambar 2D ke tensor 4D NCHW
    # Parameter:
    #   image     : array input (H, W)
    #   scalefactor: pengali nilai piksel (1.0 = tidak diubah)
    #   size       : (width, height) — sudah 224×224, tidak perlu resize lagi
    #   mean       : nilai mean untuk subtraksi (sudah dilakukan manual)
    #   swapRB     : apakah swap R dan B channel (tidak perlu untuk grayscale)
    blob = cv2.dnn.blobFromImage(
        L_normalized,
        scalefactor=1.0,
        size=(224, 224),
        mean=(0, 0, 0),
        swapRB=False
    )


    return blob
