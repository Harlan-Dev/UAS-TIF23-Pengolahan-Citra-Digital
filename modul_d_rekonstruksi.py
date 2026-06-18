import cv2
import numpy as np


# ── FUNGSI 1: Resize Channel AB ke Ukuran Asli ────────────────────────────────
def resize_ab_to_original(ab_predicted: np.ndarray,
                          original_shape: tuple) -> np.ndarray:
    """
    Mengubah ukuran channel AB hasil prediksi dari 56×56 kembali ke
    ukuran gambar asli (H × W).

    Model CNN mengeluarkan prediksi pada resolusi 56×56 karena adanya
    operasi pooling di dalam arsitektur. Sebelum digabungkan dengan
    channel L asli, AB harus dikembalikan ke resolusi penuh.

    Metode interpolasi:
        cv2.INTER_CUBIC dipilih karena memberikan hasil transisi warna
        yang lebih halus (bicubic interpolation) dibandingkan INTER_LINEAR.
        Ini penting agar tidak ada efek "kotak-kotak" (blocking artifact)
        pada gambar berwarna akhir.

    Args:
        ab_predicted  (np.ndarray): Channel AB dari model, shape (56, 56, 2)
        original_shape (tuple)    : (H, W) ukuran gambar asli

    Returns:
        np.ndarray: Channel AB yang sudah diresize, shape (H, W, 2)
    """
    h_orig, w_orig = original_shape

    # Resize AB dari 56×56 → ukuran asli
    # Catatan: cv2.resize() menerima (width, height) bukan (height, width)
    ab_resized = cv2.resize(ab_predicted, (w_orig, h_orig),
                            interpolation=cv2.INTER_CUBIC)


    return ab_resized


# ── FUNGSI 2: Gabungkan Channel L + A + B ─────────────────────────────────────
def combine_lab_channels(L_channel: np.ndarray,
                         ab_resized: np.ndarray) -> np.ndarray:
    """
    Menggabungkan channel L (dari foto asli) dengan channel A dan B
    (dari prediksi AI) menjadi gambar LAB yang lengkap.

    Strategi 'L asli + AB prediksi':
        Dengan menggunakan L asli (bukan L yang diresize ke 224×224),
        kita mempertahankan semua detail kecerahan gambar, termasuk
        tekstur halus dan detail tepi yang mungkin hilang saat resize.
        AI hanya berkontribusi pada bagian warna (A dan B).

    Proses np.concatenate():
        L_channel : shape (H, W)   → perlu ditambah dimensi: (H, W, 1)
        ab_resized: shape (H, W, 2)
        Hasil gabungan: (H, W, 3)  → LAB lengkap

    Args:
        L_channel  (np.ndarray): Channel L asli, shape (H, W), range [0, 100]
        ab_resized (np.ndarray): Channel AB prediksi, shape (H, W, 2)

    Returns:
        np.ndarray: Gambar LAB lengkap, shape (H, W, 3)
    """
    # Tambahkan dimensi channel ke L agar bisa digabungkan
    # (H, W) → (H, W, 1) menggunakan np.newaxis
    L_with_channel = L_channel[:, :, np.newaxis]

    # Gabungkan sepanjang sumbu channel (axis=2)
    # Urutan: L dulu, kemudian A, kemudian B
    img_lab_result = np.concatenate([L_with_channel, ab_resized], axis=2)


    return img_lab_result


# ── FUNGSI 3: Konversi LAB ke BGR ─────────────────────────────────────────────
def convert_lab_to_bgr(img_lab_result: np.ndarray) -> np.ndarray:
    """
    Mengkonversi gambar dari ruang warna LAB kembali ke BGR
    yang merupakan format standar untuk ditampilkan dan disimpan.

    Proses ini adalah kebalikan dari konversi yang dilakukan di Modul A.
    OpenCV secara internal menggunakan matriks transformasi warna CIE 1931
    untuk konversi ini.

    Langkah-langkah yang dilakukan fungsi ini:
        1. Clip nilai ke range yang valid untuk LAB float32
        2. Konversi LAB → BGR (range [0, 1] untuk float32 input)
        3. Clip hasil ke [0, 1] untuk menghilangkan nilai negatif/overflow
        4. Skala dari [0, 1] → [0, 255] dan konversi ke uint8

    Args:
        img_lab_result (np.ndarray): Gambar LAB lengkap, shape (H, W, 3), float32

    Returns:
        np.ndarray: Gambar berwarna format BGR, dtype uint8, range [0, 255]
    """
    # ── Step 1: Pastikan tipe data float32 ──────────────────────────────────
    img_lab_float = img_lab_result.astype(np.float32)

    # ── Step 2: Konversi LAB (float32, L=[0,100]) → BGR (float32, [0,1]) ───
    img_bgr_float = cv2.cvtColor(img_lab_float, cv2.COLOR_LAB2BGR)

    # ── Step 3: Clip nilai ke range valid [0, 1] ────────────────────────────
    # Beberapa piksel mungkin memiliki nilai sedikit di luar [0,1]
    # akibat ketidakakuratan prediksi AI atau rounding error
    img_bgr_clipped = np.clip(img_bgr_float, 0.0, 1.0)

    # ── Step 4: Skala [0, 1] → [0, 255] dan konversi ke uint8 ──────────────
    # Perkalian 255 mengubah range float ke range piksel standar
    img_bgr = (img_bgr_clipped * 255).astype(np.uint8)


    return img_bgr
