import cv2
import numpy as np


# ── FUNGSI 1: Memuat Gambar ────────────────────────────────────────────────────
def load_and_validate(image_path: str) -> np.ndarray:
    """
    Memuat gambar dari path yang diberikan dan memvalidasi bahwa
    file berhasil dibaca oleh OpenCV.

    Args:
        image_path (str): Path/lokasi file gambar input

    Returns:
        np.ndarray: Gambar dalam format BGR (format default OpenCV)

    Raises:
        FileNotFoundError: Jika file tidak ditemukan atau tidak bisa dibaca
    """
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(
            f"Gambar tidak ditemukan atau tidak bisa dibaca: '{image_path}'"
        )

    return img


# ── FUNGSI 2: Konversi ke Hitam-Putih (Grayscaling) ───────────────────────────
def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Mengkonversi gambar ke hitam-putih (grayscale).
    Hasil dikembalikan dalam format BGR 3-channel agar kompatibel
    dengan langkah selanjutnya.

    Args:
        img (np.ndarray): Gambar input format BGR

    Returns:
        np.ndarray: Gambar hitam-putih dalam format BGR (3 channel)
    """
    if len(img.shape) == 2 or img.shape[2] == 1:
        # Gambar sudah grayscale, konversi langsung ke BGR 3-channel
        img_bw = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2BGR)
    else:
        # BGR → GRAY (1 channel) → BGR (3 channel)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return img_bw


# ── FUNGSI 3: Konversi ke Ruang Warna LAB ─────────────────────────────────────
def convert_to_lab(img_bw: np.ndarray) -> np.ndarray:
    """
    Mengkonversi gambar dari ruang warna BGR ke ruang warna LAB (CIELAB).
    LAB memisahkan kecerahan (L) dari informasi warna (A, B) sehingga
    AI hanya perlu memprediksi A dan B.

    Args:
        img_bw (np.ndarray): Gambar hitam-putih format BGR

    Returns:
        np.ndarray: Gambar dalam ruang warna LAB, dtype float32
    """
    # Normalisasi ke [0, 1] sebelum konversi agar OpenCV menggunakan skala float yang benar
    img_float = img_bw.astype(np.float32) / 255.0
    img_lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)

    return img_lab


# ── FUNGSI 4: Ekstraksi Channel L ─────────────────────────────────────────────
def extract_l_channel(img_lab: np.ndarray) -> np.ndarray:
    """
    Mengekstrak channel L (kecerahan) dari gambar LAB.
    Channel L inilah yang menjadi input ke model Zhang CNN.

    Args:
        img_lab (np.ndarray): Gambar dalam ruang warna LAB

    Returns:
        np.ndarray: Channel L saja, shape (H, W)
    """
    L = img_lab[:, :, 0]

    return L
