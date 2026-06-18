import cv2
import numpy as np


# ── FUNGSI 1: Boost Saturasi Warna ────────────────────────────────────────────
def boost_saturation(img_bgr: np.ndarray,
                     factor: float = 1.5) -> np.ndarray:
    """
    Meningkatkan intensitas warna (saturasi) gambar berwarna.

    Mengapa perlu boost saturasi?
        Model Zhang CNN dirancang untuk menghasilkan warna yang "safe" —
        lebih memilih warna pastel/pudar daripada risiko menghasilkan warna
        yang salah. Akibatnya, hasil kolorisasi sering terlihat kurang vivid.
        Boost saturasi membantu mengembalikan kecerahan warna.

    Kenapa pakai HSV, bukan langsung RGB?
        Ruang warna HSV (Hue, Saturation, Value) memisahkan:
            H = jenis warna (merah, hijau, biru, dll.)
            S = kekuatan/intensitas warna (0=abu-abu, 255=sangat jenuh)
            V = kecerahan piksel
        Dengan mengubah hanya channel S, kita bisa menaikkan intensitas warna
        TANPA mengubah jenis warnanya (H) atau kecerahannya (V).

    Args:
        img_bgr (np.ndarray): Gambar berwarna BGR, dtype uint8
        factor  (float)     : Pengali saturasi. 1.0 = tidak berubah,
                              1.5 = 50% lebih jenuh, 2.0 = 2× lebih jenuh

    Returns:
        np.ndarray: Gambar dengan saturasi yang ditingkatkan, dtype uint8
    """
    # ── Step 1: Konversi BGR → HSV ──────────────────────────────────────────
    # Menggunakan float32 agar perkalian tidak overflow
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # ── Step 2: Kalikan channel S (Saturation) dengan factor ────────────────
    # Channel S ada di index 1: img_hsv[:, :, 1]
    # Range S dalam HSV uint8 adalah [0, 255]
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * factor

    # ── Step 3: Clip nilai S agar tidak melebihi 255 ────────────────────────
    # Nilai di atas 255 akan menyebabkan overflow saat konversi ke uint8
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)

    # ── Step 4: Konversi kembali ke uint8 lalu HSV → BGR ────────────────────
    img_hsv_uint8 = img_hsv.astype(np.uint8)
    img_saturated = cv2.cvtColor(img_hsv_uint8, cv2.COLOR_HSV2BGR)


    return img_saturated


# ── FUNGSI 2: Bilateral Filter ────────────────────────────────────────────────
def apply_bilateral_filter(img_bgr: np.ndarray,
                           d: int = 9,
                           sigma_color: float = 75,
                           sigma_space: float = 75) -> np.ndarray:
    """
    Menerapkan bilateral filter untuk menghaluskan 'color bleeding'
    (warna yang bocor melewati tepi objek) sambil tetap menjaga
    ketajaman tepi gambar.

    Bilateral Filter vs Gaussian Filter:
        Gaussian Filter  → menghaluskan SEMUA piksel secara merata,
                           termasuk tepi objek (menjadi buram)
        Bilateral Filter → hanya menghaluskan piksel yang MIRIP warnanya,
                           sehingga tepi tetap tajam tapi artefak warna halus

    Ini sangat penting untuk foto kolorisasi karena AI kadang menghasilkan
    warna yang "meluber" ke area yang berbeda — misalnya warna kulit wajah
    sedikit meluber ke area rambut atau pakaian.

    Parameter:
        d           : Diameter neighborhood piksel (piksel sejauh d akan
                      dipertimbangkan). Semakin besar = lebih halus tapi lambat.
        sigma_color : Toleransi perbedaan warna. Piksel dengan perbedaan warna
                      > sigma_color akan diabaikan (bukan dihaluskan).
        sigma_space : Toleransi jarak spasial. Piksel terdekat mendapat bobot lebih.

    Args:
        img_bgr     (np.ndarray): Gambar BGR uint8
        d           (int)       : Diameter filter neighborhood, default 9
        sigma_color (float)     : Toleransi warna, default 75
        sigma_space (float)     : Toleransi spasial, default 75

    Returns:
        np.ndarray: Gambar yang sudah difilter, dtype uint8
    """
    img_filtered = cv2.bilateralFilter(
        img_bgr,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )


    return img_filtered


# ── FUNGSI 3: Blending dengan Gambar Asli ─────────────────────────────────────
def blend_with_original(img_colored: np.ndarray,
                        img_bw: np.ndarray,
                        alpha: float = 0.93) -> np.ndarray:
    """
    Menggabungkan gambar berwarna hasil AI dengan versi hitam-putih asli
    menggunakan operasi weighted blending.

    Rumus blending:
        output = α × img_colored + (1 - α) × img_bw

    Mengapa melakukan blending?
        Nilai alpha mendekati 1.0 (seperti 0.93) memberikan warna yang kuat
        sekaligus mempertahankan sedikit karakter asli foto hitam-putih.
        Ini juga membantu 'grounding' warna — area yang seharusnya hitam/putih
        tidak akan terlalu diberi warna artificial oleh AI.

    Kontrol alpha:
        alpha = 1.00 → 100% warna dari AI (kadang terlalu artificial)
        alpha = 0.93 → 93% warna AI + 7% grayscale (lebih natural, DEFAULT)
        alpha = 0.70 → 70% warna AI (efek artistik semi-colored)

    Args:
        img_colored (np.ndarray): Gambar berwarna hasil kolorisasi AI
        img_bw      (np.ndarray): Gambar hitam-putih asli (BGR 3-channel)
        alpha       (float)     : Bobot untuk gambar berwarna, range [0.0, 1.0]

    Returns:
        np.ndarray: Gambar hasil blend, dtype uint8
    """
    # cv2.addWeighted() menghitung:
    #   dst = src1 × alpha + src2 × beta + gamma
    # Parameter:
    #   src1  : gambar berwarna
    #   alpha : bobot gambar berwarna
    #   src2  : gambar hitam-putih
    #   beta  : bobot gambar hitam-putih = (1 - alpha)
    #   gamma : nilai tambahan konstan (0 = tidak ada)
    img_blended = cv2.addWeighted(
        img_colored, alpha,
        img_bw,      1.0 - alpha,
        0
    )


    return img_blended


# ── FUNGSI 4: Simpan Gambar Perbandingan ──────────────────────────────────────
def create_comparison(img_bw: np.ndarray,
                      img_final: np.ndarray) -> np.ndarray:
    """
    Membuat gambar perbandingan side-by-side antara foto hitam-putih
    dan foto berwarna hasil kolorisasi.

    Berguna untuk presentasi dan laporan — memperlihatkan perbedaan
    sebelum dan sesudah kolorisasi secara visual dalam satu gambar.

    Args:
        img_bw    (np.ndarray): Gambar hitam-putih asli (BGR)
        img_final (np.ndarray): Gambar berwarna final

    Returns:
        np.ndarray: Gambar perbandingan 2:1 (lebar 2× dari asli)
    """
    h, w = img_bw.shape[:2]

    # Buat gambar hitam-putih 3-channel jika belum
    if len(img_bw.shape) == 2:
        img_bw_3ch = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2BGR)
    else:
        img_bw_3ch = img_bw

    # Tambahkan label teks pada masing-masing gambar
    bw_labeled = img_bw_3ch.copy()
    colored_labeled = img_final.copy()

    # Label "SEBELUM" di pojok kiri atas gambar kiri
    cv2.putText(bw_labeled, 'SEBELUM (B&W)',
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bw_labeled, 'SEBELUM (B&W)',
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (30, 30, 30), 1, cv2.LINE_AA)

    # Label "SESUDAH" di pojok kiri atas gambar kanan
    cv2.putText(colored_labeled, 'SESUDAH (Berwarna - Zhang CNN)',
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(colored_labeled, 'SESUDAH (Berwarna - Zhang CNN)',
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (20, 80, 20), 1, cv2.LINE_AA)

    # Gabungkan secara horizontal dengan np.hstack
    comparison = np.hstack([bw_labeled, colored_labeled])


    return comparison
