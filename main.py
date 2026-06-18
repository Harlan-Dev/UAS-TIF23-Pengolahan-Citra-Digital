import cv2
import numpy as np
import argparse
import os
import time

# ── Import semua modul (satu modul per anggota) ───────────────────────────────
from modul_a_preprocessing  import (load_and_validate, convert_to_grayscale,
                                    convert_to_lab,    extract_l_channel)
from modul_b_persiapan      import (resize_for_model, normalize_l_channel,
                                    prepare_blob)
from modul_c_kolorisasi     import (load_model, predict_ab_channels)
from modul_d_rekonstruksi   import (resize_ab_to_original, combine_lab_channels,
                                    convert_lab_to_bgr)
from modul_e_postprocessing import (boost_saturation, apply_bilateral_filter,
                                    blend_with_original, create_comparison)


# ── Parsing Argumen Command Line ───────────────────────────────────────────────
def parse_arguments():
    """
    Memproses argumen yang diberikan pengguna saat menjalankan program.
    Menggunakan modul argparse dari Python standard library.

    Jika tidak ada argumen yang diberikan, program akan membuka GUI.
    """
    parser = argparse.ArgumentParser(
        description='Kolorisasi Foto Hitam-Putih dengan AI (Zhang CNN)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python main.py                          → Buka GUI (direkomendasikan)
  python main.py --input foto_jadul.jpg   → Mode CLI
  python main.py --input foto_jadul.jpg --output hasil.jpg --saturation 1.6
  python main.py --input foto_jadul.jpg --show
        """
    )

    # Argumen opsional (bukan required lagi — tanpa argumen = buka GUI)
    parser.add_argument(
        '--input', '-i',
        required=False,
        default=None,
        help='Path file gambar hitam-putih yang akan diwarnai (.jpg, .png, dll.)\n'
             'Jika tidak diisi, program akan membuka GUI.'
    )

    # Argumen opsional
    parser.add_argument(
        '--output', '-o',
        default='output_berwarna.jpg',
        help='Path untuk menyimpan hasil (default: output_berwarna.jpg)'
    )
    parser.add_argument(
        '--saturation', '-s',
        type=float,
        default=1.4,
        help='Faktor boost saturasi warna (default: 1.4). Coba 1.2–2.0'
    )
    parser.add_argument(
        '--blend', '-b',
        type=float,
        default=0.93,
        help='Kekuatan warna AI (0.0–1.0, default: 0.93). Semakin kecil = lebih grayscale'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Tampilkan jendela perbandingan sebelum/sesudah'
    )
    parser.add_argument(
        '--save-comparison',
        action='store_true',
        help='Simpan gambar perbandingan side-by-side'
    )

    parser.add_argument(
        '--web',
        action='store_true',
        help='Buka Web GUI di browser (Flask) — direkomendasikan'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5050,
        help='Port untuk Web GUI (default: 5050)'
    )

    return parser.parse_args()


# ── Fungsi Utama ───────────────────────────────────────────────────────────────
def main():
    args = parse_arguments()

    # ── Jika tidak ada --input → Buka Web GUI ──────────────
    if args.input is None or args.web:
        print("\n  Membuka Web GUI Kolorisasi...")
        try:
            from app import launch_web
            launch_web(port=args.port)
        except ImportError as e:
            print(f"[ERROR] Flask tidak tersedia: {e}")
            print("  Install: pip3 install flask")
        return

    waktu_mulai = time.time()

    # Header
    print("\n" + "=" * 60)
    print("  SISTEM KOLORISASI FOTO HITAM-PUTIH BERBASIS AI")
    print("  Framework: Zhang Colorization CNN (ECCV 2016)")
    print("=" * 60)
    print(f"  Input      : {args.input}")
    print(f"  Output     : {args.output}")
    print(f"  Saturasi   : {args.saturation}×")
    print(f"  Blend alpha: {args.blend}")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════
    #  MODUL A — Preprocessing
    # ══════════════════════════════════════════════════════════
    print("\n[MODUL A] Preprocessing gambar...")

    # Muat gambar dari disk
    img_original = load_and_validate(args.input)

    # Konversi ke grayscale (hitam-putih) jika belum
    img_bw = convert_to_grayscale(img_original)

    # Konversi dari BGR ke ruang warna LAB
    img_lab = convert_to_lab(img_bw)

    # Ekstrak channel L (kecerahan saja) — ini input ke AI
    L_channel = extract_l_channel(img_lab)

    print(f"  [OK] Modul A selesai\n")

    # ══════════════════════════════════════════════════════════
    #  MODUL B — Persiapan Input CNN
    # ══════════════════════════════════════════════════════════
    print("[MODUL B] Mempersiapkan input untuk model Zhang CNN...")

    # Resize channel L dari ukuran asli ke 224×224 (ukuran input model)
    L_resized = resize_for_model(L_channel, target_size=(224, 224))

    # Normalisasi: kurangi nilai mean 50 (centering)
    L_normalized = normalize_l_channel(L_resized)

    # Bentuk blob tensor 4D: (1, 1, 224, 224) untuk input ke DNN
    blob = prepare_blob(L_normalized)

    print(f"  [OK] Modul B selesai\n")

    # ══════════════════════════════════════════════════════════
    #  MODUL C — Inferensi AI (Zhang CNN)
    # ══════════════════════════════════════════════════════════
    print("[MODUL C] Memuat dan menjalankan model Zhang CNN (AI)...")

    # Muat model dari file .prototxt dan .caffemodel
    net = load_model()

    # Jalankan forward pass → prediksi channel A dan B
    ab_predicted = predict_ab_channels(net, blob)

    print(f"  [OK] Modul C selesai\n")

    # ══════════════════════════════════════════════════════════
    #  MODUL D — Rekonstruksi Gambar Berwarna
    # ══════════════════════════════════════════════════════════
    print("[MODUL D] Merekonstruksi gambar berwarna...")

    h, w = img_original.shape[:2]

    # Resize channel AB dari 56×56 kembali ke ukuran asli
    ab_resized = resize_ab_to_original(ab_predicted, original_shape=(h, w))

    # Gabungkan L (asli) + AB (prediksi AI) → gambar LAB lengkap
    img_lab_result = combine_lab_channels(L_channel, ab_resized)

    # Konversi gambar LAB → BGR (format standar OpenCV)
    img_colored = convert_lab_to_bgr(img_lab_result)

    print(f"  [OK] Modul D selesai\n")

    # ══════════════════════════════════════════════════════════
    #  MODUL E — Post-Processing
    # ══════════════════════════════════════════════════════════
    print("[MODUL E] Post-processing: saturasi, filter, blending...")

    # Tingkatkan saturasi agar warna lebih vivid (tidak terlalu pucat)
    img_saturated = boost_saturation(img_colored, factor=args.saturation)

    # Bilateral filter: haluskan artefak warna tanpa memburamkan tepi
    img_filtered = apply_bilateral_filter(img_saturated, d=9,
                                          sigma_color=75, sigma_space=75)

    # Blend gambar berwarna dengan versi B&W asli untuk hasil lebih natural
    img_final = blend_with_original(img_filtered, img_bw, alpha=args.blend)

    print(f"  [OK] Modul E selesai\n")

    # ══════════════════════════════════════════════════════════
    #  SIMPAN HASIL
    # ══════════════════════════════════════════════════════════
    # Buat folder output jika perlu
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Simpan gambar berwarna final
    cv2.imwrite(args.output, img_final)

    # Simpan gambar perbandingan jika diminta
    if args.save_comparison:
        comparison = create_comparison(img_bw, img_final)
        comp_path  = args.output.replace('.jpg', '_comparison.jpg') \
                               .replace('.png', '_comparison.png')
        cv2.imwrite(comp_path, comparison)
        print(f"  → Gambar perbandingan disimpan: {comp_path}")

    # Hitung waktu proses
    durasi = time.time() - waktu_mulai

    print("=" * 60)
    print(f"  ✅ KOLORISASI SELESAI!")
    print(f"  Output   : {args.output}")
    print(f"  Durasi   : {durasi:.2f} detik")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════
    #  TAMPILKAN HASIL (OPSIONAL)
    # ══════════════════════════════════════════════════════════
    if args.show:
        print("\n  Menampilkan perbandingan... (tekan tombol apapun untuk keluar)")
        comparison = create_comparison(img_bw, img_final)

        # Resize untuk tampilan layar jika terlalu besar
        max_display_w = 1200
        if comparison.shape[1] > max_display_w:
            scale = max_display_w / comparison.shape[1]
            new_h = int(comparison.shape[0] * scale)
            comparison = cv2.resize(comparison, (max_display_w, new_h))

        cv2.imshow('Kolorisasi AI: SEBELUM (kiri) | SESUDAH (kanan)', comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
