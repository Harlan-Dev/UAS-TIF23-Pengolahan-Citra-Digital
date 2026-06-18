"""
╔══════════════════════════════════════════════════════════╗
║  DOWNLOAD MODEL — Unduh File Model Zhang CNN             ║
║  Jalankan SEKALI sebelum menggunakan main.py             ║
╚══════════════════════════════════════════════════════════╝

File yang akan diunduh:
  1. colorization_deploy_v2.prototxt   (~10 KB)  — arsitektur model
  2. pts_in_hull.npy                   (~5 KB)   — cluster warna AB
  3. colorization_release_v2.caffemodel (~125 MB) — bobot model (BESAR!)

Cara pakai:
    python download_model.py

Sumber:
  Zhang, R., Isola, P., & Efros, A.A. (2016). ECCV 2016.
  https://github.com/richzhang/colorization
"""

import urllib.request
import os
import sys


# ── Konfigurasi URL dan Path ───────────────────────────────────────────────────
MODEL_DIR = 'models'

FILES = {
    'colorization_deploy_v2.prototxt': (
        'https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_deploy_v2.prototxt',
        '10 KB'
    ),
    'pts_in_hull.npy': (
        'https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/pts_in_hull.npy',
        '5 KB'
    ),
    'colorization_release_v2.caffemodel': (
        'https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel',
        '~125 MB'
    ),
}


# ── Progress Bar Sederhana ─────────────────────────────────────────────────────
def show_progress(block_count, block_size, total_size):
    """Menampilkan progress bar saat mengunduh file besar."""
    if total_size > 0:
        downloaded = block_count * block_size
        percent = min(100, downloaded * 100 // total_size)
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        mb_done  = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f'\r  [{bar}] {percent:3d}%  {mb_done:.1f}/{mb_total:.1f} MB',
              end='', flush=True)


# ── Fungsi Utama Download ──────────────────────────────────────────────────────
def download_all():
    # Buat folder models/ jika belum ada
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("  DOWNLOAD MODEL ZHANG COLORIZATION CNN")
    print("=" * 60)

    all_ok = True

    for filename, (url, size_info) in FILES.items():
        save_path = os.path.join(MODEL_DIR, filename)

        # Lewati jika file sudah ada
        if os.path.exists(save_path):
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"\n✅ {filename}")
            print(f"   Sudah ada ({file_size_mb:.1f} MB), dilewati.")
            continue

        print(f"\n⬇️  {filename} ({size_info})")
        print(f"   URL: {url[:60]}...")

        try:
            urllib.request.urlretrieve(url, save_path, reporthook=show_progress)
            print()  # Newline setelah progress bar
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"   ✅ Berhasil! ({file_size_mb:.1f} MB)")

        except Exception as e:
            print(f"\n   ❌ GAGAL mengunduh: {e}")

            # Jika file parsial ada, hapus
            if os.path.exists(save_path):
                os.remove(save_path)

            if filename == 'colorization_release_v2.caffemodel':
                print("\n   ⚠️  CATATAN: File caffemodel (125 MB) gagal diunduh.")
                print("   Coba unduh manual dari salah satu link berikut:")
                print("   1. https://github.com/richzhang/colorization (lihat README)")
                print("   2. Cari 'colorization_release_v2.caffemodel' di GitHub")
                print("   3. Letakkan file yang diunduh ke folder 'models/'")

            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ Semua file model berhasil diunduh!")
        print("  Sekarang jalankan: python main.py --input foto.jpg --show")
    else:
        print("  ⚠️  Ada file yang gagal diunduh. Cek pesan error di atas.")
    print("=" * 60)


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    download_all()
