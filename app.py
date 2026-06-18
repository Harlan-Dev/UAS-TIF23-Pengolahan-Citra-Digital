import os, uuid, base64, json, threading, time, io, sys
from flask import Flask, request, jsonify, render_template_string, Response

import cv2
import numpy as np

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ── Import pipeline kolorisasi ─────────────────────────────────────────────────
from modul_a_preprocessing  import (load_and_validate, convert_to_grayscale,
                                    convert_to_lab,    extract_l_channel)
from modul_b_persiapan      import (resize_for_model, normalize_l_channel,
                                    prepare_blob)
from modul_c_kolorisasi     import (load_model, predict_ab_channels)
from modul_d_rekonstruksi   import (resize_ab_to_original, combine_lab_channels,
                                    convert_lab_to_bgr)
from modul_e_postprocessing import (boost_saturation, apply_bilateral_filter,
                                    blend_with_original, create_comparison)

app     = Flask(__name__)
UPLOAD  = "temp_web_uploads"
SAMPLES = "samples"
os.makedirs(UPLOAD,  exist_ok=True)
os.makedirs(SAMPLES, exist_ok=True)

# ── Session store: { session_id: {img_bw, img_colorized, img_edited, progress, status} }
sessions = {}
_net_cache = None   # cache model agar tidak reload tiap request
_net_lock  = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def bgr_to_b64(img_bgr: np.ndarray, quality: int = 88) -> str:
    """Encode gambar BGR numpy array ke base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img_bgr,
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def resize_preview(img_bgr: np.ndarray, max_dim: int = 1400) -> np.ndarray:
    """Resize gambar untuk preview agar tidak terlalu besar."""
    h, w = img_bgr.shape[:2]
    if max(h, w) <= max_dim:
        return img_bgr
    scale = max_dim / max(h, w)
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


# ══════════════════════════════════════════════════════════════════════════════
#  APPLY EDITOR ADJUSTMENTS
# ══════════════════════════════════════════════════════════════════════════════
def apply_adjustments(img_colorized: np.ndarray, sv: dict) -> np.ndarray:
    """Terapkan semua slider adjustment ke gambar hasil kolorisasi."""
    img = img_colorized.copy().astype(np.float32)

    # Exposure
    img = img * (2 ** (sv.get("exposure", 0) / 100.0))

    # Brightness
    img = img + sv.get("brightness", 0) * 1.5

    # Contrast
    con = sv.get("contrast", 0) / 100.0
    img = (img - 127.5) * (1 + con) + 127.5

    # Brilliance
    brilliance = sv.get("brilliance", 0) / 100.0
    if brilliance != 0:
        img_norm = np.clip(img / 255.0, 0.0, 1.0)
        sb = np.power(img_norm + 1e-6, 1 - 0.5 * brilliance) * 255.0
        hb = np.power(img_norm + 1e-6, 1 + 0.3 * brilliance) * 255.0
        img = sb * (1 - img_norm) + hb * img_norm

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Highlights & Shadows
    hl = sv.get("highlights", 0) / 100.0
    sh = sv.get("shadows",    0) / 100.0
    if hl != 0 or sh != 0:
        lut = np.arange(256, dtype=np.float32)
        for i in range(256):
            t = i / 255.0
            lut[i] = (i + hl * 80 * max(0, (t - 0.5) * 2)
                        - sh * 80 * max(0, (0.5 - t) * 2))
        img = np.clip(lut, 0, 255).astype(np.uint8)[img]

    # Black Point
    bp = sv.get("black_point", 0) / 100.0
    if bp != 0:
        lut = np.clip(np.arange(256, dtype=np.float32) - bp * 30, 0, 255).astype(np.uint8)
        img = lut[img]

    # Saturation & Vibrance
    sat = 1 + sv.get("saturation", 0) / 100.0
    vib = sv.get("vibrance", 0) / 100.0
    if sat != 1.0 or vib != 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        s = hsv[:, :, 1]
        if vib != 0:
            s = s * (1 + vib * (1.0 - s / 255.0))
        hsv[:, :, 1] = np.clip(s * sat, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Warmth & Tint
    warmth = sv.get("warmth", 0) / 100.0
    tint   = sv.get("tint",   0) / 100.0
    if warmth != 0 or tint != 0:
        b, g, r = cv2.split(img.astype(np.float32))
        r += warmth * 20; b -= warmth * 20
        g += tint * 15;   r -= tint * 8;  b -= tint * 8
        img = cv2.merge([np.clip(b,0,255),
                         np.clip(g,0,255),
                         np.clip(r,0,255)]).astype(np.uint8)

    # Sharpness
    sharp = sv.get("sharpness", 0) / 100.0
    if sharp > 0:
        k = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float32)
        sharpened = cv2.filter2D(img, -1, k)
        img = cv2.addWeighted(img, 1 - sharp * 0.5, sharpened, sharp * 0.5, 0)

    # Definition
    defn = sv.get("definition", 0) / 100.0
    if defn > 0:
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        img  = cv2.addWeighted(img, 1 + defn * 0.7, blur, -defn * 0.7, 0)

    # Noise Reduction
    nr = sv.get("noise_reduction", 0) / 100.0
    if nr > 0:
        hs = max(1, int(nr * 10))
        img = cv2.fastNlMeansDenoisingColored(img, None, hs, hs, 7, 21)

    # Vignette
    vig = sv.get("vignette", 0) / 100.0
    if vig != 0:
        rows, cols = img.shape[:2]
        sigma = 0.5 / max(abs(vig), 0.01)
        X = cv2.getGaussianKernel(cols, sigma * cols)
        Y = cv2.getGaussianKernel(rows, sigma * rows)
        mask = (Y * X.T); mask /= mask.max()
        factor = (1 - vig * (1 - mask) if vig > 0
                  else 1 + abs(vig) * (1 - mask) * 0.5)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] * factor, 0, 255).astype(np.uint8)

    return img


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return generate_page()


@app.route("/upload", methods=["POST"])
def upload():
    """Terima upload gambar, simpan sementara, kembalikan preview."""
    f = request.files.get("image")
    if not f:
        return jsonify(error="No file"), 400

    sid = str(uuid.uuid4())
    path = os.path.join(UPLOAD, f"{sid}_orig.jpg")
    f.save(path)

    img = cv2.imread(path)
    if img is None:
        return jsonify(error="Cannot read image"), 400

    h, w = img.shape[:2]
    prev = resize_preview(img)
    sessions[sid] = {
        "orig_path": path,
        "img_bw":        None,
        "img_colorized": None,
        "img_edited":    None,
        "progress":      0,
        "status":        "uploaded",
    }

    return jsonify(
        session_id=sid,
        preview=bgr_to_b64(prev),
        width=w, height=h,
        filename=f.filename
    )


@app.route("/colorize/<sid>")
def colorize_stream(sid):
    """
    SSE stream — jalankan pipeline kolorisasi dan kirim progress.
    Client menerima event: progress, done, error.
    """
    if sid not in sessions:
        return Response("data: {\"error\":\"session not found\"}\n\n",
                        mimetype="text/event-stream")

    def generate():
        global _net_cache
        sess = sessions[sid]

        def send(evt, data):
            return f"event: {evt}\ndata: {json.dumps(data)}\n\n"

        try:
            img = cv2.imread(sess["orig_path"])

            # Modul A
            yield send("progress", {"pct": 10, "msg": "Modul A: Preprocessing..."})
            img_bw    = convert_to_grayscale(img)
            img_lab   = convert_to_lab(img_bw)
            L_channel = extract_l_channel(img_lab)
            sess["img_bw"] = img_bw

            # Modul B
            yield send("progress", {"pct": 25, "msg": "Modul B: Persiapan CNN..."})
            L_resized    = resize_for_model(L_channel, target_size=(224, 224))
            L_normalized = normalize_l_channel(L_resized)
            blob         = prepare_blob(L_normalized)

            # Modul C — load model
            yield send("progress", {"pct": 40, "msg": "Modul C: Loading model AI (±125 MB)..."})
            with _net_lock:
                if _net_cache is None:
                    _net_cache = load_model()
                net = _net_cache

            yield send("progress", {"pct": 58, "msg": "Modul C: Inferensi Zhang CNN..."})
            ab_predicted = predict_ab_channels(net, blob)

            # Modul D
            yield send("progress", {"pct": 72, "msg": "Modul D: Rekonstruksi warna..."})
            h, w = img.shape[:2]
            ab_resized  = resize_ab_to_original(ab_predicted, (h, w))
            img_lab_res = combine_lab_channels(L_channel, ab_resized)
            img_colored = convert_lab_to_bgr(img_lab_res)

            # Modul E
            yield send("progress", {"pct": 86, "msg": "Modul E: Post-processing..."})
            img_sat  = boost_saturation(img_colored, factor=1.4)
            img_filt = apply_bilateral_filter(img_sat)
            img_fin  = blend_with_original(img_filt, img_bw, alpha=0.93)

            sess["img_colorized"] = img_fin
            sess["img_edited"]    = img_fin.copy()
            sess["status"]        = "done"

            prev_bw  = resize_preview(img_bw)
            prev_col = resize_preview(img_fin)

            yield send("done", {
                "pct": 100,
                "msg": "✅ Kolorisasi selesai!",
                "before": bgr_to_b64(prev_bw),
                "after":  bgr_to_b64(prev_col),
            })

        except Exception as ex:
            sess["status"] = "error"
            yield send("error", {"msg": str(ex)})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/adjust/<sid>", methods=["POST"])
def adjust(sid):
    """Terapkan slider adjustments ke gambar hasil kolorisasi."""
    if sid not in sessions:
        return jsonify(error="session not found"), 404

    sess = sessions[sid]
    if sess.get("img_colorized") is None:
        return jsonify(error="Belum dikolorisasi"), 400

    sv = request.json or {}
    try:
        img_edited = apply_adjustments(sess["img_colorized"], sv)
        sess["img_edited"] = img_edited
        prev = resize_preview(img_edited)
        return jsonify(after=bgr_to_b64(prev))
    except Exception as ex:
        return jsonify(error=str(ex)), 500


@app.route("/download/<sid>/<mode>")
def download(sid, mode):
    """Download hasil: mode = 'result' atau 'comparison'."""
    if sid not in sessions:
        return "Session not found", 404

    sess = sessions[sid]
    if sess.get("img_edited") is None:
        return "Belum ada hasil", 400

    if mode == "comparison":
        img = create_comparison(sess["img_bw"], sess["img_edited"])
        fname = "comparison.jpg"
    else:
        img = sess["img_edited"]
        fname = "colorized.jpg"

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    from flask import send_file
    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/jpeg",
        as_attachment=True,
        download_name=fname
    )


@app.route("/samples")
def list_samples():
    """Daftar semua sample image di folder samples/."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = sorted([
        f for f in os.listdir(SAMPLES)
        if f.lower().endswith(exts) and not f.startswith('.')
    ])
    items = []
    for f in files:
        # Buat label dari nama file: sample_1_rose → Rose
        name = os.path.splitext(f)[0]
        parts = name.split('_')
        label = parts[-1].capitalize() if len(parts) > 1 else name
        items.append({"filename": f, "label": label})
    return jsonify(samples=items)


@app.route("/sample-img/<path:filename>")
def serve_sample(filename):
    """Serve file sample image (full size, untuk upload)."""
    from flask import send_from_directory
    return send_from_directory(SAMPLES, filename)


@app.route("/thumb/<path:filename>")
def serve_thumb(filename):
    """Serve thumbnail kecil (150px) untuk gallery preview."""
    from flask import send_from_directory
    path = os.path.join(SAMPLES, filename)
    if not os.path.exists(path):
        return "Not found", 404
    img = cv2.imread(path)
    if img is None:
        return "Cannot read", 400
    h, w = img.shape[:2]
    target_w = 200
    target_h = int(h * target_w / w)
    thumb = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    from flask import Response
    return Response(buf.tobytes(), mimetype="image/jpeg")


# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLE GALLERY HTML GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_META = [
    ("sample_1_rose.jpg",   "Rose"),
    ("sample_2_gears.png",  "Gears"),
    ("sample_3_bird.png",   "Bird"),
    ("sample_4_eye.png",    "Eye"),
    ("sample_5_aurora.png", "Aurora"),
]

def build_sample_thumbs_html():
    """Buat HTML thumbnail untuk semua sample yang ada di folder samples/."""
    items = []
    for fname, label in SAMPLE_META:
        path = os.path.join(SAMPLES, fname)
        if os.path.exists(path):
            items.append(f'''
      <div class="sample-thumb" title="{label}" onclick="loadSampleImage('{fname}', '{label}')">
        <img src="/thumb/{fname}" alt="{label}" loading="lazy">
        <div class="sample-label">{label}</div>
      </div>''')
    return "\n".join(items)


def generate_page():
    """Generate halaman HTML dengan thumbnail sample yang sudah di-render."""
    thumbs_html = build_sample_thumbs_html()
    page = HTML_TEMPLATE.replace("{{SAMPLE_THUMBS}}", thumbs_html)
    return page


# ══════════════════════════════════════════════════════════════════════════════
#  HTML PAGE (single-file, embed CSS + JS)
# ══════════════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Photo Macro Colorization — TIF23</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0a12;
    --panel:    #12121e;
    --card:     #1a1a2a;
    --hover:    #22223a;
    --border:   #2a2a45;
    --purple:   #8b5cf6;
    --purple2:  #6d28d9;
    --blue:     #3b82f6;
    --green:    #10b981;
    --pink:     #ec4899;
    --text:     #f0f0ff;
    --muted:    #8888aa;
    --dim:      #44445a;
    --radius:   12px;
  }

  html, body {
    height: 100%; background: var(--bg);
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--text); overflow: hidden;
  }

  /* ── LAYOUT ─────────────────────────────────────────────── */
  #app {
    display: grid;
    grid-template-columns: 260px 1fr 270px;
    grid-template-rows: 100vh;
    height: 100vh;
  }

  /* ── LEFT PANEL ─────────────────────────────────────────── */
  #left {
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    padding: 20px 14px; gap: 10px;
    overflow-y: auto;
  }

  .app-title { text-align: center; padding: 4px 0 12px; }
  .app-title h1 {
    font-size: 15px; font-weight: 700;
    background: linear-gradient(135deg, var(--purple), var(--blue));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .app-title p { font-size: 10px; color: var(--muted); margin-top: 3px; }

  .divider { height: 1px; background: var(--border); margin: 2px 0; }

  /* Drop zone */
  #drop-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 20px 10px;
    text-align: center;
    cursor: pointer;
    transition: all .2s;
    background: var(--card);
    position: relative;
  }
  #drop-zone:hover, #drop-zone.drag-over {
    border-color: var(--purple);
    background: #1e1a35;
    transform: scale(1.01);
  }
  #drop-zone .dz-icon { font-size: 32px; margin-bottom: 6px; }
  #drop-zone p { font-size: 11px; color: var(--muted); line-height: 1.5; }
  #drop-zone input { display: none; }

  #file-info {
    font-size: 10px; color: var(--muted);
    text-align: center; line-height: 1.5;
  }

  /* Buttons */
  .btn {
    width: 100%; padding: 9px 12px;
    border: none; border-radius: 8px;
    font-family: inherit; font-size: 12px; font-weight: 600;
    cursor: pointer; transition: all .2s; color: var(--text);
    display: flex; align-items: center; justify-content: center; gap: 6px;
  }
  .btn:hover { filter: brightness(1.15); transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  .btn:disabled { opacity: .4; cursor: not-allowed; transform: none; filter: none; }
  .btn-purple { background: linear-gradient(135deg, var(--purple2), var(--purple)); }
  .btn-blue   { background: linear-gradient(135deg, #1d4ed8, var(--blue)); }
  .btn-green  { background: linear-gradient(135deg, #059669, var(--green)); }
  .btn-dark   { background: var(--hover); border: 1px solid var(--border); }

  /* Progress */
  #status-text { font-size: 10px; color: var(--muted); text-align: center; }
  .progress-bar {
    height: 4px; background: var(--card); border-radius: 2px; overflow: hidden;
  }
  .progress-fill {
    height: 100%; width: 0%;
    background: linear-gradient(90deg, var(--purple2), var(--purple), var(--blue));
    border-radius: 2px;
    transition: width .4s ease;
  }

  /* ── CENTER PANEL ───────────────────────────────────────── */
  #center {
    display: flex; flex-direction: column;
    background: #08080f;
    overflow: hidden;
  }

  #preview-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 20px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    font-size: 11px; font-weight: 600;
  }
  #preview-header .label-before { color: var(--muted); }
  #preview-header .label-mid    { color: var(--dim); }
  #preview-header .label-after  { color: var(--purple); }

  #preview-area {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 2px 1fr;
    overflow: hidden;
  }

  .img-panel {
    display: flex; align-items: center; justify-content: center;
    overflow: hidden; position: relative;
    background: #08080f;
  }
  .img-panel img {
    max-width: 100%; max-height: 100%;
    object-fit: contain; display: block;
    transition: opacity .3s;
  }
  .img-panel .placeholder {
    color: var(--dim); font-size: 14px;
    text-align: center; line-height: 1.8;
    user-select: none;
  }
  .img-panel .corner-label {
    position: absolute; top: 10px; left: 12px;
    font-size: 9px; font-weight: 700; letter-spacing: 1px;
    padding: 3px 8px; border-radius: 4px;
    background: rgba(0,0,0,.6); backdrop-filter: blur(4px);
  }
  .img-panel.before .corner-label { color: var(--muted); }
  .img-panel.after  .corner-label { color: var(--purple); }

  #divider { background: var(--purple); opacity: .6; }

  /* ── RIGHT PANEL ────────────────────────────────────────── */
  #right {
    background: var(--panel);
    border-left: 1px solid var(--border);
    display: flex; flex-direction: column;
    overflow: hidden;
  }

  #editor-header {
    padding: 16px 14px 8px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  #editor-header h2 { font-size: 13px; font-weight: 600; margin-bottom: 2px; }
  #editor-header p  { font-size: 10px; color: var(--muted); }

  #editor-actions {
    display: flex; gap: 6px;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  #editor-actions .btn { font-size: 11px; padding: 7px 10px; }

  #sliders-area {
    flex: 1; overflow-y: auto; padding: 8px 14px 20px;
  }
  #sliders-area::-webkit-scrollbar { width: 4px; }
  #sliders-area::-webkit-scrollbar-track { background: transparent; }
  #sliders-area::-webkit-scrollbar-thumb {
    background: var(--border); border-radius: 2px;
  }

  /* Slider rows */
  .slider-row {
    margin: 10px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .slider-top {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 4px;
  }
  .slider-name { font-size: 11px; font-weight: 500; color: var(--text); }
  .slider-val  {
    font-size: 10px; font-weight: 600; font-family: monospace;
    color: var(--purple); min-width: 32px; text-align: right;
  }

  /* Custom range slider */
  input[type=range] {
    -webkit-appearance: none; appearance: none;
    width: 100%; height: 3px;
    border-radius: 2px;
    outline: none; cursor: pointer;
    background: var(--border);
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 16px; height: 16px; border-radius: 50%;
    background: var(--text); border: 2px solid var(--purple);
    cursor: pointer; transition: all .15s;
    box-shadow: 0 1px 4px rgba(0,0,0,.4);
  }
  input[type=range]::-webkit-slider-thumb:hover {
    background: var(--purple); transform: scale(1.15);
  }
  input[type=range]::-moz-range-thumb {
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--text); border: 2px solid var(--purple);
    cursor: pointer;
  }

  /* ── LOADING OVERLAY ────────────────────────────────────── */
  #loading {
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,.75);
    backdrop-filter: blur(6px);
    z-index: 999;
    flex-direction: column;
    align-items: center; justify-content: center; gap: 16px;
  }
  #loading.show { display: flex; }
  #loading .spinner {
    width: 48px; height: 48px;
    border: 3px solid var(--border);
    border-top-color: var(--purple);
    border-radius: 50%;
    animation: spin .8s linear infinite;
  }
  #loading h3 { font-size: 16px; font-weight: 600; }
  #loading p  { font-size: 12px; color: var(--muted); }
  #loading .load-bar {
    width: 280px; height: 4px;
    background: var(--border); border-radius: 2px; overflow: hidden;
  }
  #loading .load-fill {
    height: 100%; width: 0%;
    background: linear-gradient(90deg, var(--purple2), var(--purple), var(--blue));
    border-radius: 2px; transition: width .4s ease;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── TOAST ──────────────────────────────────────────────── */
  #toast {
    position: fixed; bottom: 24px; left: 50%;
    transform: translateX(-50%) translateY(80px);
    background: var(--card); border: 1px solid var(--border);
    padding: 10px 20px; border-radius: 8px;
    font-size: 12px; font-weight: 500;
    transition: transform .3s ease;
    z-index: 1000;
  }
  #toast.show { transform: translateX(-50%) translateY(0); }

  /* ── MODAL ABOUT US ─────────────────────────────────────── */
  #about-modal {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,.8); backdrop-filter: blur(4px);
    z-index: 2000; align-items: center; justify-content: center;
    opacity: 0; transition: opacity .3s;
  }
  #about-modal.show { display: flex; opacity: 1; }
  .modal-content {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 16px; padding: 24px; width: 90%; max-width: 400px;
    box-shadow: 0 10px 30px rgba(0,0,0,.5);
    transform: translateY(20px); transition: transform .3s;
    position: relative;
  }
  #about-modal.show .modal-content { transform: translateY(0); }
  .modal-close {
    position: absolute; top: 12px; right: 16px;
    font-size: 20px; cursor: pointer; color: var(--muted);
  }
  .modal-close:hover { color: var(--text); }
  .modal-title { font-size: 16px; font-weight: 700; margin-bottom: 8px; color: var(--text); }
  .modal-desc { font-size: 12px; color: var(--muted); line-height: 1.5; margin-bottom: 16px; }
  .team-list { list-style: none; padding: 0; margin: 0; }
  .team-item {
    background: var(--card); border: 1px solid var(--border);
    padding: 10px 12px; border-radius: 8px; margin-bottom: 8px;
    display: flex; align-items: center; gap: 10px;
  }
  .team-icon { font-size: 20px; }
  .team-nim { font-size: 10px; color: var(--purple); font-weight: 600; }
  .team-name { font-size: 12px; font-weight: 500; color: var(--text); }

  /* ── SAMPLE IMAGES GALLERY ─────────────────────────────── */
  .samples-section {
    margin-top: 2px;
  }
  .samples-title {
    font-size: 11px; font-weight: 600; color: var(--text);
    margin-bottom: 2px;
  }
  .samples-subtitle {
    font-size: 9px; color: var(--muted); margin-bottom: 8px;
  }
  .samples-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }
  .sample-thumb {
    position: relative;
    aspect-ratio: 4 / 3;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    border: 2px solid transparent;
    transition: all .25s ease;
    background: var(--card);
  }
  .sample-thumb:hover {
    border-color: var(--purple);
    transform: scale(1.04);
    box-shadow: 0 0 12px rgba(139, 92, 246, .3);
  }
  .sample-thumb:active {
    transform: scale(0.97);
  }
  .sample-thumb img {
    width: 100%; height: 100%;
    object-fit: cover;
    display: block;
    filter: brightness(0.85);
    transition: filter .2s;
  }
  .sample-thumb:hover img {
    filter: brightness(1);
  }
  .sample-thumb .sample-label {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,.75));
    padding: 12px 6px 4px;
    font-size: 9px; font-weight: 600;
    color: #fff; text-align: center;
    letter-spacing: 0.5px;
  }
</style>
</head>
<body>

<div id="app">

  <!-- ── LEFT PANEL ──────────────────────────────────────────── -->
  <aside id="left">
    <div class="app-title">
      <h1>✨ AI Photo Macro Colorization</h1>
      <p>TIF23 — Pengolahan Citra Digital</p>
    </div>

    <div class="divider"></div>

    <!-- Drop zone -->
    <div id="drop-zone" onclick="document.getElementById('file-input').click()">
      <div class="dz-icon">🖼️</div>
      <p>Drag &amp; drop gambar di sini<br>atau klik untuk memilih</p>
      <input type="file" id="file-input" accept="image/*">
    </div>

    <div id="file-info">Belum ada gambar dipilih</div>

    <div class="divider"></div>

    <button class="btn btn-blue" onclick="triggerBrowse()">
      📂 Browse File
    </button>

    <button class="btn btn-purple" id="btn-colorize" disabled onclick="runColorize()">
      🎨 Colorize!
    </button>

    <div id="status-text">Pilih gambar untuk mulai</div>
    <div class="progress-bar">
      <div class="progress-fill" id="progress-fill"></div>
    </div>

    <div class="divider"></div>

    <div class="samples-section">
      <div class="samples-title">📷 Contoh Gambar B&W</div>
      <div class="samples-subtitle">Klik untuk mencoba</div>
      <div id="samples-grid" class="samples-grid">
        {{SAMPLE_THUMBS}}
      </div>
    </div>

    <div class="divider"></div>

    <button class="btn btn-green" id="btn-save" disabled onclick="downloadResult('result')">
      💾 Save Result
    </button>
    <button class="btn btn-dark" id="btn-cmp" disabled onclick="downloadResult('comparison')">
      ⚖️ Save Comparison
    </button>
    
    <div style="flex-grow: 1;"></div>
    <div style="text-align: center; margin-top: 6px; padding-top: 8px; border-top: 1px dashed var(--border);">
      <a href="#" onclick="showAbout(); return false;" style="color: var(--muted); font-size: 11px; text-decoration: none; display: inline-block; padding: 4px; transition: color 0.2s;" onmouseover="this.style.color='var(--text)'" onmouseout="this.style.color='var(--muted)'">
        About Us
      </a>
    </div>

  </aside>

  <!-- ── CENTER PANEL ────────────────────────────────────────── -->
  <main id="center">
    <div id="preview-header">
      <span class="label-before">BEFORE</span>
      <span class="label-mid">◀ Preview ▶</span>
      <span class="label-after">AFTER</span>
    </div>
    <div id="preview-area">
      <div class="img-panel before" id="panel-before">
        <span class="corner-label">BEFORE</span>
        <div class="placeholder">Pilih gambar<br>untuk memulai 🖼️</div>
      </div>
      <div id="divider"></div>
      <div class="img-panel after" id="panel-after">
        <span class="corner-label">AFTER</span>
        <div class="placeholder">Klik Colorize!<br>untuk mewarnai 🎨</div>
      </div>
    </div>
  </main>

  <!-- ── RIGHT PANEL ─────────────────────────────────────────── -->
  <aside id="right">
    <div id="editor-header">
      <h2>✦ Editor</h2>
      <p>Adjust setelah colorize</p>
    </div>
    <div id="editor-actions">
      <button class="btn btn-purple" style="flex:1" onclick="autoEnhance()">
        ⚡ Auto
      </button>
      <button class="btn btn-dark" style="flex:1" onclick="resetEdits()">
        ↺ Reset
      </button>
    </div>
    <div id="sliders-area" id="sliders-container"></div>
  </aside>

</div>

<!-- Loading overlay -->
<div id="loading">
  <div class="spinner"></div>
  <h3 id="load-title">Memproses...</h3>
  <p id="load-msg">Harap tunggu</p>
  <div class="load-bar">
    <div class="load-fill" id="load-fill"></div>
  </div>
</div>

<!-- About Modal -->
<div id="about-modal" onclick="if(event.target===this) hideAbout()">
  <div class="modal-content">
    <div class="modal-close" onclick="hideAbout()">×</div>
    <div class="modal-title">✨ Tentang Sistem Kolorisasi</div>
    <div class="modal-desc">
      Aplikasi ini dikembangkan sebagai pemenuhan Ujian Akhir Semester (UAS) mata kuliah Pengolahan Citra Digital (TIF23). Kami menggunakan framework Zhang Colorization CNN (ECCV 2016) untuk mewarnai foto hitam-putih secara otomatis.
    </div>
    <div class="modal-title" style="font-size: 14px; margin-top:20px;">👥 Anggota Kelompok</div>
    <ul class="team-list">
      <li class="team-item">
        <div class="team-icon">👨‍💻</div>
        <div>
          <div class="team-nim">32230033</div>
          <div class="team-name">Harlan Luthi Permana</div>
        </div>
      </li>
      <li class="team-item">
        <div class="team-icon">👨‍💻</div>
        <div>
          <div class="team-nim">32230048</div>
          <div class="team-name">Ruben Wijaya</div>
        </div>
      </li>
      <li class="team-item">
        <div class="team-icon">👨‍💻</div>
        <div>
          <div class="team-nim">32230064</div>
          <div class="team-name">Mohammad Bintang Indy Taura Putra</div>
        </div>
      </li>
    </ul>
  </div>
</div>

<!-- Toast -->
<div id="toast"></div>

<script>
// ══════════════════════════════════════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════════════════════════════════════
let sessionId    = null;
let isColorized  = false;
let debounceTimer = null;

const SLIDER_DEFS = [
  { key:"exposure",       label:"Exposure",        min:-100, max:100, def:0 },
  { key:"brilliance",     label:"Brilliance",       min:-100, max:100, def:0 },
  { key:"highlights",     label:"Highlights",       min:-100, max:100, def:0 },
  { key:"shadows",        label:"Shadows",          min:-100, max:100, def:0 },
  { key:"contrast",       label:"Contrast",         min:-100, max:100, def:0 },
  { key:"brightness",     label:"Brightness",       min:-100, max:100, def:0 },
  { key:"black_point",    label:"Black Point",      min:-100, max:100, def:0 },
  { key:"saturation",     label:"Saturation",       min:-100, max:100, def:0 },
  { key:"vibrance",       label:"Vibrance",         min:-100, max:100, def:0 },
  { key:"warmth",         label:"Warmth",           min:-100, max:100, def:0 },
  { key:"tint",           label:"Tint",             min:-100, max:100, def:0 },
  { key:"sharpness",      label:"Sharpness",        min:0,    max:100, def:0 },
  { key:"definition",     label:"Definition",       min:0,    max:100, def:0 },
  { key:"noise_reduction",label:"Noise Reduction",  min:0,    max:100, def:0 },
  { key:"vignette",       label:"Vignette",         min:-100, max:100, def:0 },
];

// ══════════════════════════════════════════════════════════════════════════
//  BUILD SLIDERS
// ══════════════════════════════════════════════════════════════════════════
function buildSliders() {
  const container = document.getElementById("sliders-area");
  container.innerHTML = "";
  SLIDER_DEFS.forEach(s => {
    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML = `
      <div class="slider-top">
        <span class="slider-name">${s.label}</span>
        <span class="slider-val" id="val-${s.key}">0</span>
      </div>
      <input type="range" id="sl-${s.key}"
        min="${s.min}" max="${s.max}" value="${s.def}" step="1"
        oninput="onSlider('${s.key}', this.value)"
      >`;
    container.appendChild(row);
    updateSliderTrack(s.key, s.def, s.min, s.max);
  });
}

function updateSliderTrack(key, val, min, max) {
  const el = document.getElementById(`sl-${key}`);
  if (!el) return;
  const pct = ((val - min) / (max - min)) * 100;
  el.style.background = `linear-gradient(to right,
    #8b5cf6 0%, #8b5cf6 ${pct}%, #2a2a45 ${pct}%, #2a2a45 100%)`;
}

function onSlider(key, val) {
  const numVal = parseFloat(val);
  const el = SLIDER_DEFS.find(s => s.key === key);
  const valEl = document.getElementById(`val-${key}`);
  if (valEl) {
    valEl.textContent = (el.min < 0)
      ? (numVal >= 0 ? "+" : "") + Math.round(numVal)
      : Math.round(numVal);
  }
  updateSliderTrack(key, numVal, el.min, el.max);
  if (!isColorized) return;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(sendAdjust, 400);
}

// ══════════════════════════════════════════════════════════════════════════
//  DRAG & DROP
// ══════════════════════════════════════════════════════════════════════════
const dropZone = document.getElementById("drop-zone");
dropZone.addEventListener("dragover", e => {
  e.preventDefault(); dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) uploadFile(file);
});

document.getElementById("file-input").addEventListener("change", e => {
  if (e.target.files[0]) uploadFile(e.target.files[0]);
});

function triggerBrowse() {
  document.getElementById("file-input").click();
}

// ══════════════════════════════════════════════════════════════════════════
//  UPLOAD
// ══════════════════════════════════════════════════════════════════════════
async function uploadFile(file) {
  setStatus("Mengupload gambar...", 5);
  const fd = new FormData();
  fd.append("image", file);

  try {
    const res = await fetch("/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (data.error) { showToast("❌ " + data.error); return; }

    sessionId   = data.session_id;
    isColorized = false;

    // Tampilkan di panel before
    showImage("panel-before", "data:image/jpeg;base64," + data.preview);
    clearPanel("panel-after", "Klik Colorize!\nuntuk mewarnai 🎨");

    document.getElementById("file-info").textContent =
      `${data.filename}\n${data.width}×${data.height} px`;
    document.getElementById("btn-colorize").disabled = false;
    document.getElementById("btn-save").disabled  = true;
    document.getElementById("btn-cmp").disabled   = true;
    setStatus("Gambar dimuat! Klik Colorize 🎨", 0);
    resetEdits(false);
    showToast("✅ Gambar berhasil dimuat");
  } catch (e) {
    showToast("❌ Upload gagal: " + e.message);
  }
}

// ══════════════════════════════════════════════════════════════════════════
//  COLORIZE (SSE streaming)
// ══════════════════════════════════════════════════════════════════════════
function runColorize() {
  if (!sessionId) return;
  document.getElementById("btn-colorize").disabled = true;
  showLoading(true, "Menjalankan AI...", "Zhang Colorization CNN");

  const es = new EventSource(`/colorize/${sessionId}`);

  es.addEventListener("progress", e => {
    const d = JSON.parse(e.data);
    setLoadProgress(d.pct, d.msg);
    setStatus(d.msg, d.pct);
  });

  es.addEventListener("done", e => {
    es.close();
    const d = JSON.parse(e.data);
    showImage("panel-before", "data:image/jpeg;base64," + d.before);
    showImage("panel-after",  "data:image/jpeg;base64," + d.after);
    isColorized = true;
    showLoading(false);
    setStatus("✅ Kolorisasi selesai!", 100);
    document.getElementById("btn-colorize").disabled = false;
    document.getElementById("btn-save").disabled  = false;
    document.getElementById("btn-cmp").disabled   = false;
    showToast("🎨 Kolorisasi berhasil!");
  });

  es.addEventListener("error", e => {
    es.close();
    showLoading(false);
    document.getElementById("btn-colorize").disabled = false;
    try {
      const d = JSON.parse(e.data);
      showToast("❌ Error: " + d.msg);
      setStatus("Error: " + d.msg, 0);
    } catch(_) {
      showToast("❌ Koneksi terputus");
    }
  });

  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) return;
    es.close();
    showLoading(false);
    document.getElementById("btn-colorize").disabled = false;
    showToast("❌ Koneksi ke server terputus");
  };
}

// ══════════════════════════════════════════════════════════════════════════
//  ADJUST
// ══════════════════════════════════════════════════════════════════════════
async function sendAdjust() {
  if (!sessionId || !isColorized) return;

  const sv = {};
  SLIDER_DEFS.forEach(s => {
    sv[s.key] = parseFloat(document.getElementById(`sl-${s.key}`).value);
  });

  try {
    const res = await fetch(`/adjust/${sessionId}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(sv)
    });
    const data = await res.json();
    if (data.after) {
      showImage("panel-after", "data:image/jpeg;base64," + data.after);
    }
  } catch(e) {
    console.error("Adjust error:", e);
  }
}

// ══════════════════════════════════════════════════════════════════════════
//  AUTO ENHANCE & RESET
// ══════════════════════════════════════════════════════════════════════════
const AUTO_PRESETS = {
  exposure:5, brilliance:20, highlights:-10, shadows:15,
  contrast:10, brightness:5, black_point:0, saturation:15,
  vibrance:25, warmth:5, tint:0, sharpness:20,
  definition:15, noise_reduction:0, vignette:-10,
};

function autoEnhance() {
  SLIDER_DEFS.forEach(s => {
    const v = AUTO_PRESETS[s.key] ?? 0;
    const el = document.getElementById(`sl-${s.key}`);
    if (el) { el.value = v; onSlider(s.key, v); }
  });
  if (isColorized) sendAdjust();
}

function resetEdits(sendReq = true) {
  SLIDER_DEFS.forEach(s => {
    const el = document.getElementById(`sl-${s.key}`);
    if (el) { el.value = s.def; onSlider(s.key, s.def); }
  });
  if (sendReq && isColorized) sendAdjust();
}

// ══════════════════════════════════════════════════════════════════════════
//  DOWNLOAD
// ══════════════════════════════════════════════════════════════════════════
function downloadResult(mode) {
  if (!sessionId) return;
  window.location.href = `/download/${sessionId}/${mode}`;
}

// ══════════════════════════════════════════════════════════════════════════
//  HELPERS — UI
// ══════════════════════════════════════════════════════════════════════════
function showImage(panelId, src) {
  const panel = document.getElementById(panelId);
  panel.innerHTML = `
    <span class="corner-label">${panelId === 'panel-before' ? 'BEFORE' : 'AFTER'}</span>
    <img src="${src}" alt="preview">
  `;
}

function clearPanel(panelId, text) {
  const panel = document.getElementById(panelId);
  panel.innerHTML = `
    <span class="corner-label">${panelId === 'panel-before' ? 'BEFORE' : 'AFTER'}</span>
    <div class="placeholder">${text}</div>
  `;
}

function setStatus(msg, pct) {
  document.getElementById("status-text").textContent = msg;
  document.getElementById("progress-fill").style.width = pct + "%";
}

function showLoading(show, title="", msg="") {
  const el = document.getElementById("loading");
  el.classList.toggle("show", show);
  if (show) {
    document.getElementById("load-title").textContent = title;
    document.getElementById("load-msg").textContent   = msg;
    document.getElementById("load-fill").style.width  = "0%";
  }
}

function setLoadProgress(pct, msg) {
  document.getElementById("load-fill").style.width = pct + "%";
  document.getElementById("load-msg").textContent  = msg;
}

let toastTimer;
function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 3000);
}

// ══════════════════════════════════════════════════════════════════════════
//  SAMPLE IMAGES
// ══════════════════════════════════════════════════════════════════════════
async function loadSampleImage(filename, label) {
  setStatus('Memuat sample: ' + label + '...', 5);
  try {
    // Fetch gambar sample sebagai blob lalu upload seperti biasa
    const imgRes = await fetch('/sample-img/' + filename);
    const blob = await imgRes.blob();
    const file = new File([blob], filename, { type: blob.type });
    await uploadFile(file);
    showToast('📷 Sample "' + label + '" dimuat!');
  } catch(e) {
    showToast('❌ Gagal memuat sample: ' + e.message);
  }
}

// ══════════════════════════════════════════════════════════════════════════
//  MODAL ABOUT US
// ══════════════════════════════════════════════════════════════════════════
function showAbout() {
  document.getElementById('about-modal').classList.add('show');
}
function hideAbout() {
  document.getElementById('about-modal').classList.remove('show');
}

// ══════════════════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════════════════
buildSliders();
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def launch_web(port: int = 5050, open_browser: bool = True):
    print("\n" + "=" * 55)
    print("  ✨ AI Colorization Web GUI")
    print(f"  Buka browser: http://localhost:{port}")
    print("  Tekan Ctrl+C untuk berhenti")
    print("=" * 55 + "\n")

    if open_browser:
        import webbrowser, threading
        threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    launch_web()
