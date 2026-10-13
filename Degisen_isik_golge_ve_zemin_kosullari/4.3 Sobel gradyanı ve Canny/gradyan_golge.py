"""4.3 Sobel gradyanı ve Canny: gradyan maskesi ve gölge sınırlarının ayıklanması.

Amaç
    Kavram ve uygulama 2.2 ile aynıdır: Sobel (x yönü) ya da Canny ile kenar maskesi.
    Bu başlıktaki fark: gölge sınırları da güçlü gradyan üretir ve yanlış şerit sanılır.
    Bu yüzden gradyan tek başına kullanılmaz, renk maskesiyle birleştirilir (4.4).
    Tür: Kavramsal. Girdi -> çıktı: gri tonlu görüntü -> kenar maskesi.

Akış
    1. Girdi (tek görüntü, görüntü klasörü ya da video) kare kare okunur.
    2. undistort (kalibrasyon verildiyse) -> gri ton -> blur (Gaussian / bilateral).
    3. Sobel-x (mutlak, sabit ölçekli, eşikli) ya da Canny ile kenar maskesi (2.2 ile aynı).
    4. Çift kenar filtresi (varsayılan açık): açık renkli şerit çizgisinin solunda yükselen
       (koyu -> açık, gx > 0), sağında düşen (gx < 0) kenar vardır ve aralarındaki uzaklık
       en çok bir çizgi genişliğidir. Gölge sınırı tek basamaktır (eşi yok), direk gölgesi
       koyu bir şerittir (önce düşen sonra yükselen): ikisi de atılır.
    5. Maske kaydedilir; atılan kenar oranı yazdırılır. Maske 4.4'te renkle birleştirilmeli.

Gereksinimde belirtilen araçlar:
    - Ön işleme sırası undistort -> blur -> gradyan (undistort --calibration ile).
    - cv2.GaussianBlur ya da cv2.bilateralFilter.
    - cv2.Sobel x yönünde (dikeye yakın şeritler); mutlak değer, ölçekleme, eşik.
    - cv2.Canny (--method canny).
    - Gradyan tek başına kullanılmaz: çıktı 4.4'te renk maskesiyle birleştirilir.

Gereksinimde belirtilmeyen, eklenenler (kodda [EK] ile işaretli):
    - [EK] Çift kenar (şerit sırtı) filtresi (--pair-filter, varsayılan açık): gölge
      sınırını soluk çizgiden ayırır. Renk ikisini de göremez, OR ikisini de alır, AND
      ikisini de atar (4.4 testi); gradyanın işaret yapısı ise ikisini ayırır.
      İşaretli gradyanda pencere araması cv2.dilate ile yapılır (yön, anchor ile).
    - [EK] Sabit ölçekleme (|gx| / (4 x 255), 2.2 ile aynı) ve Canny'de L2 gradyan normu.
    - [EK] Atılan kenar oranı: gölge / basamak kenarlarının ne kadar olduğunu gösterir.
    - [EK] CameraCalibration (2.2'den kopya), klasör / video girişi, kayıpsız maske videosu
      (FFV1), 3 panelli karşılaştırma (kalan kenar beyaz, atılan kırmızı).
    Not: Çizgi genişliğinden geniş açık şeritler (yaya geçidi, çok yakın çizgi) filtrede
         kenarlarını kaybeder; --max-line-width kuş bakışına / perspektife göre ayarlanmalı.
    Not: İki gölge arasında kalan, çizgi genişliğinden dar güneşli şerit bir çizgi gibi
         görünür ve filtreden geçer; renkle birleştirme (4.4) bunu azaltır.
    Not: --no-pair-filter ile çıktı 2.2 ile birebir aynıdır (testte doğrulandı).

Kullanım (konteyner içinde; varsayılan yollar proje köküne göredir):
    docker exec -it ads-cv-lab bash
    cd "/workspace/Degisen_isik_golge_ve_zemin_kosullari/4.3 Sobel gradyanı ve Canny"
    python gradyan_golge.py --compare                        # INPUT_PATH, Sobel-x + çift kenar
    python gradyan_golge.py --method canny --compare
    python gradyan_golge.py --input /workspace/data/zor.mp4 --max-line-width 0.04
    python gradyan_golge.py --no-pair-filter                 # 2.2 ile aynı çıktı
    python gradyan_golge.py --help

Okunan dosya (isteğe bağlı, --calibration): .npz
    camera_matrix (3x3), dist_coeffs (N), image_size (genişlik, yükseklik) (2.2 ile aynı varsayım).

Kaydedilen dosyalar (ADS/outputs/<bu klasörün adı>/):
    <ad>_<yöntem>_maske.png   tek kanal uint8, girdiyle aynı boyut; 0 = arka plan,
                              255 = kenar pikseli. <yöntem>: sobelx ya da canny.
    <klasör>/...              klasör girişinde aynı biçim, alt klasörde.
    <ad>_<yöntem>_maske.mkv   video girişinde: FFV1 kayıpsız, gri; okurken frame[:, :, 0] > 0.
    <ad>_<yöntem>_karsilastirma.png / .mp4   --compare: girdi | filtre öncesi (kalan beyaz,
                              atılan kırmızı) | sonuç maskesi. Yalnızca görsel amaçlı.
"""

import argparse
import math
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# AYARLAR
# Komut satırı argümanları bu varsayılanları ezer (python ... --help).
# [DEĞİŞTİR] işaretli olanları kendi verine göre ayarla.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / Path(__file__).resolve().parent.name

# [DEĞİŞTİR] Renkli ya da gri kare: tek görüntü, görüntü klasörü ya da video.
INPUT_PATH = DATA_DIR / "yol_karesi.jpg"

# [DEĞİŞTİR] Kamera kalibrasyonu (.npz). None ise undistort atlanır.
CALIBRATION_PATH = None

METHOD = "sobel"              # "sobel" (x yönünde) ya da "canny"

# Bulanıklaştırma (gri görüntüye; 2.2 ile aynı): "gaussian", "bilateral" ya da "none".
BLUR = "gaussian"
GAUSSIAN_KSIZE = 5
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 15.0  # gri görüntüde 15 (2.2'deki ölçüm; 75 soluk çizgiyi siler)
BILATERAL_SIGMA_SPACE = 75.0

SOBEL_KSIZE = 3               # 3, 5 ya da 7
# [DEĞİŞTİR] Sabit ölçekli |gx| eşiği (0-255), ~kenardaki parlaklık farkı. Çift kenar
# filtresinde Canny kenarlarının yönü de bu alt eşikten güçlü gx ile belirlenir.
SOBEL_RANGE = (12, 255)
# [DEĞİŞTİR] Canny çift eşiği (gradyan büyüklüğü, ölçeksiz).
CANNY_THRESHOLDS = (30, 60)

# Çift kenar filtresi (gölge sınırlarını ayıklar).
PAIR_FILTER = True
# [DEĞİŞTİR] Şerit çizgisinin en büyük genişliği, görüntü genişliğinin oranı
# (1280'de ~38 px). Kuş bakışında çizgi genişliği sabittir; perspektifte altta en geniştir.
MAX_LINE_WIDTH = 0.03

# Video: en çok kaç kare işlensin (None = tümü).
MAX_FRAMES = None
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


# [EK] 2.2'deki kalibrasyon yükleyicisinin kopyası (klasörler arası import yapılmaz).
class CameraCalibration:
    """Kamera kalibrasyonunu yükler ve kareleri düzeltir (undistort)."""

    REQUIRED_KEYS = ("camera_matrix", "dist_coeffs", "image_size")

    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise InputError(
                f"Kalibrasyon dosyası bulunamadı: {path}\n"
                "  -> --calibration ile doğru yolu ver ya da undistort'suz çalıştır "
                "(--calibration vermeden).")
        try:
            with np.load(path) as data:
                missing = [k for k in self.REQUIRED_KEYS if k not in data.files]
                if missing:
                    raise ValueError(f"eksik alanlar: {', '.join(missing)}")
                self.camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
                self.dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).ravel()
                self.image_size = tuple(int(v) for v in data["image_size"])
            if self.camera_matrix.shape != (3, 3):
                raise ValueError(f"camera_matrix 3x3 olmalı, verilen {self.camera_matrix.shape}")
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise InputError(
                f"Kalibrasyon dosyası okunamadı: {path} ({exc})\n"
                "  -> Beklenen alanlar: camera_matrix (3x3), dist_coeffs, image_size.") from None
        self.path = path

    def check_size(self, width, height):
        if (int(width), int(height)) != self.image_size:
            raise InputError(
                f"Görüntü boyutu {width}x{height}, ancak kalibrasyon "
                f"{self.image_size[0]}x{self.image_size[1]} için yapıldı ({self.path}).\n"
                "  -> Kalibrasyonu bu çözünürlükte yap ya da kareleri aynı boyutta ver.")

    def undistort(self, frame):
        self.check_size(frame.shape[1], frame.shape[0])
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)


# --------------------------------------------------------------------------- gradyan (2.2)
def blur_gray(gray, args):
    if args.blur == "gaussian":
        return cv2.GaussianBlur(gray, (args.gaussian_ksize, args.gaussian_ksize), 0)
    if args.blur == "bilateral":
        return cv2.bilateralFilter(gray, args.bilateral_d, args.bilateral_sigma_color,
                                   args.bilateral_sigma_space)
    return gray


# [EK] Sabit ölçek: |gx|'in alabileceği en büyük değer (uint8 girdi için).
def sobel_x_max(ksize):
    kx, ky = cv2.getDerivKernels(1, 0, ksize)
    return 255.0 * float(kx[kx > 0].sum()) * float(ky.sum())


def sobel_x(gray, ksize):
    """x yönünde Sobel. Dönüş: (2.2 ile aynı ölçekli |gx| uint8, işaretli ölçekli gx).

    İşaretli gx: + koyu -> açık (çizginin sol kenarı), - açık -> koyu (sağ kenar).
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    denom = sobel_x_max(ksize)
    scaled = np.clip(255.0 * np.abs(gx) / denom, 0, 255).astype(np.uint8)   # 2.2 ile aynı sıra
    return scaled, 255.0 * gx / denom


def raw_edge_mask(smooth, scaled, args):
    """2.2 ile aynı kenar maskesi (0/255)."""
    if args.method == "canny":
        # [EK] L2gradient: büyüklük sqrt(gx^2 + gy^2) ile hesaplanır.
        return cv2.Canny(smooth, args.canny_thresholds[0], args.canny_thresholds[1], L2gradient=True)
    return cv2.inRange(scaled, args.sobel_range[0], args.sobel_range[1])


# [EK] Çift kenar filtresi: açık şerit = solda yükselen, sağda düşen kenar.
def pair_filter(edges, gx, min_strength, max_width_px):
    """Eşi (zıt işaretli kenar) en çok max_width_px uzakta olmayan kenar piksellerini atar.

    Yükselen piksel (gx > 0) sağındaki [x, x+w] içinde düşen kenar arar; düşen piksel
    (gx < 0) solundaki [x-w, x] içinde yükselen kenar arar. Gölge sınırı (tek basamak) ve
    koyu şerit (önce düşen, sonra yükselen: direk gölgesi) böylece atılır.
    """
    on = edges > 0
    rising = (on & (gx >= min_strength)).astype(np.uint8)
    falling = (on & (gx <= -min_strength)).astype(np.uint8)
    kernel = np.ones((1, max_width_px + 1), np.uint8)
    # anchor (0, 0): çıktı x'te, girdinin [x, x+w] aralığında piksel varsa 1 (sağa bakar).
    falling_right = cv2.dilate(falling, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT,
                               borderValue=0)
    # anchor (w, 0): çıktı x'te, girdinin [x-w, x] aralığında piksel varsa 1 (sola bakar).
    rising_left = cv2.dilate(rising, kernel, anchor=(max_width_px, 0),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
    keep = (rising & falling_right) | (falling & rising_left)
    return keep.astype(np.uint8) * 255


def edge_mask(frame_bgr, args, calibration=None):
    """Dönüş: (ön işlenmiş kare, filtre öncesi maske, sonuç maskesi), maskeler 0/255."""
    if calibration is not None:
        frame_bgr = calibration.undistort(frame_bgr)
    smooth = blur_gray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), args)
    scaled, gx = sobel_x(smooth, args.sobel_ksize)
    raw = raw_edge_mask(smooth, scaled, args)
    if not args.pair_filter:
        return frame_bgr, raw, raw
    width_px = max(1, round(args.max_line_width * frame_bgr.shape[1]))
    return frame_bgr, raw, pair_filter(raw, gx, args.sobel_range[0], width_px)


def ratio(mask):
    return cv2.countNonZero(mask) / mask.size


def method_tag(args):
    return "canny" if args.method == "canny" else "sobelx"


# --------------------------------------------------------------------------- girdi / çıktı
def read_image(path):
    img = cv2.imread(str(path), flags=cv2.IMREAD_COLOR)
    if img is None:
        raise InputError(
            f"Görüntü okunamadı: {path}\n"
            "  -> Dosyanın bozuk olmadığını ve desteklenen bir biçimde "
            f"({', '.join(sorted(IMAGE_EXTS))}) olduğunu kontrol et.")
    return img


def write_image(path, img):
    if not cv2.imwrite(str(path), img):
        raise InputError(f"Görüntü yazılamadı: {path}\n  -> --out-dir'i ve yazma iznini kontrol et.")


def open_writer(path, fourcc, fps, size, is_color):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size,
                             isColor=is_color)
    if not writer.isOpened():
        raise InputError(f"Video yazılamadı: {path} ({fourcc})\n"
                         "  -> --out-dir'i ve yazma iznini kontrol et.")
    return writer


def put_label(img, text, fs, th):
    org = (10, round(35 * fs / 0.8))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                thickness=th + 2, lineType=cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 165, 255),
                thickness=th, lineType=cv2.LINE_AA)


# [EK] Girdi | filtre öncesi (kalan beyaz, atılan kırmızı) | sonuç maskesi.
def draw_compare(frame, raw, final, args, undistorted):
    width = frame.shape[1]
    fs = 0.8 * width / 1280.0
    th = max(1, round(2 * width / 1280.0))
    left = frame.copy()
    put_label(left, "Girdi (duzeltilmis)" if undistorted else "Girdi", fs, th)
    mid = np.zeros_like(frame)
    mid[(raw > 0) & (final == 0)] = (0, 0, 255)
    mid[final > 0] = (255, 255, 255)
    name = "Canny" if args.method == "canny" else "Sobel-x"
    put_label(mid, f"{name}: kalan beyaz, atilan kirmizi", fs, th)
    right = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    filt = "cift kenar" if args.pair_filter else "filtresiz"
    put_label(right, f"Sonuc ({filt}): %{100 * ratio(final):.2f} - tek basina kullanma, 4.4", fs, th)
    return cv2.hconcat([left, mid, right])


def describe(raw, final):
    removed = 1 - (ratio(final) / ratio(raw)) if ratio(raw) > 0 else 0.0
    return (f"kenar %{100 * ratio(raw):.2f} -> çift kenar sonrası %{100 * ratio(final):.2f} "
            f"(atılan %{100 * removed:.1f})")


# --------------------------------------------------------------------------- işleme
def process_image(path, out_dir, args, calibration):
    shown, raw, final = edge_mask(read_image(path), args, calibration)
    stem = f"{path.stem}_{method_tag(args)}"
    out_path = out_dir / f"{stem}_maske.png"
    write_image(out_path, final)
    if args.compare:
        write_image(out_dir / f"{stem}_karsilastirma.png",
                    draw_compare(shown, raw, final, args, calibration is not None))
    print(f"{path.name}: {describe(raw, final)} -> {out_path}")


def process_folder(path, out_dir, args, calibration):
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise InputError(f"Klasörde görüntü yok: {path}\n"
                         f"  -> Desteklenen uzantılar: {', '.join(sorted(IMAGE_EXTS))}")
    sub_dir = out_dir / path.name
    sub_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        process_image(file, sub_dir, args, calibration)
    print(f"{len(files)} görüntü işlendi -> {sub_dir}")


def process_video(path, out_dir, args, calibration):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InputError(
            f"Video açılamadı: {path}\n"
            "  -> Dosyanın bozuk olmadığını ve desteklenen bir biçimde "
            f"({', '.join(sorted(VIDEO_EXTS))}) olduğunu kontrol et.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (math.isfinite(fps) and fps > 0):
        fps = 25.0
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise InputError(f"Videodan hiç kare okunamadı: {path}")
    height, width = frame.shape[:2]
    if calibration is not None:
        calibration.check_size(width, height)
    stem = f"{path.stem}_{method_tag(args)}"
    out_path = out_dir / f"{stem}_maske.mkv"
    writers = []
    try:
        # [EK] FFV1 kayıpsızdır; mp4 sıkıştırması maskenin 0/255 değerlerini bozar.
        writer = open_writer(out_path, "FFV1", fps, (width, height), is_color=False)
        writers.append(writer)
        compare_writer = None
        if args.compare:
            compare_writer = open_writer(out_dir / f"{stem}_karsilastirma.mp4", "mp4v", fps,
                                         (3 * width, height), is_color=True)
            writers.append(compare_writer)
        count, raw_sum, final_sum = 0, 0.0, 0.0
        while ok and (args.max_frames is None or count < args.max_frames):
            if frame.shape[:2] != (height, width):
                raise InputError(
                    f"Videonun {count}. karesinin boyutu değişti "
                    f"({frame.shape[1]}x{frame.shape[0]}, ilk kare {width}x{height}).")
            shown, raw, final = edge_mask(frame, args, calibration)
            writer.write(final)
            if compare_writer is not None:
                compare_writer.write(draw_compare(shown, raw, final, args, calibration is not None))
            raw_sum += ratio(raw)
            final_sum += ratio(final)
            count += 1
            ok, frame = cap.read()
    finally:
        cap.release()
        for w in writers:
            w.release()
    removed = 1 - final_sum / raw_sum if raw_sum > 0 else 0.0
    print(f"{path.name}: {count} kare, ortalama kenar %{100 * raw_sum / count:.2f} -> "
          f"%{100 * final_sum / count:.2f} (atılan %{100 * removed:.1f}) -> {out_path}")


# --------------------------------------------------------------------------- argümanlar
class TurkishArgumentParser(argparse.ArgumentParser):
    """Argüman hatasında "HATA: ..." yazar ve 1 koduyla çıkar."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"HATA: Geçersiz komut satırı argümanı: {message}\n"
              "  -> Seçenekler için --help kullan.", file=sys.stderr)
        sys.exit(1)


def parse_args(argv=None):
    p = TurkishArgumentParser(
        description="Sobel-x ya da Canny kenar maskesi; çift kenar filtresiyle gölge sınırlarını ayıklar.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"Görüntü, görüntü klasörü ya da video (varsayılan: {INPUT_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--calibration", type=Path, default=CALIBRATION_PATH,
                   help="Kamera kalibrasyonu .npz; verilirse önce undistort (varsayılan: yok)")
    p.add_argument("--method", choices=("sobel", "canny"), default=METHOD,
                   help=f"Kenar yöntemi (varsayılan: {METHOD})")
    p.add_argument("--blur", choices=("gaussian", "bilateral", "none"), default=BLUR,
                   help=f"Bulanıklaştırma (varsayılan: {BLUR})")
    p.add_argument("--gaussian-ksize", type=int, default=GAUSSIAN_KSIZE,
                   help=f"Gaussian çekirdek boyutu, tek sayı (varsayılan: {GAUSSIAN_KSIZE})")
    p.add_argument("--bilateral-d", type=int, default=BILATERAL_D,
                   help=f"Bilateral komşuluk çapı (varsayılan: {BILATERAL_D})")
    p.add_argument("--bilateral-sigma-color", type=float, default=BILATERAL_SIGMA_COLOR,
                   help=f"Bilateral renk sigması (varsayılan: {BILATERAL_SIGMA_COLOR})")
    p.add_argument("--bilateral-sigma-space", type=float, default=BILATERAL_SIGMA_SPACE,
                   help=f"Bilateral uzay sigması (varsayılan: {BILATERAL_SIGMA_SPACE})")
    p.add_argument("--sobel-ksize", type=int, choices=(3, 5, 7), default=SOBEL_KSIZE,
                   help=f"Sobel çekirdek boyutu (varsayılan: {SOBEL_KSIZE})")
    p.add_argument("--sobel-range", type=int, nargs=2, default=SOBEL_RANGE, metavar=("ALT", "UST"),
                   help=f"Sabit ölçekli |gx| eşiği (varsayılan: {SOBEL_RANGE})")
    p.add_argument("--canny-thresholds", type=float, nargs=2, default=CANNY_THRESHOLDS,
                   metavar=("ALT", "UST"), help=f"Canny çift eşiği (varsayılan: {CANNY_THRESHOLDS})")
    p.add_argument("--no-pair-filter", dest="pair_filter", action="store_false", default=PAIR_FILTER,
                   help="Çift kenar filtresini kapat (çıktı 2.2 ile aynı olur)")
    p.add_argument("--max-line-width", type=float, default=MAX_LINE_WIDTH,
                   help=f"En büyük çizgi genişliği, görüntü genişliğinin oranı (varsayılan: {MAX_LINE_WIDTH})")
    p.add_argument("--compare", action="store_true",
                   help="Girdi, filtre öncesi ve sonuç maskesini yan yana kaydet")
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                   help="Videoda en çok kaç kare işlensin (varsayılan: tümü)")
    return p.parse_args(argv)


def validate_args(args):
    if args.gaussian_ksize <= 0 or args.gaussian_ksize % 2 == 0:
        raise InputError(f"--gaussian-ksize pozitif tek sayı olmalı, verilen: {args.gaussian_ksize}")
    if args.bilateral_d <= 0:
        raise InputError(f"--bilateral-d pozitif olmalı, verilen: {args.bilateral_d}")
    for name, value in (("--bilateral-sigma-color", args.bilateral_sigma_color),
                        ("--bilateral-sigma-space", args.bilateral_sigma_space)):
        if not (math.isfinite(value) and value > 0):
            raise InputError(f"{name} pozitif bir sayı olmalı, verilen: {value}")
    low, high = args.sobel_range
    if not 1 <= low <= high <= 255:
        raise InputError(f"--sobel-range 1 <= ALT <= UST <= 255 olmalı, verilen: {low} {high}")
    low, high = args.canny_thresholds
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
        raise InputError(f"--canny-thresholds 0 <= ALT <= UST olmalı, verilen: {low} {high}")
    if not (math.isfinite(args.max_line_width) and 0 < args.max_line_width <= 0.5):
        raise InputError(f"--max-line-width 0 < değer <= 0.5 olmalı, verilen: {args.max_line_width}")
    if args.max_frames is not None and args.max_frames <= 0:
        raise InputError(f"--max-frames pozitif olmalı, verilen: {args.max_frames}")


# --------------------------------------------------------------------------- ana akış
def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    calibration = CameraCalibration(args.calibration) if args.calibration is not None else None
    path = Path(args.input)
    if not path.exists():
        raise InputError(
            f"Girdi bulunamadı: {path}\n"
            "  -> --input ile doğru yolu ver ya da AYARLAR'daki INPUT_PATH'i düzelt.")
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"Çıktı klasörü oluşturulamadı: {args.out_dir} ({exc})\n"
                         "  -> --out-dir'i kontrol et.") from None

    method = (f"Canny {args.canny_thresholds[0]:g}-{args.canny_thresholds[1]:g}" if args.method == "canny"
              else f"Sobel-x eşik {args.sobel_range[0]}-{args.sobel_range[1]}")
    pair = f"açık (en büyük çizgi genişliği %{100 * args.max_line_width:g})" if args.pair_filter else "kapalı"
    print(f"Yöntem: {method}; blur: {args.blur}; çift kenar filtresi: {pair}")
    if path.is_dir():
        process_folder(path, args.out_dir, args, calibration)
    elif path.suffix.lower() in VIDEO_EXTS:
        process_video(path, args.out_dir, args, calibration)
    else:
        process_image(path, args.out_dir, args, calibration)
    print("Not: gradyan maskesini tek başına kullanma; 4.4'te renk maskesiyle birleştir.")


if __name__ == "__main__":
    try:
        main()
    except InputError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
