"""2.2 Sobel veya Canny kenar algılama: gri tonlu kareden kenar maskesi.

Amaç
    Sobel parlaklığın ne kadar hızlı değiştiğini (gradyan) ölçer; Canny gradyan + ince
    kenar + çift eşik (histerezis) ile temiz kenar çıkarır. Problem: renk yetmediğinde
    (soluk çizgi) şerit sınırını parlaklık sıçramasından yakalamak. Neden: şerit çizgisi
    asfaltla arasında keskin bir geçiş oluşturur. Tür: Kavramsal.
    Girdi -> çıktı: gri tonlu görüntü -> kenar haritası / ikili maske.
    Uyarı (4.3): gölge kenarları da güçlü gradyan üretir; bu maske tek başına
    kullanılmamalı, renk maskesiyle (2.1) birleştirilmeli (4.4).

Akış
    1. Girdi (tek görüntü, görüntü klasörü ya da video) kare kare okunur.
    2. Kalibrasyon dosyası verildiyse kare düzeltilir (undistort).
    3. Gri tona çevrilir ve bulanıklaştırılır (Gaussian ya da bilateral).
    4a. sobel: x yönünde Sobel -> mutlak değer -> 0-255'e ölçekleme -> eşikleme.
    4b. canny: Canny (gradyan + ince kenar + histerezis).
    5. İkili maske (0 / 255) kaydedilir; --compare ile girdi, kenar haritası ve maske
       yan yana kaydedilir.

Gereksinimde belirtilen araçlar:
    - Ön işleme sırası undistort -> blur -> gradyan (undistort, --calibration ile
      kalibrasyon dosyası verildiğinde yapılır; cv2.undistort).
    - cv2.GaussianBlur ya da cv2.bilateralFilter: gürültüyü bastırır (--blur).
    - cv2.Sobel x yönünde: şeritler dikeye yakın olduğu için; mutlak değer, ölçekleme, eşik.
    - cv2.Canny: gradyan + ince kenar + histerezis (--method canny).

Gereksinimde belirtilmeyen, eklenenler (kodda [EK] ile işaretli):
    - [EK] Sabit ölçekleme (varsayılan): |gx|, çekirdeğin kuramsal en büyük değerine
      (ksize=3 için 4 x 255) bölünür. Karedeki en büyük değere bölmek (--sobel-scale max,
      İnşa'daki klasik yol) eşiği karedeki en güçlü kenara bağlar: güçlü bir kenar varken
      soluk çizgi ezilir, kenarsız karede gürültü büyür. Sabit ölçekte değer kabaca
      "kenardaki parlaklık farkı"dır ve her karede aynı anlamı taşır.
    - [EK] Canny'de L2 gradyan normu: büyüklük sqrt(gx^2 + gy^2) ile daha doğru hesaplanır.
    - [EK] CameraCalibration sınıfı: kalibrasyonu bir kez yükler, görüntü boyutunu denetler.
    - [EK] Klasör / video girişi; kayıpsız maske videosu (FFV1, .mkv): mp4 0/255'i bozar.
    - [EK] Üç panelli karşılaştırma (girdi | kenar haritası | maske): eşik ayarını kolaylaştırır.
    - [EK] Maske oranı (%): eşiğin gevşek ya da sıkı olduğunu sayıyla gösterir.
    Not: Gradyan büyüklüğü / yön eşikleri eklenmedi; x yönündeki Sobel zaten dikeye yakın
         kenarları seçer, yön eşiği gürültülü kenarlarda kararsızdır.
    Not: Gölge kenarını ayıklayan "çift kenar" (açık şeridin iki yanında zıt işaretli
         gradyan) filtresi eklenmedi; perspektifle değişen çizgi genişliği gibi piksel
         parametresi gerektirir. Gölgeye karşı çözüm renk + gradyan birleşimi (4.4).
    Not: Kalibrasyon modülü henüz yok; anahtar adları (aşağıda) varsayımdır, modüller
         birleştirilirken uyarlanacak. Kalibrasyon verilmezse undistort atlanır.
    Not: Bilateral sigmaColor varsayılanı yaygın 75 değil 15: 75, soluk çizgi kenarını
         (~25 gri seviye) siler; bu modülün yakalaması gereken tam da o kenardır.
    Not: Bulanıklaştırma ve Sobel çekirdek boyutları ~1280x720 için seçilmiş ayarlardır,
         hesaplanmış piksel değeri değildir; kalibrasyonun görüntü boyutu ise denetlenir.

Kullanım (konteyner içinde; varsayılan yollar proje köküne göredir):
    docker exec -it ads-cv-lab bash
    cd "/workspace/Şeritten sapma ve şerit merkezinin belirlenmesi/2.2 Sobel veya Canny kenar algılama"
    python kenar_algilama.py --compare                       # INPUT_PATH, Sobel-x
    python kenar_algilama.py --method canny --compare
    python kenar_algilama.py --input /workspace/data/yol.mp4 --blur bilateral
    python kenar_algilama.py --calibration /workspace/data/kamera_kalibrasyon.npz
    python kenar_algilama.py --help

Okunan dosya (isteğe bağlı, --calibration): .npz
    camera_matrix   float64[3, 3]  kamera matrisi K
    dist_coeffs     float64[N]     bozulma katsayıları (k1, k2, p1, p2[, k3 ...])
    image_size      int64[2]       kalibrasyonun yapıldığı görüntü (genişlik, yükseklik)

Kaydedilen dosyalar (ADS/outputs/<bu klasörün adı>/):
    <ad>_<yöntem>_maske.png   tek kanal uint8, girdiyle aynı boyut;
                              0 = arka plan, 255 = kenar pikseli (mantıksal 1).
                              <yöntem>: sobelx ya da canny.
    <klasör>/...              klasör girişinde aynı biçim, girdi klasörünün adıyla alt klasörde.
    <ad>_<yöntem>_maske.mkv   video girişinde: FFV1 kayıpsız, gri, girdiyle aynı fps ve boyut.
                              cv2.VideoCapture kareleri 3 kanal döndürür; frame[:, :, 0] > 0 kullan.
    <ad>_<yöntem>_karsilastirma.png / .mp4   --compare: girdi | kenar haritası | maske
                              (Canny'de girdi | kenarlar). Yalnızca görsel amaçlı.
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

# Bulanıklaştırma: "gaussian", "bilateral" ya da "none".
BLUR = "gaussian"
GAUSSIAN_KSIZE = 5            # tek sayı, px
BILATERAL_D = 9               # komşuluk çapı, px
# sigmaColor korunacak en küçük kenar farkından küçük olmalı: yaygın 75 değeri
# ~25 gri seviyelik soluk çizgi kenarını "benzer renk" sayıp siler (testte %0),
# 15 korur (%100). Bu yüzden 15.
BILATERAL_SIGMA_COLOR = 15.0
BILATERAL_SIGMA_SPACE = 75.0

# Sobel-x.
SOBEL_KSIZE = 3               # 3, 5 ya da 7
SOBEL_SCALE = "fixed"         # "fixed" (sabit, önerilen) ya da "max" (karedeki en büyük değer)
# [DEĞİŞTİR] Ölçekli |gx| eşiği (0-255). Sabit ölçekte ~ kenardaki parlaklık farkı:
# 12, ~25 gri seviye farkı olan soluk çizgiyi yakalar; asfalt dokusu güçlüyse artır.
SOBEL_RANGE = (12, 255)

# [DEĞİŞTİR] Canny çift eşiği (gradyan büyüklüğü, ölçeksiz). Alt eşiğin altı atılır,
# üst eşiğin üstü kesin kenardır, arası güçlü kenara bağlıysa tutulur.
# Yaygın (50, 150) soluk çizgiyi (~64) kaçırır.
CANNY_THRESHOLDS = (30, 60)

# Video: en çok kaç kare işlensin (None = tümü).
MAX_FRAMES = None
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


# [EK] Kalibrasyonu bir kez yükleyen sınıf; görüntü boyutunu denetler.
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


# --------------------------------------------------------------------------- kenar
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


def sobel_x_scaled(gray, ksize, scale_mode):
    """x yönünde Sobel -> mutlak değer -> 0-255 ölçekli uint8 kenar haritası."""
    abs_gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize))
    if scale_mode == "max":
        peak = abs_gx.max()
        denom = peak if peak > 0 else 1.0
    else:
        denom = sobel_x_max(ksize)
    return np.clip(255.0 * abs_gx / denom, 0, 255).astype(np.uint8)


def edge_mask(frame_bgr, args, calibration=None):
    """Dönüş: (ön işlenmiş renkli kare, kenar haritası, ikili maske 0/255)."""
    if calibration is not None:
        frame_bgr = calibration.undistort(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    smooth = blur_gray(gray, args)
    if args.method == "canny":
        # [EK] L2gradient: büyüklük sqrt(gx^2 + gy^2) ile hesaplanır.
        edges = cv2.Canny(smooth, args.canny_thresholds[0], args.canny_thresholds[1],
                          L2gradient=True)
        return frame_bgr, edges, edges
    scaled = sobel_x_scaled(smooth, args.sobel_ksize, args.sobel_scale)
    mask = cv2.inRange(scaled, args.sobel_range[0], args.sobel_range[1])
    return frame_bgr, scaled, mask


# [EK] Maske oranı: eşiğin gevşek ya da sıkı olduğunu sayıyla gösterir.
def mask_ratio(mask):
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


# [EK] Girdi | kenar haritası | maske (Canny'de girdi | kenarlar).
def draw_compare(frame, edge_map, mask, args, undistorted):
    width = frame.shape[1]
    fs = 0.8 * width / 1280.0
    th = max(1, round(2 * width / 1280.0))
    left = frame.copy()
    put_label(left, "Girdi (duzeltilmis)" if undistorted else "Girdi", fs, th)
    panels = [left]
    if args.method == "canny":
        edges = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        low, high = args.canny_thresholds
        put_label(edges, f"Canny {low}-{high} (blur: {args.blur})", fs, th)
        panels.append(edges)
    else:
        # Sabit ölçekte değerler küçük (~10-100); yalnızca görselde 3 kat parlatılır.
        grad = cv2.cvtColor(cv2.convertScaleAbs(edge_map, alpha=3), cv2.COLOR_GRAY2BGR)
        put_label(grad, f"Sobel-x |gx| (gorsel x3), olcek: {args.sobel_scale}, "
                        f"ksize {args.sobel_ksize}", fs, th)
        binary = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        low, high = args.sobel_range
        put_label(binary, f"Maske {low}-{high} (blur: {args.blur})", fs, th)
        panels += [grad, binary]
    return cv2.hconcat(panels)


# --------------------------------------------------------------------------- işleme
def process_image(path, out_dir, args, calibration):
    frame = read_image(path)
    shown, edge_map, mask = edge_mask(frame, args, calibration)
    stem = f"{path.stem}_{method_tag(args)}"
    mask_path = out_dir / f"{stem}_maske.png"
    write_image(mask_path, mask)
    if args.compare:
        write_image(out_dir / f"{stem}_karsilastirma.png",
                    draw_compare(shown, edge_map, mask, args, calibration is not None))
    print(f"{path.name}: kenar maskesi %{100 * mask_ratio(mask):.2f} -> {mask_path}")


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
        calibration.check_size(width, height)  # yazıcıları açmadan önce denetle
    stem = f"{path.stem}_{method_tag(args)}"
    mask_path = out_dir / f"{stem}_maske.mkv"
    panels = 2 if args.method == "canny" else 3
    writers = []
    try:
        # [EK] FFV1 kayıpsızdır; mp4 sıkıştırması maskenin 0/255 değerlerini bozar.
        mask_writer = open_writer(mask_path, "FFV1", fps, (width, height), is_color=False)
        writers.append(mask_writer)
        compare_writer = None
        if args.compare:
            compare_writer = open_writer(out_dir / f"{stem}_karsilastirma.mp4", "mp4v", fps,
                                         (panels * width, height), is_color=True)
            writers.append(compare_writer)

        count, ratio_sum = 0, 0.0
        while ok and (args.max_frames is None or count < args.max_frames):
            if frame.shape[:2] != (height, width):
                raise InputError(
                    f"Videonun {count}. karesinin boyutu değişti "
                    f"({frame.shape[1]}x{frame.shape[0]}, ilk kare {width}x{height}).")
            shown, edge_map, mask = edge_mask(frame, args, calibration)
            mask_writer.write(mask)
            if compare_writer is not None:
                compare_writer.write(draw_compare(shown, edge_map, mask, args,
                                                  calibration is not None))
            ratio_sum += mask_ratio(mask)
            count += 1
            ok, frame = cap.read()
    finally:
        cap.release()
        for writer in writers:
            writer.release()
    print(f"{path.name}: {count} kare, ortalama kenar maskesi %{100 * ratio_sum / count:.2f} "
          f"-> {mask_path}")


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
        description="Gri tonlu kareden Sobel-x ya da Canny ile ikili kenar maskesi üretir.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"Görüntü, görüntü klasörü ya da video (varsayılan: {INPUT_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--calibration", type=Path, default=CALIBRATION_PATH,
                   help="Kamera kalibrasyonu .npz; verilirse önce undistort yapılır "
                        "(varsayılan: yok)")
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
    p.add_argument("--sobel-scale", choices=("fixed", "max"), default=SOBEL_SCALE,
                   help=f"|gx| ölçekleme: sabit ya da karedeki en büyük değer (varsayılan: {SOBEL_SCALE})")
    p.add_argument("--sobel-range", type=int, nargs=2, default=SOBEL_RANGE,
                   metavar=("ALT", "UST"), help=f"Ölçekli |gx| eşiği (varsayılan: {SOBEL_RANGE})")
    p.add_argument("--canny-thresholds", type=float, nargs=2, default=CANNY_THRESHOLDS,
                   metavar=("ALT", "UST"), help=f"Canny çift eşiği (varsayılan: {CANNY_THRESHOLDS})")
    p.add_argument("--compare", action="store_true",
                   help="Girdi, kenar haritası ve maskeyi yan yana kaydet")
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
    if not 0 <= low <= high <= 255:
        raise InputError(f"--sobel-range 0 <= ALT <= UST <= 255 olmalı, verilen: {low} {high}")
    low, high = args.canny_thresholds
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
        raise InputError(f"--canny-thresholds 0 <= ALT <= UST olmalı, verilen: {low} {high}")
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

    if args.method == "canny":
        detail = f"Canny {args.canny_thresholds[0]:g}-{args.canny_thresholds[1]:g}"
    else:
        detail = (f"Sobel-x ksize {args.sobel_ksize}, ölçek {args.sobel_scale}, "
                  f"eşik {args.sobel_range[0]}-{args.sobel_range[1]}")
    print(f"Yöntem: {detail}; blur: {args.blur}; undistort: "
          f"{'var (' + str(args.calibration) + ')' if calibration else 'yok'}")
    if path.is_dir():
        process_folder(path, args.out_dir, args, calibration)
    elif path.suffix.lower() in VIDEO_EXTS:
        process_video(path, args.out_dir, args, calibration)
    else:
        process_image(path, args.out_dir, args, calibration)


if __name__ == "__main__":
    try:
        main()
    except InputError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
