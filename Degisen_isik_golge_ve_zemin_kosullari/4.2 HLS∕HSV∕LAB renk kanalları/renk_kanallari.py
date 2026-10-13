"""4.2 HLS/HSV/LAB renk kanalları: kanalları birlikte kullanan renk maskesi ve kanal kararlılığı.

Amaç
    Kavram ve uygulama 2.1 ile aynıdır: renk uzayı kanallarını eşikleyip ikili şerit maskesi
    üretmek. Bu başlıktaki fark kanal seçimidir: gölgede L düşer; HLS-S ve LAB-B sarı çizgide
    daha kararlı kalır, LAB-B sarıyı ışıktan büyük ölçüde bağımsız yakalar. Kanallar tek tek
    değil birlikte kullanılır (birleştirme 4.4). Tür: Kavramsal.
    Girdi -> çıktı: renkli kare -> birleşik kanal maskesi + kanal kararlılık istatistiği.

Akış
    1. Girdi (tek görüntü, görüntü klasörü ya da video) kare kare okunur.
    2. Kare HLS ve LAB'a çevrilir (2.1'in kodu); istenirse L kanalına CLAHE uygulanır.
    3. Her kanal ayrı eşiklenir:
         L : beyaz çizgi  (HLS-L, ya da CLAHE-L)
         S : sarı çizgi   (HLS-S yüksek ve ton sarı aralığında)
         B : sarı çizgi   (LAB-B yüksek)
    4. Seçilen kanal maskeleri OR'lanır -> birleşik maske (0 / 255) kaydedilir.
    5. Kanal başına maske doluluğu kare kare kaydedilir; videoda ortalama, std ve değişim
       katsayısı yazdırılır ("zor videoda hangi kanal kararlı kalıyor?").

Gereksinimde belirtilen araçlar:
    - Sıra 4'ün (2.1) kodu: cv2.cvtColor (HLS, LAB), cv2.inRange, maskeleri OR'lama.
    - Kanal seçimi: L (beyaz), HLS-S ve LAB-B (sarı); kanallar birlikte kullanılır (--channels).
    - İsteğe bağlı cv2.createCLAHE, L kanalına (--clahe).
    - Zor videoda kanal kararlılığına bakma: kanal başına doluluk, std, değişim katsayısı.

Gereksinimde belirtilmeyen, eklenenler (kodda [EK] ile işaretli):
    - [EK] S kanalında ton (H) koşulu: S tek başına gökyüzü, turuncu gibi her doygun rengi
      yakalar (2.1 testinde S >= 170 gökyüzünün %100'ünü aldı); ton sarı aralığıyla sınırlanır.
    - [EK] Kanal kararlılık istatistiği (.npz ve özet tablo): doğru maske olmadan kanalları
      sayıyla karşılaştırmayı sağlar. Gölgeye giren videoda L maskesinin doluluğu düşer,
      S ve B'ninki daha az değişir.
    - [EK] Kanal katkısını renkle gösteren 3x2 karşılaştırma görseli.
    - [EK] Klasör / video girişi; kayıpsız maske videosu (FFV1, .mkv).
    Not: S kanalı beyaz çizgi için kullanılmadı: ölçümde beyaz boyanın S'si aydınlıkta ~40,
         gölgede ~7, asfaltın ~9-15; gölgede değişmese bile beyazı asfalttan ayıramaz.
         İnşa'daki "S sarı/beyaz çizgide kararlı" ifadesi bu veride yalnız sarı için doğru çıktı.
    Not: CLAHE varsayılan kapalı: tam karede (gökyüzü, beton varken) düz parlak bölgeleri
         şerit sayar (ölçümde L>=160 ile sahte beyaz 1 -> ~233 bin). Kuş bakışı (yalnız yol)
         görüntüde ise --clahe --white-l-range 160 255 gölgedeki beyazın ~%48'ini sahte
         piksel olmadan yakaladı (CLAHE'siz %0); kuş bakışı girdide denenmeye değer.
    Not: Maviye kayan (gökyüzü ışıklı) gölgede S sarıda 255 -> ~122'ye düşer ve asfaltın
         S'si ~85'e çıkar; LAB-B sarıda 209 -> ~166 ile daha kararlıdır. S, ton koşuluyla
         yedek olarak kalır; ikisi birlikte kullanılır.
    Not: HSV kullanılmadı: HSV-V gölgede L gibi düşer (beyaz 238 -> 144), HSV-S beyazda
         HLS-S ile aynı sorunu taşır; sarı için HLS-S + LAB-B yeterli.

Kullanım (konteyner içinde; varsayılan yollar proje köküne göredir):
    docker exec -it ads-cv-lab bash
    cd "/workspace/Degisen_isik_golge_ve_zemin_kosullari/4.2 HLS∕HSV∕LAB renk kanalları"
    python renk_kanallari.py --compare                       # INPUT_PATH, kanallar L S B
    python renk_kanallari.py --input /workspace/data/zor.mp4 --compare
    python renk_kanallari.py --channels B                    # tek kanal (karşılaştırma için)
    python renk_kanallari.py --clahe --white-l-range 160 255
    python renk_kanallari.py --help

Kaydedilen dosyalar (ADS/outputs/<bu klasörün adı>/):
    <ad>_kanal_maske.png       birleşik maske: tek kanal uint8, girdiyle aynı boyut;
                               0 = arka plan, 255 = şerit pikseli (mantıksal 1).
    <klasör>/...               klasör girişinde aynı biçim, alt klasörde.
    <ad>_kanal_maske.mkv       video girişinde: FFV1 kayıpsız, gri; okurken frame[:, :, 0] > 0.
    <ad>_kanal_istatistik.npz  frame (str[N]), channels (str[K]: "L", "S", "B"),
                               ratios (float64[N, K]: kanal maskesi doluluğu, 0-1),
                               combined (float64[N]: birleşik maske doluluğu), clahe (bool).
    <ad>_kanal_karsilastirma.png / .mp4   --compare: girdi | L | S, B | kanal katkısı | birleşik.
"""

import argparse
import math
import sys
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

# [DEĞİŞTİR] Renkli kare: tek görüntü, görüntü klasörü ya da video (zor video önerilir).
INPUT_PATH = DATA_DIR / "yol_karesi.jpg"

# Birlikte kullanılacak kanallar: "L" (beyaz), "S" (sarı, HLS-S + ton), "B" (sarı, LAB-B).
CHANNELS = ("L", "S", "B")

# [DEĞİŞTİR] Eşikler (OpenCV 8 bit; H 0-180, diğerleri 0-255).
WHITE_L_RANGE = (200, 255)    # HLS-L (CLAHE açıksa CLAHE-L; o zaman düşürmek gerekebilir)
YELLOW_S_MIN = 100            # HLS-S alt eşiği
YELLOW_H_RANGE = (15, 35)     # sarı ton aralığı (S kanalı için)
YELLOW_B_RANGE = (155, 255)   # LAB-B: 128 nötr, büyüdükçe sarı

# İsteğe bağlı CLAHE (L kanalına). Varsayılan kapalı; gerekçe dosya başındaki Not'ta.
# Kuş bakışı (yalnız yol) girdide açıp WHITE_L_RANGE'i (160, 255) yapmayı dene.
USE_CLAHE = False
CLAHE_CLIP = 2.0
CLAHE_TILES = 8               # karo ızgarası (8 x 8), çözünürlükten bağımsız

# Video: en çok kaç kare işlensin (None = tümü).
MAX_FRAMES = None
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
CHANNEL_TR = {"L": "L (beyaz)", "S": "S+ton (sarı)", "B": "LAB-B (sarı)"}
CHANNEL_COLOR = {"L": (255, 255, 255), "S": (0, 140, 255), "B": (0, 255, 255)}   # BGR


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


# --------------------------------------------------------------------------- kanallar
def channel_images(frame_bgr, clahe):
    """2.1'in dönüşümleri: HLS ve LAB kanalları; istenirse L'ye CLAHE."""
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    hue, light, sat = cv2.split(hls)
    if clahe is not None:
        light = clahe.apply(light)    # isteğe bağlı CLAHE: yerel kontrast
    return {"H": hue, "L": light, "S": sat, "B": cv2.split(lab)[2]}


def channel_masks(ch, args):
    """Kanal başına maske (0/255) ve birleşik maske (OR)."""
    masks = {}
    if "L" in args.channels:
        masks["L"] = cv2.inRange(ch["L"], args.white_l_range[0], args.white_l_range[1])
    if "S" in args.channels:
        # [EK] Ton koşulu: S tek başına her doygun rengi (gökyüzü, turuncu) yakalar.
        masks["S"] = cv2.bitwise_and(cv2.inRange(ch["S"], args.yellow_s_min, 255),
                                     cv2.inRange(ch["H"], args.yellow_h_range[0], args.yellow_h_range[1]))
    if "B" in args.channels:
        masks["B"] = cv2.inRange(ch["B"], args.yellow_b_range[0], args.yellow_b_range[1])
    combined = np.zeros_like(ch["L"])
    for mask in masks.values():
        combined = cv2.bitwise_or(combined, mask)     # kanallar birlikte (OR)
    return masks, combined


def ratio(mask):
    return cv2.countNonZero(mask) / mask.size


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


# [EK] 3x2: girdi | L | S, B | kanal katkısı | birleşik maske.
def draw_compare(frame, ch, masks, combined, args):
    width = frame.shape[1]
    fs = 0.8 * width / 1280.0
    th = max(1, round(2 * width / 1280.0))
    gray = lambda c: cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
    contrib = np.zeros_like(frame)
    for name in ("B", "S", "L"):                      # L en üstte
        if name in masks:
            contrib[masks[name] > 0] = CHANNEL_COLOR[name]
    l_name = "CLAHE-L" if args.clahe else "HLS-L"
    panels = [(frame.copy(), "Girdi"), (gray(ch["L"]), l_name), (gray(ch["S"]), "HLS-S"),
              (gray(ch["B"]), "LAB-B"),
              (contrib, "Kanal katkisi: beyaz L, turuncu S, sari B"),
              (gray(combined), f"Birlesik ({'+'.join(args.channels)}): %{100 * ratio(combined):.2f}")]
    for img, text in panels:
        put_label(img, text, fs, th)
    imgs = [p[0] for p in panels]
    return cv2.vconcat([cv2.hconcat(imgs[:3]), cv2.hconcat(imgs[3:])])


# --------------------------------------------------------------------------- işleme
class StatsTable:
    """[EK] Kare başına kanal maskesi doluluğu; .npz olarak kaydedilir."""

    def __init__(self, channels):
        self.channels = tuple(channels)
        self.frames, self.ratios, self.combined = [], [], []

    def add(self, name, masks, combined):
        self.frames.append(name)
        self.ratios.append([ratio(masks[c]) for c in self.channels])
        self.combined.append(ratio(combined))

    def save(self, path, clahe):
        np.savez(path, frame=np.array(self.frames, dtype=np.str_),
                 channels=np.array(self.channels, dtype=np.str_),
                 ratios=np.array(self.ratios, dtype=np.float64).reshape(len(self.frames), -1),
                 combined=np.array(self.combined, dtype=np.float64), clahe=np.bool_(clahe))

    def summary(self):
        r = np.array(self.ratios).reshape(len(self.frames), -1)
        lines = ["Kanal kararlılığı (doluluk %; değişim katsayısı = std / ortalama, düşük = kararlı):"]
        for i, c in enumerate(self.channels):
            mean, std = 100 * r[:, i].mean(), 100 * r[:, i].std()
            cv = std / mean if mean > 0 else math.nan
            lines.append(f"  {CHANNEL_TR[c]:14s} ortalama %{mean:.2f}, std %{std:.2f}, "
                         f"en az %{100 * r[:, i].min():.2f}, en çok %{100 * r[:, i].max():.2f}, "
                         f"değişim katsayısı {cv:.2f}")
        comb = 100 * np.array(self.combined)
        lines.append(f"  {'birleşik':14s} ortalama %{comb.mean():.2f}, std %{comb.std():.2f}")
        return "\n".join(lines)


def format_ratios(masks, combined):
    return ", ".join(f"{c} %{100 * ratio(m):.2f}" for c, m in masks.items()) + \
        f" | birleşik %{100 * ratio(combined):.2f}"


def process_image(path, out_dir, args, clahe, table, name=None):
    frame = read_image(path)
    ch = channel_images(frame, clahe)
    masks, combined = channel_masks(ch, args)
    name = name or path.stem
    write_image(out_dir / f"{name}_kanal_maske.png", combined)
    if args.compare:
        write_image(out_dir / f"{name}_kanal_karsilastirma.png",
                    draw_compare(frame, ch, masks, combined, args))
    table.add(name, masks, combined)
    print(f"{path.name}: {format_ratios(masks, combined)}")


def process_video(path, out_dir, args, clahe, table):
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
    writers = []
    try:
        # [EK] FFV1 kayıpsızdır; mp4 maskenin 0/255 değerlerini bozar.
        mask_writer = open_writer(out_dir / f"{path.stem}_kanal_maske.mkv", "FFV1", fps,
                                  (width, height), is_color=False)
        writers.append(mask_writer)
        compare_writer = None
        if args.compare:
            compare_writer = open_writer(out_dir / f"{path.stem}_kanal_karsilastirma.mp4", "mp4v",
                                         fps, (3 * width, 2 * height), is_color=True)
            writers.append(compare_writer)
        count = 0
        while ok and (args.max_frames is None or count < args.max_frames):
            if frame.shape[:2] != (height, width):
                raise InputError(
                    f"Videonun {count}. karesinin boyutu değişti "
                    f"({frame.shape[1]}x{frame.shape[0]}, ilk kare {width}x{height}).")
            ch = channel_images(frame, clahe)
            masks, combined = channel_masks(ch, args)
            mask_writer.write(combined)
            if compare_writer is not None:
                compare_writer.write(draw_compare(frame, ch, masks, combined, args))
            table.add(f"kare{count:05d}", masks, combined)
            count += 1
            ok, frame = cap.read()
    finally:
        cap.release()
        for w in writers:
            w.release()
    print(f"{path.name}: {count} kare işlendi")


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
        description="HLS-L, HLS-S ve LAB-B kanallarını birlikte eşikleyip renk maskesi üretir; "
                    "kanal kararlılığını ölçer.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"Görüntü, görüntü klasörü ya da video (varsayılan: {INPUT_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--channels", nargs="+", choices=("L", "S", "B"), default=list(CHANNELS),
                   help=f"Birlikte kullanılacak kanallar (varsayılan: {' '.join(CHANNELS)})")
    p.add_argument("--white-l-range", type=int, nargs=2, default=WHITE_L_RANGE, metavar=("ALT", "UST"),
                   help=f"Beyaz için L (ya da CLAHE-L) aralığı (varsayılan: {WHITE_L_RANGE})")
    p.add_argument("--yellow-s-min", type=int, default=YELLOW_S_MIN,
                   help=f"Sarı için HLS-S alt eşiği (varsayılan: {YELLOW_S_MIN})")
    p.add_argument("--yellow-h-range", type=int, nargs=2, default=YELLOW_H_RANGE, metavar=("ALT", "UST"),
                   help=f"Sarı ton aralığı, H 0-180 (varsayılan: {YELLOW_H_RANGE})")
    p.add_argument("--yellow-b-range", type=int, nargs=2, default=YELLOW_B_RANGE, metavar=("ALT", "UST"),
                   help=f"Sarı için LAB-B aralığı (varsayılan: {YELLOW_B_RANGE})")
    p.add_argument("--clahe", action="store_true", default=USE_CLAHE,
                   help="L kanalına CLAHE uygula (varsayılan: kapalı)")
    p.add_argument("--clahe-clip", type=float, default=CLAHE_CLIP,
                   help=f"CLAHE kırpma sınırı (varsayılan: {CLAHE_CLIP})")
    p.add_argument("--clahe-tiles", type=int, default=CLAHE_TILES,
                   help=f"CLAHE karo ızgarası, N x N (varsayılan: {CLAHE_TILES})")
    p.add_argument("--compare", action="store_true",
                   help="Girdi, kanallar, kanal katkısı ve birleşik maskeyi 3x2 kaydet")
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                   help="Videoda en çok kaç kare işlensin (varsayılan: tümü)")
    args = p.parse_args(argv)
    args.channels = tuple(dict.fromkeys(args.channels))   # sırayı koru, tekrarları at
    return args


def validate_args(args):
    for name, (low, high), top in (("--white-l-range", args.white_l_range, 255),
                                   ("--yellow-b-range", args.yellow_b_range, 255),
                                   ("--yellow-h-range", args.yellow_h_range, 180)):
        if not 0 <= low <= high <= top:
            raise InputError(f"{name} 0 <= ALT <= UST <= {top} olmalı, verilen: {low} {high}")
    if not 0 <= args.yellow_s_min <= 255:
        raise InputError(f"--yellow-s-min 0-255 aralığında olmalı, verilen: {args.yellow_s_min}")
    if not (math.isfinite(args.clahe_clip) and args.clahe_clip > 0):
        raise InputError(f"--clahe-clip pozitif olmalı, verilen: {args.clahe_clip}")
    if args.clahe_tiles <= 0:
        raise InputError(f"--clahe-tiles pozitif olmalı, verilen: {args.clahe_tiles}")
    if args.max_frames is not None and args.max_frames <= 0:
        raise InputError(f"--max-frames pozitif olmalı, verilen: {args.max_frames}")


# --------------------------------------------------------------------------- ana akış
def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
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
    clahe = (cv2.createCLAHE(clipLimit=args.clahe_clip, tileGridSize=(args.clahe_tiles, args.clahe_tiles))
             if args.clahe else None)
    print(f"Kanallar: {', '.join(CHANNEL_TR[c] for c in args.channels)}; "
          f"CLAHE: {'açık (clip ' + format(args.clahe_clip, 'g') + ')' if clahe else 'kapalı'}")

    table = StatsTable(args.channels)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise InputError(f"Klasörde görüntü yok: {path}\n"
                             f"  -> Desteklenen uzantılar: {', '.join(sorted(IMAGE_EXTS))}")
        out_dir = args.out_dir / path.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            process_image(file, out_dir, args, clahe, table)
        stats_path = out_dir / f"{path.name}_kanal_istatistik.npz"
    elif path.suffix.lower() in VIDEO_EXTS:
        process_video(path, args.out_dir, args, clahe, table)
        stats_path = args.out_dir / f"{path.stem}_kanal_istatistik.npz"
    else:
        process_image(path, args.out_dir, args, clahe, table)
        stats_path = args.out_dir / f"{path.stem}_kanal_istatistik.npz"
    if len(table.frames) > 1:
        print(table.summary())
    try:
        table.save(stats_path, args.clahe)
    except OSError as exc:
        raise InputError(f"İstatistik yazılamadı: {stats_path} ({exc})") from None
    print(f"Kaydedildi: maske(ler) ve {stats_path}")


if __name__ == "__main__":
    try:
        main()
    except InputError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
