"""5.4 Non-Maximum Suppression (NMS): IoU eşiğinin etkisini ölçme ve açıklama.

Amaç
    NMS, aynı nesne için üst üste binen kutulardan en güvenlisini tutup diğerlerini eler
    (IoU eşiğine göre). Problem: aynı araç için birden çok kutu çıkması, yani çift sayım.
    Neden: dedektörler doğal olarak çok aday kutu üretir. Tür: Kavramsal.
    Girdi -> çıktı: kutular + skorlar -> temizlenmiş kutular.
    İnşa: NMS ultralytics içinde otomatik çalışır, NMS kodu yazılmaz; iou parametresi
    ayarlanır ve ne işe yaradığı anlatılır. Bu modül bunu ölçerek yapar.

    IoU (kesişim / birleşim) iki kutunun ne kadar çakıştığıdır: 0 ayrık, 1 aynı kutu.
    NMS güvene göre sıralar; en güvenli kutu tutulur, onunla IoU'su eşiği AŞAN aynı sınıftan
    kutular silinir. Eşik yükseldikçe daha az kutu silinir:
      iou yüksek (0.9, 1.0) : aynı nesnenin kutuları kalır -> çift sayım
      iou düşük  (0.3)      : üst üste duran gerçek ayrı nesneler (kalabalık yaya, yan yana
                              araç) birleşir -> kaçırma
      ultralytics varsayılanı 0.7.

Akış
    1. Model bir kez yüklenir; girdi (görüntü, klasör ya da video) kare kare okunur.
    2. Her kare her iou değeri için ultralytics ile tahmin edilir (NMS onun içinde çalışır).
    3. Her iou için: kutu sayısı ve NMS'den sonra kalan "aynı sınıftan, IoU > 0.5" kutu
       çiftleri (çift sayım göstergesi) sayılır.
    4. Tablo .npz / .csv olarak kaydedilir, açıklama sayılarla yazdırılır; --compare ile her
       iou değeri için kutular yan yana çizilir.

Gereksinimde belirtilen araçlar:
    - ultralytics'in kendi NMS'i (model.predict, iou parametresi); NMS kodu yazılmadı.
    - iou parametresinin ayarlanması ve etkisinin anlatılması (--iou-values, açıklama).

Gereksinimde belirtilmeyen, eklenenler (kodda [EK] ile işaretli):
    - [EK] iou = 1.0 karşılaştırması: hiçbir kutu bastırılmaz, NMS'siz durumu (çift sayım) gösterir.
    - [EK] Çift sayım göstergesi: NMS sonrası aynı sınıftan IoU > 0.5 kutu çifti sayısı.
      IoU, ölçüm için numpy ile hesaplanır (NMS'te kullanılmaz).
    - [EK] Sınıflar arası NMS seçeneği (--agnostic): aynı aracın hem car hem truck kutusu
      gibi farklı sınıflı çift sayımı da bastırır; yakın farklı nesneleri birleştirme riski var.
    - [EK] Kutu sayısının iou'ya göre tablosu (.npz, .csv) ve panel görseli (--compare).
    Not: Her iou değeri için model yeniden çalıştırılır (CPU'da yavaş ama yalnızca analiz
         içindir); seçilen değer 5.1'de --iou ile kullanılır.
    Not: "Doğru" eşik veriye bağlıdır; doğru kutular (etiket) olmadan sayılar yol gösterir,
         karar görsele bakılarak verilir.
    Not: Görüntü kenarına değen kutular NMS'den SONRA kırpılır; NMS kırpılmamış (daha büyük)
         kutunun IoU'sunu kullanır. Kenardaki kutularda çıktıdan ölçülen IoU, NMS'in gördüğünden
         büyük olabilir (testte görüldü).
    Not: Örtülen nesnede YOLO çoğunlukla yalnız görünen kısmı kutular; öndeki ve arkadaki
         nesnenin kutuları az çakışır, düşük iou bile arkadakini silmeyebilir (testte IoU ~0).

Kullanım (konteyner içinde; varsayılan yollar proje köküne göredir):
    docker exec -it ads-cv-lab bash
    cd "/workspace/Carpisma_riski_ve_engel_farkindaligi/5.4 Non-Maximum Suppression (NMS)"
    python nms_iou_analizi.py --compare                         # INPUT_PATH, iou 0.3 0.5 0.7 0.9 1.0
    python nms_iou_analizi.py --input /workspace/data/kare.jpg --iou-values 0.45 0.7 --agnostic
    python nms_iou_analizi.py --help
    Seçilen değeri 5.1'de kullan: python yolo_tespit.py --iou 0.6

Okunan dosya: MODEL_PATH (.pt, 5.1 ile aynı model).

Kaydedilen dosyalar (ADS/outputs/<bu klasörün adı>/):
    <ad>_nms_analiz.npz
      iou_values     float64[K]     denenen IoU eşikleri
      counts         int64[K]       tüm karelerde toplam kutu sayısı
      overlap_pairs  int64[K]       NMS sonrası aynı sınıftan IoU > 0.5 kutu çifti sayısı
      class_counts   int64[K, 80]   sınıf başına kutu sayısı
      class_names    str[80]; n_frames int64; conf float64; agnostic bool
    <ad>_nms_analiz.csv     aynı tablo (iou, kutu, cift_sayim_cifti).
    <ad>_nms_karsilastirma.png   --compare: ilk karede her iou değeri için kutular yan yana.
"""

import argparse
import csv
import math
import os
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

# [DEĞİŞTİR] Video, görüntü ya da görüntü klasörü (5.1 ile aynı girdi).
INPUT_PATH = DATA_DIR / "yol.mp4"
MODEL_PATH = DATA_DIR / "yolo11n.pt"

# Denenecek NMS IoU eşikleri; 1.0 NMS'siz durumu gösterir, 0.7 ultralytics varsayılanı.
IOU_VALUES = (0.3, 0.5, 0.7, 0.9, 1.0)
CONF = 0.25                   # 5.1 ile aynı güven eşiği (NMS bu adayların üzerinde çalışır)
IMGSZ = 640
DEVICE = "cpu"
AGNOSTIC = False              # [EK] sınıflar arası NMS
DUPLICATE_IOU = 0.5           # [EK] çift sayım göstergesi: aynı sınıftan bu IoU'nun üstündeki çiftler

# Video: her FRAME_STEP karede bir, en çok MAX_FRAMES kare (her kare K kez çalıştırılır).
FRAME_STEP = 10
MAX_FRAMES = 20
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


def load_model(path):
    # 5.1 ile aynı: ağa çıkma ve kullanım verisi gönderme yok.
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    try:
        from ultralytics import YOLO, settings
    except ImportError:
        raise InputError("ultralytics kurulu değil.\n"
                         "  -> İmajı yeniden kur (host'ta): docker compose -f docker/docker-compose.yml build") from None
    try:
        if settings.get("sync", False):
            settings.update({"sync": False})
    except Exception:
        pass
    path = Path(path)
    if not path.is_file():
        raise InputError(f"Model dosyası bulunamadı: {path}\n"
                         "  -> 5.1'de indir: python yolo_tespit.py --download-model")
    try:
        return YOLO(str(path))
    except Exception as exc:
        raise InputError(f"Model yüklenemedi: {path} ({type(exc).__name__}: {exc})") from None


# --------------------------------------------------------------------------- ölçüm
def pairwise_iou(boxes):
    """[EK] Kutular arası IoU matrisi (yalnız ölçüm için; NMS ultralytics içinde)."""
    if len(boxes) == 0:
        return np.zeros((0, 0))
    x1, y1, x2, y2 = (boxes[:, i][:, None] for i in range(4))
    iw = np.clip(np.minimum(x2, x2.T) - np.maximum(x1, x1.T), 0, None)
    ih = np.clip(np.minimum(y2, y2.T) - np.maximum(y1, y1.T), 0, None)
    inter = iw * ih
    area = (x2 - x1) * (y2 - y1)
    return inter / np.maximum(area + area.T - inter, 1e-9)


def duplicate_pairs(boxes, cls, threshold):
    """[EK] Aynı sınıftan IoU > eşik kutu çifti sayısı (çift sayım göstergesi)."""
    iou = pairwise_iou(boxes)
    same = cls[:, None] == cls[None, :]
    upper = np.triu(np.ones_like(iou, dtype=bool), k=1)
    return int(np.count_nonzero((iou > threshold) & same & upper))


def predict(model, frame, iou, args):
    """Gereksinimin özü: ultralytics kendi NMS'ini bu iou ile uygular."""
    try:
        boxes = model.predict(source=frame, conf=args.conf, iou=iou, imgsz=args.imgsz, device=args.device,
                              agnostic_nms=args.agnostic, verbose=False)[0].boxes
    except Exception as exc:
        raise InputError(f"YOLO tahmini başarısız ({type(exc).__name__}: {exc})\n"
                         "  -> --device'ı kontrol et (imajdaki torch CPU sürümüdür).") from None
    return boxes.xyxy.cpu().numpy().astype(np.float64), boxes.cls.cpu().numpy().astype(np.int64), \
        boxes.conf.cpu().numpy()


# --------------------------------------------------------------------------- girdi
def iter_frames(path, args):
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise InputError(f"Klasörde görüntü yok: {path}")
        for file in files:
            img = cv2.imread(str(file), flags=cv2.IMREAD_COLOR)
            if img is None:
                raise InputError(f"Görüntü okunamadı: {file}")
            yield file.stem, img
        return
    if path.suffix.lower() not in VIDEO_EXTS:
        img = cv2.imread(str(path), flags=cv2.IMREAD_COLOR)
        if img is None:
            raise InputError(f"Görüntü okunamadı: {path}\n  -> Dosyanın bozuk olmadığını kontrol et.")
        yield path.stem, img
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InputError(f"Video açılamadı: {path}\n  -> Dosyanın bozuk olmadığını kontrol et.")
    index = used = 0
    try:
        while args.max_frames is None or used < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % args.frame_step == 0:
                used += 1
                yield f"kare{index:05d}", frame
            index += 1
    finally:
        cap.release()
    if used == 0:
        raise InputError(f"Videodan hiç kare okunamadı: {path}")


# --------------------------------------------------------------------------- görsel
def draw_panel(frame, boxes, cls, conf, names, title):
    width = frame.shape[1]
    fs = 0.6 * max(width, 640) / 1280.0 + 0.2
    th = max(2, round(3 * width / 1280.0))
    img = frame.copy()
    for (x1, y1, x2, y2), c, p in zip(boxes, cls, conf):
        rng = np.random.default_rng(int(c) * 7919 + 17)
        color = tuple(int(v) for v in rng.integers(60, 256, 3))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=th)
    org = (10, round(30 * fs / 0.6))
    cv2.putText(img, title, org, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, (0, 0, 0), thickness=th + 3, lineType=cv2.LINE_AA)
    cv2.putText(img, title, org, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, (0, 165, 255), thickness=th + 1, lineType=cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- ana akış
def run(args):
    path = Path(args.input)
    if not path.exists():
        raise InputError(f"Girdi bulunamadı: {path}\n"
                         "  -> --input ile doğru yolu ver ya da AYARLAR'daki INPUT_PATH'i düzelt.")
    model = load_model(args.model)
    names = model.names
    ious = sorted(set(args.iou_values))
    k = len(ious)
    counts = np.zeros(k, np.int64)
    pairs = np.zeros(k, np.int64)
    class_counts = np.zeros((k, len(names)), np.int64)
    n_frames, first = 0, None
    for name, frame in iter_frames(path, args):
        n_frames += 1
        panels = []
        for j, iou in enumerate(ious):
            boxes, cls, conf = predict(model, frame, iou, args)
            counts[j] += len(boxes)
            pairs[j] += duplicate_pairs(boxes, cls, args.duplicate_iou)
            np.add.at(class_counts[j], cls, 1)
            if first is None and args.compare:
                note = " (NMS etkisiz)" if iou >= 1.0 else (" (varsayilan)" if iou == 0.7 else "")
                panels.append(draw_panel(frame, boxes, cls, conf, names, f"iou {iou:g}{note}: {len(boxes)} kutu"))
        if first is None:
            first = name
            if args.compare:
                try:
                    args.out_dir.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise InputError(f"Çıktı klasörü oluşturulamadı: {args.out_dir} ({exc})") from None
                stem = path.stem if path.is_file() else path.name
                if not cv2.imwrite(str(args.out_dir / f"{stem}_nms_karsilastirma.png"), cv2.hconcat(panels)):
                    raise InputError(f"Görsel yazılamadı: {args.out_dir}")

    # Tablo
    print(f"{n_frames} kare, güven >= {args.conf:g}, sınıflar arası NMS: {'açık' if args.agnostic else 'kapalı'}")
    print(f"{'iou':>6} | {'kutu':>6} | aynı sınıftan IoU>{args.duplicate_iou:g} çift (çift sayım göstergesi) | sınıflar")
    for j, iou in enumerate(ious):
        top = np.argsort(-class_counts[j])[:4]
        cls_text = ", ".join(f"{names[c]} {class_counts[j, c]}" for c in top if class_counts[j, c])
        print(f"{iou:6.2f} | {counts[j]:6d} | {pairs[j]:6d} | {cls_text}")

    # Açıklama (İnşa: iou'nun ne işe yaradığını anlat), ölçülen sayılarla.
    lo, hi = ious[0], ious[-1]
    print("\nNe işe yarar: NMS güvene göre sıralar, en güvenli kutuyu tutar ve onunla IoU'su eşiği aşan "
          "aynı sınıftan kutuları siler.")
    print(f"  Bu girdide iou {hi:g}'de {counts[-1]} kutu ve {pairs[-1]} çakışan çift var"
          + (" (iou 1.0 hiçbir kutuyu silmez: NMS'siz durum, çift sayım)." if hi >= 1.0 else "."))
    print(f"  iou {lo:g}'de {counts[0]} kutu kalıyor; eşik düştükçe üst üste duran ayrı nesneler de "
          "birleşebilir (kaçırma).")
    if 0.7 in ious:
        j = ious.index(0.7)
        print(f"  Varsayılan 0.7: {counts[j]} kutu, {pairs[j]} çakışan çift. Çift sayım görürsen eşiği düşür, "
              "kalabalıkta nesne kaybedersen yükselt; seçtiğin değeri 5.1'de --iou ile ver.")

    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem if path.is_file() else path.name
        out_path = args.out_dir / f"{stem}_nms_analiz.npz"
        np.savez(out_path, iou_values=np.array(ious, dtype=np.float64), counts=counts, overlap_pairs=pairs,
                 class_counts=class_counts, class_names=np.array([names[i] for i in range(len(names))], dtype=np.str_),
                 n_frames=np.int64(n_frames), conf=np.float64(args.conf), agnostic=np.bool_(args.agnostic))
        with open(args.out_dir / f"{stem}_nms_analiz.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iou", "kutu", "cift_sayim_cifti"])
            for j, iou in enumerate(ious):
                w.writerow([f"{iou:g}", int(counts[j]), int(pairs[j])])
    except OSError as exc:
        raise InputError(f"Sonuç yazılamadı: {args.out_dir} ({exc})") from None
    print(f"\nKaydedildi: {out_path} (+ .csv)")


class TurkishArgumentParser(argparse.ArgumentParser):
    """Argüman hatasında "HATA: ..." yazar ve 1 koduyla çıkar."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"HATA: Geçersiz komut satırı argümanı: {message}\n"
              "  -> Seçenekler için --help kullan.", file=sys.stderr)
        sys.exit(1)


def parse_args(argv=None):
    p = TurkishArgumentParser(
        description="NMS IoU eşiğinin etkisini ölçer: her iou için kutu sayısı ve çift sayım göstergesi.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"Video, görüntü ya da görüntü klasörü (varsayılan: {INPUT_PATH})")
    p.add_argument("--model", type=Path, default=MODEL_PATH, help=f"YOLO ağırlıkları (varsayılan: {MODEL_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--iou-values", type=float, nargs="+", default=list(IOU_VALUES),
                   help=f"Denenecek IoU eşikleri, 0-1 (varsayılan: {' '.join(map(str, IOU_VALUES))})")
    p.add_argument("--conf", type=float, default=CONF, help=f"Güven eşiği (varsayılan: {CONF})")
    p.add_argument("--imgsz", type=int, default=IMGSZ, help=f"Ağ giriş boyutu (varsayılan: {IMGSZ})")
    p.add_argument("--device", default=DEVICE, help=f"Cihaz (varsayılan: {DEVICE})")
    p.add_argument("--agnostic", action="store_true", default=AGNOSTIC,
                   help="Sınıflar arası NMS (farklı sınıftan çakışan kutuları da bastırır)")
    p.add_argument("--duplicate-iou", type=float, default=DUPLICATE_IOU,
                   help=f"Çift sayım göstergesi için IoU sınırı (varsayılan: {DUPLICATE_IOU})")
    p.add_argument("--compare", action="store_true", help="İlk karede her iou için kutuları yan yana çiz")
    p.add_argument("--frame-step", type=int, default=FRAME_STEP, help=f"Videoda her N karede bir (varsayılan: {FRAME_STEP})")
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES, help=f"Videoda en çok kare (varsayılan: {MAX_FRAMES})")
    return p.parse_args(argv)


def validate_args(args):
    for v in args.iou_values:
        if not (math.isfinite(v) and 0 < v <= 1):
            raise InputError(f"--iou-values 0 < değer <= 1 olmalı, verilen: {v}")
    for name, v in (("--conf", args.conf), ("--duplicate-iou", args.duplicate_iou)):
        if not (math.isfinite(v) and 0 < v <= 1):
            raise InputError(f"{name} 0 < değer <= 1 olmalı, verilen: {v}")
    if args.imgsz < 32 or args.imgsz % 32:
        raise InputError(f"--imgsz 32'nin pozitif katı olmalı, verilen: {args.imgsz}")
    if args.frame_step <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        raise InputError("--frame-step ve --max-frames pozitif olmalı.")


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    run(args)


if __name__ == "__main__":
    try:
        main()
    except InputError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
