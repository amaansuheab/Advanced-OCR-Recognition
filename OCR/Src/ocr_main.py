import sys
import re
import csv
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pytesseract
from PIL import Image
from paddleocr import PaddleOCR

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}
DEFAULT_OUTPUT        = "bill_titles3211.csv"
TITLE_CROP            = 0.38
BODY_START            = 0.33
TARGET_TOP_PX         = 800    # max px for top crop fed to PaddleOCR — smaller = faster
TARGET_BOT_PX         = 900    # max px for body crop fed to Tesseract
SKEW_THRESHOLD        = 1.5    # degrees — skip deskew if below this, saves ~50ms

# Single global OCR engine — loaded once, reused for every image
paddle_ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False,
                        det_db_score_mode="fast", det_db_thresh=0.3,
                        det_db_box_thresh=0.5, rec_batch_num=6)

# Single global thread pool — created once, not per image
_POOL = ThreadPoolExecutor(max_workers=2)


# ── FAST ROI ──────────────────────────────────────────────────────────────────

def _find_bill_roi(gray: np.ndarray) -> np.ndarray:
    h, w     = gray.shape
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    row_means = thresh.mean(axis=1)
    col_means = thresh.mean(axis=0)
    threshold = 130
    rows = np.where(row_means > threshold)[0]
    cols = np.where(col_means > threshold)[0]
    if len(rows) < h * 0.1 or len(cols) < w * 0.1:
        rows = np.where(row_means > 80)[0]
        cols = np.where(col_means > 80)[0]
    if len(rows) == 0 or len(cols) == 0:
        return gray
    y1 = max(0, int(rows[0])  - 5)
    y2 = min(h, int(rows[-1]) + 5)
    x1 = max(0, int(cols[0])  - 5)
    x2 = min(w, int(cols[-1]) + 5)
    if x1 > w * 0.05 or x2 < w * 0.95 or y1 > h * 0.05 or y2 < h * 0.95:
        return gray[y1:y2, x1:x2]
    return gray


# ── IMAGE PREP ────────────────────────────────────────────────────────────────

def _preprocess(gray: np.ndarray) -> np.ndarray:
    gray  = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    return cv2.filter2D(gray, -1,
                        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32))


def _detect_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=80, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 - x1 == 0:
            continue
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < a < 45:
            angles.append(a)
    return statistics.median(angles) if angles else 0.0


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    M    = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _resize_to(gray: np.ndarray, max_px: int) -> np.ndarray:
    h, w = gray.shape
    if max(h, w) <= max_px:
        return gray
    scale = max_px / max(h, w)
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def load_and_split(image_path: str):
    img  = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _find_bill_roi(gray)

    # Detect skew once on a downscaled version — cheap
    small = _resize_to(gray, 600)
    angle = _detect_skew_angle(small)
    if abs(angle) >= SKEW_THRESHOLD:
        gray = _rotate(gray, angle)

    h, w      = gray.shape
    split_top = int(h * TITLE_CROP)
    split_bot = int(h * BODY_START)

    top_gray = _resize_to(gray[:split_top, :], TARGET_TOP_PX)
    top_gray = _preprocess(top_gray)

    bot_gray = _resize_to(gray[split_bot:, :], TARGET_BOT_PX)
    bot_gray = _preprocess(bot_gray)

    return top_gray, bot_gray


# ── OCR ───────────────────────────────────────────────────────────────────────

def _run_paddle(gray: np.ndarray):
    result = paddle_ocr.ocr(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), cls=False)
    words  = []
    if not result or not result[0]:
        return words
    for line in result[0]:
        bbox, (text, conf) = line
        text = text.strip()
        if not text or conf < 0.25:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        words.append({
            "text":   text,   "conf":   float(conf),
            "left":   min(xs),"top":    min(ys),
            "right":  max(xs),"bottom": max(ys),
            "height": max(ys) - min(ys),
        })
    return words


def _run_tesseract(gray: np.ndarray) -> str:
    data = pytesseract.image_to_data(
        Image.fromarray(gray),
        output_type=pytesseract.Output.DICT,
        config="--psm 11 --oem 3"   # psm 11 = sparse text, faster than psm 6
    )
    lines_map = {}
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text or int(data["conf"][i]) < 30:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines_map.setdefault(key, []).append(text)
    return "\n".join(" ".join(ws) for ws in lines_map.values())


# ── LINE GROUPING ─────────────────────────────────────────────────────────────

def _group_lines(words: list, img_w: int):
    if not words:
        return []
    words   = sorted(words, key=lambda w: w["top"])
    buckets = [[words[0]]]
    for w in words[1:]:
        last    = buckets[-1][-1]
        cy_last = (last["top"] + last["bottom"]) / 2
        cy_curr = (w["top"]   + w["bottom"])   / 2
        tol     = max(last["height"], w["height"]) * 0.6
        if abs(cy_curr - cy_last) <= tol:
            buckets[-1].append(w)
        else:
            buckets.append([w])
    lines = []
    for bucket in buckets:
        bucket = sorted(bucket, key=lambda w: w["left"])
        l, r   = min(w["left"] for w in bucket), max(w["right"] for w in bucket)
        lines.append({
            "words":      bucket,
            "top":        min(w["top"]    for w in bucket),
            "bottom":     max(w["bottom"] for w in bucket),
            "avg_height": statistics.mean(w["height"] for w in bucket),
            "text":       " ".join(w["text"] for w in bucket),
            "center_x":   (l + r) / 2,
            "span":       r - l,
            "img_w":      img_w,
            "avg_conf":   statistics.mean(w["conf"] for w in bucket),
        })
    return sorted(lines, key=lambda ln: ln["top"])


# ── SCORING ───────────────────────────────────────────────────────────────────

NOISE_RE = re.compile(
    r"^(paid|reprint|receipt|invoice|copy|duplicate|void|draft|"
    r"thank\s*you|subtotal|sub\s*total|total|balance\s*due|amount\s*due|"
    r"cash|change|tax|tip|gratuity|discount|refund|welcome|"
    r"please\s*come\s*again|call\s*again|thanks|x{4,}|\*{4,}|-{4,}|={4,}|"
    r"your\s*guest\s*number|see\s*back|survey|chance\s*to\s*win|"
    r"pickup|take.?out|carry.?out|walk.?in|dine.?in|open\s*daily|"
    r"hours|we\s*accept|visa|mastercard|amex|www\.|http|customer\s*copy|"
    r"minimum|credit|debit|approved|authorization|april|january|february|"
    r"march|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE
)

ADDRESS_LINE_RE = re.compile(
    r"^\d+\s+\w+.*(blvd|ave|st|rd|dr|hwy|pkwy|way|lane|ln|place|pl|court|ct|"
    r"street|road|drive|highway|parkway)\b",
    re.IGNORECASE
)

PURE_DATA_RE = re.compile(r"^[\d\s\(\)\-\.\,\/\:\*\#]+$")

# US state abbreviations — 2 capital letters at end of word e.g. "EverettWA", "LOSANGELESCA"
STATE_SUFFIX_RE = re.compile(
    r",?\s*(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|"
    r"MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|"
    r"VT|VA|WA|WV|WI|WY|DC)\.?\s*\d{0,5}$",
    re.IGNORECASE
)

# City+State smashed together with no space e.g. "EverettWA", "HamdenCT"
CITYSTATE_RE = re.compile(
    r"[a-z](AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|"
    r"MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|"
    r"VT|VA|WA|WV|WI|WY|DC)\.?\s*\d{0,5}$"
)

VENUE_RE = re.compile(
    r"\b(restaurant|cafe|café|coffee|diner|bistro|brasserie|tavern|bar|grill|"
    r"grille|kitchen|eatery|buffet|pizzeria|pizza|sushi|bakery|steakhouse|"
    r"smokehouse|bbq|noodle|ramen|pho|thai|chinese|indian|mexican|italian|"
    r"japanese|korean|vietnamese|mediterranean|fusion|cantina|taqueria|pub|"
    r"lounge|market|deli|wings|burgers|burger|chicken|seafood|steak|wok|"
    r"bowl|brew|brewery|provisions|trading|company|express|inc|llc|ltd)\b",
    re.IGNORECASE
)

_SPLIT_RE = [
    re.compile(r"\b(table|server|cashier|clerk|station|seat|terminal|tab|pos)\s*[:#\d]", re.IGNORECASE),
    re.compile(r"\b(order|check|ticket|ref|auth)\s*[:#\d]", re.IGNORECASE),
    re.compile(r"\b(dine\s*in|take\s*out|takeout|carry\s*out|walk\s*in|delivery)\b", re.IGNORECASE),
    re.compile(r"\b(blvd|ave|rd|dr|hwy|pkwy)\b", re.IGNORECASE),
    re.compile(r"\s\d{3,5}\s+[A-Z]"),
    re.compile(r"\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4}"),
    re.compile(r"\b\d{5}(-\d{4})?\b"),
    re.compile(r"\b(guests?|party|covers?)\s*[:#\d]", re.IGNORECASE),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b", re.IGNORECASE),
    re.compile(r"\b(store\s*#|store\s*no|unit\s*#|ks\s*#)\s*\d+", re.IGNORECASE),
    re.compile(r"#\s*\d{2,}"),
    re.compile(r"\b\d{2,}[-/]\d{2,}[-/]\d{2,}\b"),   # date variants
]


def clean_title(text: str) -> str:
    # Step 1 — cut at first stop-word boundary
    earliest = len(text)
    for pat in _SPLIT_RE:
        m = pat.search(text)
        if m and m.start() < earliest:
            after          = text[m.start(): m.start() + 50]
            venue_in_after = VENUE_RE.search(after)
            earliest       = m.start() + venue_in_after.end() if venue_in_after else m.start()
    text = text[:earliest].strip()

    # Step 2 — strip trailing state abbreviation + zip  e.g. "CAFE PARISIEN LARCHMONT LLC LOSANGELESCA"
    text = STATE_SUFFIX_RE.sub("", text).strip()

    # Step 3 — strip CityState smash  e.g. "Katana Sushi EverettWA" -> "Katana Sushi"
    m = CITYSTATE_RE.search(text)
    if m:
        text = text[:m.start() + 1].strip()

    # Step 4 — remove embedded numbers that are clearly not part of name
    #   keep: "7-Eleven", "A1 Sauce", "Route 66"
    #   remove: standalone numbers, prices, percentages mid-string
    text = re.sub(r"\s+\d+\.\d+", "", text)          # prices e.g. "45.00"
    text = re.sub(r"\s+\d{3,}", "", text)             # long standalone numbers
    text = re.sub(r"(?<![A-Za-z])\d+(?![A-Za-z])\s", " ", text)  # isolated numbers

    # Step 5 — strip trailing/leading junk
    text = re.sub(r"[&,\-/\\|#@\d\.]+$", "", text).strip()
    text = re.sub(r"^[#\d\s\-\*\.]+", "", text).strip()

    # Step 6 — remove special characters except apostrophe and ampersand in names
    text = re.sub(r"[^\w\s'&\-\.]", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def _score(line, all_lines, img_h, img_w) -> float:
    text  = line["text"]
    rel_y = line["top"] / img_h

    if rel_y <= 0.30:
        pos_score = 1.0
    elif rel_y <= 0.70:
        pos_score = 1.0 - ((rel_y - 0.30) / 0.40) * 0.5
    else:
        pos_score = max(0.0, 0.5 - (rel_y - 0.70) * 2.0)

    all_h      = sorted(l["avg_height"] for l in all_lines if l["avg_height"] > 0)
    med_h      = statistics.median(all_h) if all_h else 1
    p85_h      = all_h[int(len(all_h) * 0.85)] if all_h else 1
    size_score = min(line["avg_height"] / med_h / 2.5, 1.0)
    if line["avg_height"] >= p85_h:
        size_score = min(size_score + 0.3, 1.0)

    toks = re.findall(r"[A-Za-z]+", text)
    if not toks:
        case_score = 0.0
    else:
        caps  = sum(1 for w in toks if w.isupper() and len(w) > 1) / len(toks)
        title = sum(1 for w in toks if w.istitle()) / len(toks)
        first = sum(1 for w in toks if w[0].isupper()) / len(toks)
        case_score = max(caps, title * 0.85, first * 0.5)

    center_score = 1.0 - abs(line["center_x"] - img_w / 2) / (img_w / 2 + 1e-6)
    span_score   = 1.0 - min(line["span"] / (img_w + 1e-6), 1.0) * 0.4
    total_alnum  = sum(1 for c in text if c.isalnum())
    alpha_score  = sum(1 for c in text if c.isalpha()) / total_alnum if total_alnum else 0.0
    wc           = len(line["words"])
    wc_score     = 1.0 if wc <= 4 else (0.6 if wc <= 8 else max(0.0, 1.0 - (wc - 8) * 0.07))

    venue_boost      = 0.15 if VENUE_RE.search(text) else 0.0
    noise_penalty    = 0.35 if NOISE_RE.search(text) else 0.0
    address_penalty  = 0.30 if ADDRESS_LINE_RE.search(text) else 0.0
    puredata_penalty = 0.40 if PURE_DATA_RE.search(text) else 0.0

    # Penalise lines where >30% of characters are digits — prices, dates, codes
    digit_ratio      = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    digit_penalty    = 0.25 if digit_ratio > 0.30 else 0.0

    # Penalise lines where words are smashed together (no spaces, long tokens) — OCR garbage
    tokens           = text.split()
    avg_tok_len      = statistics.mean(len(t) for t in tokens) if tokens else 0
    garbage_penalty  = 0.20 if avg_tok_len > 12 else 0.0

    return round(
        0.28 * pos_score + 0.26 * size_score + 0.18 * case_score +
        0.10 * center_score + 0.07 * span_score + 0.05 * line["avg_conf"] +
        0.04 * alpha_score + 0.02 * wc_score
        + venue_boost
        - noise_penalty - address_penalty - puredata_penalty
        - digit_penalty - garbage_penalty, 4
    )


def _extract_title(lines, img_h, img_w) -> dict:
    scored = [
        (s, l) for l in lines
        if len(re.sub(r"[^A-Za-z0-9 ]", "", l["text"]).strip()) >= 3
        for s in [_score(l, lines, img_h, img_w)]
    ]
    if not scored:
        return {"title": "", "candidates": ""}
    scored.sort(key=lambda x: -x[0])
    best_s, best_l = scored[0]
    title = best_l["text"].strip()
    if len(scored) > 1:
        s2, l2 = scored[1]
        if s2 >= best_s * 0.65 and (l2["top"] - best_l["bottom"]) < best_l["avg_height"] * 2.5:
            title = title + " " + l2["text"].strip()
    return {
        "title":      clean_title(title),
        "candidates": " | ".join(l["text"].strip() for _, l in scored[:5]),
    }


# ── PER-IMAGE PROCESSING ──────────────────────────────────────────────────────

def process_image(image_path: str, index: int, total: int) -> dict:
    name  = Path(image_path).name
    start = time.perf_counter()
    print(f"[{index}/{total}] Processing: {name}", flush=True)
    try:
        top_gray, bot_gray = load_and_split(image_path)
        img_h, img_w       = top_gray.shape

        # Submit both OCR jobs to the global thread pool
        f_paddle = _POOL.submit(_run_paddle, top_gray)
        f_tess   = _POOL.submit(_run_tesseract, bot_gray)
        paddle_words = f_paddle.result()
        body_text    = f_tess.result()

        elapsed = round(time.perf_counter() - start, 2)

        if not paddle_words:
            print(f"  -> no text detected  ({elapsed}s)", flush=True)
            return {"file": name, "title": "", "candidates": "no text detected",
                    "body_text": body_text, "time_sec": elapsed}

        lines  = _group_lines(paddle_words, img_w)
        result = _extract_title(lines, img_h, img_w)
        print(f"  -> {result['title']}  ({elapsed}s)", flush=True)
        return {"file": name, **result, "body_text": body_text, "time_sec": elapsed}

    except Exception as e:
        elapsed = round(time.perf_counter() - start, 2)
        print(f"  -> ERROR: {e}  ({elapsed}s)", flush=True)
        return {"file": name, "title": "", "candidates": f"ERROR: {e}",
                "body_text": "", "time_sec": elapsed}


# ── FOLDER RUNNER ─────────────────────────────────────────────────────────────

def collect_images(folder: str):
    return sorted(
        str(p) for p in Path(folder).rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def run(folder: str, output_csv: str):
    images = collect_images(folder)
    if not images:
        print(f"No images found in: {folder}")
        sys.exit(1)

    print(f"Found {len(images)} images in '{folder}'")
    print("-" * 50, flush=True)

    results = [process_image(p, i, len(images)) for i, p in enumerate(images, 1)]
    results.sort(key=lambda r: r["file"])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "title", "candidates", "body_text", "time_sec"])
        writer.writeheader()
        writer.writerows(results)

    _POOL.shutdown(wait=False)

    total_t = sum(r["time_sec"] for r in results)
    print("-" * 50)
    print(f"Done. CSV saved -> {output_csv}  ({len(results)} rows)")
    print(f"Total: {round(total_t, 2)}s  |  Avg: {round(total_t / len(results), 2)}s per image")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.10 bill_extractor.py <folder> [output.csv]")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT)
