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
DEFAULT_OUTPUT = "bill_titles.csv"
TITLE_CROP = 0.38
BODY_START = 0.33
TARGET_TOP_PX = 800
TARGET_BOT_PX = 900
SKEW_THRESHOLD = 1.5

paddle_ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False,
                       det_db_score_mode="fast", det_db_thresh=0.3,
                       det_db_box_thresh=0.5, rec_batch_num=6)

_POOL = ThreadPoolExecutor(max_workers=2)

def _find_bill_roi(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
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
    y1 = max(0, int(rows[0]) - 5)
    y2 = min(h, int(rows[-1]) + 5)
    x1 = max(0, int(cols[0]) - 5)
    x2 = min(w, int(cols[-1]) + 5)
    if x1 > w * 0.05 or x2 < w * 0.95 or y1 > h * 0.05 or y2 < h * 0.95:
        return gray[y1:y2, x1:x2]
    return gray

def _preprocess(gray: np.ndarray) -> np.ndarray:
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.filter2D(gray, -1,
                        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))

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
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def _resize_to(gray: np.ndarray, max_px: int) -> np.ndarray:
    h, w = gray.shape
    if max(h, w) <= max_px:
        return gray
    scale = max_px / max(h, w)
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def load_and_split(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _find_bill_roi(gray)

    small = _resize_to(gray, 600)
    angle = _detect_skew_angle(small)
    if abs(angle) >= SKEW_THRESHOLD:
        gray = _rotate(gray, angle)

    h, w = gray.shape
    split_top = int(h * TITLE_CROP)
    split_bot = int(h * BODY_START)

    top_gray = _resize_to(gray[:split_top, :], TARGET_TOP_PX)
    top_gray = _preprocess(top_gray)

    bot_gray = _resize_to(gray[split_bot:, :], TARGET_BOT_PX)
    bot_gray = _preprocess(bot_gray)

    return top_gray, bot_gray

def _run_paddle(gray: np.ndarray):
    result = paddle_ocr.ocr(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), cls=False)
    words = []
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
            "text": text, "conf": float(conf),
            "left": min(xs), "top": min(ys),
            "right": max(xs), "bottom": max(ys),
            "height": max(ys) - min(ys),
        })
    return words

def _run_tesseract(gray: np.ndarray) -> str:
    data = pytesseract.image_to_data(
        Image.fromarray(gray),
        output_type=pytesseract.Output.DICT,
        config="--psm 11 --oem 3"
    )
    lines_map = {}
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text or int(data["conf"][i]) < 30:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines_map.setdefault(key, []).append(text)
    return "\n".join(" ".join(ws) for ws in lines_map.values())

def _group_lines(words: list, img_w: int):
    if not words:
        return []
    words = sorted(words, key=lambda w: w["top"])
    buckets = [[words[0]]]
    for w in words[1:]:
        last = buckets[-1][-1]
        cy_last = (last["top"] + last["bottom"]) / 2
        cy_curr = (w["top"] + w["bottom"]) / 2
        tol = max(last["height"], w["height"]) * 0.6
        if abs(cy_curr - cy_last) <= tol:
            buckets[-1].append(w)
        else:
            buckets.append([w])
    lines = []
    for bucket in buckets:
        bucket = sorted(bucket, key=lambda w: w["left"])
        l, r = min(w["left"] for w in bucket), max(w["right"] for w in bucket)
        lines.append({
            "words": bucket,
            "top": min(w["top"] for w in bucket),
            "bottom": max(w["bottom"] for w in bucket),
            "avg_height": statistics.mean(w["height"] for w in bucket),
            "text": " ".join(w["text"] for w in bucket),
            "center_x": (l + r) / 2,
            "span": r - l,
            "img_w": img_w,
            "avg_conf": statistics.mean(w["conf"] for w in bucket),
        })
    return sorted(lines, key=lambda ln: ln["top"])

NOISE_RE = re.compile(
    r"^(paid|reprint|receipt|invoice|copy|duplicate|void|draft|thank\s*you|subtotal|sub\s*total|total|balance\s*due|amount\s*due|cash|change|tax|tip|gratuity|discount|refund|welcome|please\s*come\s*again|call\s*again|thanks|x{4,}|\*{4,}|-{4,}|={4,}|your\s*guest\s*number|see\s*back|survey|chance\s*to\s*win|pickup|take.?out|carry.?out|walk.?in|dine.?in|open\s*daily|hours|we\s*accept|visa|mastercard|amex|www\.|http|customer\s*copy|minimum|credit|debit|approved|authorization|april|january|february|march|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE
)

ADDRESS_LINE_RE = re.compile(
    r"^\d+\s+\w+.*(blvd|ave|st|rd|dr|hwy|pkwy|way|lane|ln|place|pl|court|ct|street|road|drive|highway|parkway)\b",
    re.IGNORECASE
)

PURE_DATA_RE = re.compile(r"^[\d\s\(\)\-\.\,\/\:\*\#]+$")

STATE_SUFFIX_RE = re.compile(
    r",?\s*(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)\.?\s*\d{0,5}$",
    re.IGNORECASE
)

CITYSTATE_RE = re.compile(
    r"[a-z](AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)\.?\s*\d{0,5}$"
)

VENUE_RE = re.compile(
    r"\b(restaurant|cafe|café|coffee|diner|bistro|brasserie|tavern|bar|grill|grille|kitchen|eatery|buffet|pizzeria|pizza|sushi|bakery|steakhouse|smokehouse|bbq|noodle|ramen|pho|thai|chinese|indian|mexican|italian|japanese|korean|vietnamese|mediterranean|fusion|cantina|taqueria|pub|lounge|market|deli|wings|burgers|burger|chicken|seafood|steak|wok|bowl|brew|brewery|provisions|trading|company|express|inc|llc|ltd)\b",
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
    re.compile(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b"),
    re.compile(r"\b(store\s*#|store\s*no|unit\s*#|ks\s*#)\s*\d+", re.IGNORECASE),
    re.compile(r"#\s*\d{2,}"),
    re.compile(r"\b\d{2,}[-/]\d{2,}[-/]\d{2,}\b"),
]

def clean_title(text: str) -> str:
    earliest = len(text)
    for pat in _SPLIT_RE:
        m = pat.search(text)
        if m and m.start() < earliest:
            after = text[m.start(): m.start() + 50]
            venue_in_after = VENUE_RE.search(after)
            earliest = m.start() + venue_in_after.end() if venue_in_after else m.start()
    text = text[:earliest].strip()
    text = STATE_SUFFIX_RE.sub("", text).strip()
    m = CITYSTATE_RE.search(text)
    if m:
        text = text[:m.start() + 1].strip()
    text = re.sub(r"\s+\d+\.\d+", "", text)
    text = re.sub(r"\s+\d{3,}", "", text)
    text = re.sub(r"(?<![A-Za-z])\d+(?![A-Za-z])\s", " ", text)
    text = re.sub(r"[&,\-/\\|#@\d\.]+$", "", text).strip()
    text = re.sub(r"^[#\d\s\-\*\.]+", "", text).strip()
    text = re.sub(r"[^\w\s'&\-\.]", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text
