from __future__ import annotations

import re
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

LABELS = {
    "current": ["現在値", "現在", "株価", "取引値", "现价", "当前价", "current", "last"],
    "open": ["始値", "开盘", "open"],
    "high": ["高値", "最高", "high"],
    "low": ["安値", "最低", "low"],
    "volume": ["出来高", "成交量", "volume"],
}


def preprocess_image(img: Image.Image) -> Image.Image:
    im = img.convert("L")
    im = ImageOps.autocontrast(im)
    im = ImageEnhance.Contrast(im).enhance(1.5)
    im = im.filter(ImageFilter.SHARPEN)
    if im.width < 1400:
        scale = 1400 / im.width
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    return im


def run_ocr(img: Image.Image) -> str:
    try:
        import pytesseract
        pre = preprocess_image(img)
        return pytesseract.image_to_string(pre, lang="jpn+eng", config="--psm 6")
    except Exception as e:
        return f"[OCR unavailable: {e}]"


def _num(s: str) -> float | None:
    s = s.replace(",", "").replace("円", "").strip()
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def extract_fields(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for line in lines:
        lower = line.lower()
        for key, labels in LABELS.items():
            if key in result:
                continue
            if any(lbl.lower() in lower for lbl in labels):
                value = _num(line)
                if value is not None:
                    result[key] = value
    # Common stock screenshots place a ticker code near the top.
    m = re.search(r"\b([0-9]{4}|[0-9]{3}[A-Z])\b", text.upper())
    if m:
        result["code"] = m.group(1)
    return result
