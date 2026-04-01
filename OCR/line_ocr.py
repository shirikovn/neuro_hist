import json
import cv2
import numpy as np
import pytesseract
import ollama

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
import os


tessdata_dir = "tessdata_best"
MODEL = "qwen2.5:7b"

IMAGE_PATH = "input/page.jpg"
BBOX_JSON  = "input/boxes.json"
OUTPUT_PDF = "output/page_searchable.pdf"
OUTPUT_TXT = "output/lines.txt"

FONT_PATH  = "DejaVuSans.ttf"
FONT_NAME  = "DejaVuSans"


def llm_correct_with_context(prev_text, curr_text, next_text):
    prompt = f"""
Ты исправляешь ошибки OCR в дореволюционных русских текстах.

Тебе даётся строка и её контекст.

Важно:
- исправь ТОЛЬКО текущую строку
- не возвращай ничего кроме исправленной текущей строки
- не добавляй пояснений
- не склеивай строки

Правила:
- исправляй только ошибки OCR
- сохраняй дореволюционную орфографию

Контекст:

Предыдущая строка:
{prev_text}

Текущая строка:
{curr_text}

Следующая строка:
{next_text}

Исправленная текущая строка:
"""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()


def register_font():
    if os.path.exists(FONT_PATH):
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
        return FONT_NAME
    return "Helvetica"

def split_lines_projection(block_img):
    gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # чуть почистим шум
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    proj = np.sum(th, axis=1)

    lines = []
    start = None

    for i, val in enumerate(proj):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            if i - start > 5:
                lines.append((start, i))
            start = None

    if start is not None:
        lines.append((start, len(proj)))

    return lines


image = cv2.imread(IMAGE_PATH)
h_img, w_img = image.shape[:2]

with open(BBOX_JSON, "r") as f:
    boxes = json.load(f)

config = f'--tessdata-dir "{tessdata_dir}" --oem 1 --psm 7'

lines_data = []

print("line detection\n")

for obj in boxes:
    if obj["score"] < 0.6:
        continue

    x, y, w, h = map(int, obj["bbox"])

    block = image[y:y+h, x:x+w]

    line_ranges = split_lines_projection(block)

    for (y1, y2) in line_ranges:
        line_img = block[y1:y2, :]

        scale = 2
        line_scaled = cv2.resize(line_img, None, fx=scale, fy=scale)

        text = pytesseract.image_to_string(
            line_scaled,
            lang="rus",
            config=config
        ).strip()

        if not text:
            continue

        lines_data.append({
            "text": text,
            "x": x,
            "y": y + y1,
            "w": w,
            "h": (y2 - y1)
        })

print(f"Detected lines: {len(lines_data)}")
lines_data.sort(key=lambda l: (l["y"], l["x"]))

print("LLM correction with context\n")

for i, line in enumerate(lines_data):
    prev_text = lines_data[i-1]["text"] if i > 0 else ""
    curr_text = line["text"]
    next_text = lines_data[i+1]["text"] if i < len(lines_data)-1 else ""

    corrected = llm_correct_with_context(prev_text, curr_text, next_text)

    line["corrected"] = corrected.split("\n")[0]

    print(f"{i+1}/{len(lines_data)}")
    print("prev:", prev_text)
    print("orig:", curr_text)
    print("next:", next_text)
    print("corr:", line["corrected"])
    print("-" * 40)


with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for i, line in enumerate(lines_data):
        orig = line["text"]
        corr = line.get("corrected", "")
        f.write(f"{corr}\n")

print("txt saved:", OUTPUT_TXT)

font_name = register_font()

orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(orig_rgb)

tmp_img = "_tmp.png"
pil_image.save(tmp_img)

pdf_w = float(w_img)
pdf_h = float(h_img)

c = canvas.Canvas(OUTPUT_PDF, pagesize=(pdf_w, pdf_h))

# background
c.drawImage(tmp_img, 0, 0, width=pdf_w, height=pdf_h)

# invisible corrected text
for line in lines_data:
    text = line.get("corrected", "").strip()
    if not text:
        continue

    x = line["x"]
    y = pdf_h - line["y"] - line["h"]
    w = line["w"]
    h = line["h"]

    font_size = max(h, 1)

    c.setFont(font_name, font_size)
    text_width = c.stringWidth(text, font_name, font_size)

    if text_width <= 0:
        continue

    scale_x = (w / text_width) * 100

    c.saveState()
    c.setFillAlpha(0)
    c.setStrokeAlpha(0)

    c.translate(x, y)

    t = c.beginText()
    t.setFont(font_name, font_size)
    t.setTextRenderMode(3)
    t.setHorizScale(scale_x)
    t.setTextOrigin(0, 0)
    t.textLine(text)

    c.drawText(t)
    c.restoreState()

c.save()

os.remove(tmp_img)

print("done:", OUTPUT_PDF)