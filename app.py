import os
import uuid
import io
import re
import time
import json

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import pymupdf
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-3.6-flash")
GEMINI_MODEL_FALLBACK = os.getenv("GEMINI_MODEL_FALLBACK", "").strip()

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# MVP limits (keeps app fast + safe)
MAX_PDF_PAGES_ALLOWED = 12
PDF_TEXT_PAGES = 3
PDF_OCR_PAGES = 2

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("✅ Running app.py (masking + Gemini + OCR enabled)")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img


def ocr_image(img: Image.Image) -> str:
    config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(img, lang="eng", config=config)
    return clean_text(text)


def extract_text_from_image(image_path: str) -> tuple[str, str]:
    img = Image.open(image_path)
    img = img.resize((int(img.width * 2), int(img.height * 2)))
    img = preprocess_for_ocr(img)
    return ocr_image(img), "OCR (image)"


def extract_text_from_pdf(pdf_path: str) -> tuple[str, str]:
    doc = pymupdf.open(pdf_path)

    parts = []
    for i in range(min(PDF_TEXT_PAGES, doc.page_count)):
        page = doc.load_page(i)
        parts.append(page.get_text("text"))

    text = clean_text("\n".join(parts))
    method = "PDF text layer"

    if len(text) < 50:
        ocr_parts = []
        for i in range(min(PDF_OCR_PAGES, doc.page_count)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=350)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            img = img.resize((int(img.width * 1.3), int(img.height * 1.3)))
            img = preprocess_for_ocr(img)

            ocr_parts.append(ocr_image(img))

        text = clean_text("\n".join(ocr_parts))
        method = "OCR (PDF scan)"

    doc.close()
    return text, method


def gemini_generate_text(prompt: str) -> str:
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY not found in .env")

    models_to_try = [GEMINI_MODEL]
    if GEMINI_MODEL_FALLBACK:
        models_to_try.append(GEMINI_MODEL_FALLBACK)

    last_error = None

    for model in models_to_try:
        for attempt in range(3):
            try:
                resp = gemini_client.models.generate_content(model=model, contents=prompt)
                return resp.text or ""
            except Exception as e:
                last_error = e
                msg = str(e)

                if ("503" in msg) or ("UNAVAILABLE" in msg) or ("high demand" in msg):
                    time.sleep(1 + attempt * 2)
                    continue

                if ("404" in msg) or ("NOT_FOUND" in msg) or ("no longer available" in msg):
                    break

                raise

    raise RuntimeError(f"Gemini failed (busy/model issue). Last error: {last_error}")


def extract_json_from_text(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON in Gemini response")
    return json.loads(text[start:end + 1])


def analyze_document_with_gemini(extracted_text: str) -> dict:
    doc_text = extracted_text[:12000]
    prompt = f"""
Return ONLY valid JSON.

{{
  "document_type": "string or null",
  "important_information": [
    {{"label":"string","value":"string or null","sensitive": true/false}}
  ],
  "simple_explanation": "string",
  "things_to_notice": ["string"],
  "is_medical": true/false,
  "medical_disclaimer": "string or null"
}}

Rules:
- Don't invent details not in text.
- If medical: do NOT diagnose; include disclaimer.
- Mark truly sensitive identifiers as sensitive=true (Aadhaar, bank account, IDs, etc.)

DOCUMENT TEXT:
\"\"\"{doc_text}\"\"\"
""".strip()

    raw = gemini_generate_text(prompt)
    return extract_json_from_text(raw)


def mask_sensitive_value(value: str | None) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""

    digits = re.sub(r"\D", "", s)

    # If it looks like a long number (IDs, account numbers), mask all but last 4
    if len(digits) >= 8:
        last4 = digits[-4:]
        masked_digits = "X" * (len(digits) - 4) + last4

        # group into 4s: XXXX XXXX 1234
        grouped = " ".join([masked_digits[i:i+4] for i in range(0, len(masked_digits), 4)])
        return grouped.strip()

    # fallback masking for short-ish identifiers
    if len(s) > 4:
        return "X" * (len(s) - 2) + s[-2:]

    return "XX"


def should_force_sensitive(label: str, value: str | None) -> bool:
    lab = (label or "").lower()
    if any(k in lab for k in ["aadhaar", "aadhar", "account", "pan", "passport", "id", "number", "certificate no", "certificate number"]):
        return True

    if value:
        # Aadhaar-like pattern in text: 12 digits
        digits = re.sub(r"\D", "", value)
        if len(digits) == 12:
            return True
        # Bank/account-like: 9-18 digits
        if 9 <= len(digits) <= 18:
            return True

    return False


def postprocess_analysis(analysis: dict) -> dict:
    if not analysis or not isinstance(analysis, dict):
        return analysis

    info = analysis.get("important_information", [])
    if not isinstance(info, list):
        info = []

    new_info = []
    for item in info:
        if not isinstance(item, dict):
            continue

        label = item.get("label", "")
        value = item.get("value", None)
        sensitive = bool(item.get("sensitive", False))

        # Force sensitive when label/value strongly suggests it
        if should_force_sensitive(label, value):
            sensitive = True

        masked_value = mask_sensitive_value(value) if sensitive else (value if value is not None else "")

        new_info.append({
            "label": label,
            "value": value if value is not None else "",
            "sensitive": sensitive,
            "masked_value": masked_value
        })

    analysis["important_information"] = new_info
    return analysis


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/gemini-test")
def gemini_test():
    try:
        return gemini_generate_text("Reply with exactly: Gemini is working")
    except Exception as e:
        return f"Gemini test failed: {e}", 500


@app.route("/upload", methods=["POST"])
def upload():
    if "document" not in request.files:
        return render_template("index.html", error="No file part found. Please try again.")

    file = request.files["document"]
    if file.filename == "":
        return render_template("index.html", error="No file selected. Please choose a file.")

    if not allowed_file(file.filename):
        return render_template("index.html", error="Invalid file type. Please upload a PDF or image.")

    original_name = secure_filename(file.filename)
    ext = original_name.rsplit(".", 1)[1].lower()

    temp_name = f"{uuid.uuid4().hex}.{ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_name)

    pdf_page_count = None
    analysis = None
    analysis_error = None
    warn_msg = None
    extraction_method = ""
    extracted_text = ""

    try:
        file.save(temp_path)

        # PDF page limit
        if ext == "pdf":
            doc = pymupdf.open(temp_path)
            pdf_page_count = doc.page_count
            doc.close()

            if pdf_page_count > MAX_PDF_PAGES_ALLOWED:
                return render_template(
                    "index.html",
                    error=f"PDF has {pdf_page_count} pages. For MVP, please upload up to {MAX_PDF_PAGES_ALLOWED} pages (or upload the most important pages)."
                )

            extracted_text, extraction_method = extract_text_from_pdf(temp_path)
        else:
            extracted_text, extraction_method = extract_text_from_image(temp_path)

        extracted_text = (extracted_text or "").strip()

        if extraction_method.startswith("OCR"):
            warn_msg = "Scanned/compressed doc detected. OCR may be inaccurate and slower. Upload a clearer scan for better results."

        if extracted_text:
            try:
                analysis = analyze_document_with_gemini(extracted_text)
                analysis = postprocess_analysis(analysis)
            except Exception as e:
                analysis_error = str(e)
        else:
            warn_msg = "Could not extract readable text. Try a clearer scan or a higher quality PDF."

        word_count = len(extracted_text.split())
        preview = extracted_text[:2500]

        return render_template(
            "result.html",
            filename=original_name,
            filetype=ext.upper(),
            message="Processed successfully.",
            extraction_method=extraction_method,
            word_count=word_count,
            extracted_preview=preview,
            extraction_error=warn_msg,
            analysis=analysis,
            analysis_error=analysis_error,
            pdf_page_count=pdf_page_count
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True)