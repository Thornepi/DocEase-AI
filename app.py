import os
import uuid
import io
import re
import time
import json
import shutil

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import pymupdf
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

from dotenv import load_dotenv
from google import genai


load_dotenv()


# =========================================================
# GEMINI SETTINGS
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "models/gemini-3.6-flash"
)

GEMINI_MODEL_FALLBACK = os.getenv(
    "GEMINI_MODEL_FALLBACK",
    ""
).strip()


gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)


# =========================================================
# APP SETTINGS
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {
    "pdf",
    "png",
    "jpg",
    "jpeg"
}

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


MAX_PDF_PAGES_ALLOWED = 12

PDF_TEXT_PAGES = 3

PDF_OCR_PAGES = 2


print("✅ Running DocEase AI")
# =========================================================
# BASIC HELPERS
# =========================================================

def allowed_file(filename: str) -> bool:

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def clean_text(text: str) -> str:

    if not text:
        return ""

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        " ",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# TESSERACT / OCR
# =========================================================

def check_tesseract():

    path = shutil.which("tesseract")

    if path:

        print(
            f"✅ Tesseract found: {path}"
        )

        return True

    print(
        "⚠️ Tesseract executable NOT found."
    )

    return False


def preprocess_for_ocr(
    img: Image.Image
) -> Image.Image:

    img = img.convert("L")

    img = ImageOps.autocontrast(img)

    img = ImageEnhance.Contrast(
        img
    ).enhance(1.3)

    img = ImageEnhance.Sharpness(
        img
    ).enhance(1.2)

    return img


def ocr_image(
    img: Image.Image
) -> str:

    if not check_tesseract():

        raise RuntimeError(
            "Tesseract OCR is not installed."
        )

    config = (
        "--oem 3 "
        "--psm 6 "
        "-c preserve_interword_spaces=1"
    )

    text = pytesseract.image_to_string(
        img,
        lang="eng",
        config=config
    )

    return clean_text(text)
# =========================================================
# IMAGE TEXT EXTRACTION
# =========================================================

def extract_text_from_image(
    image_path: str
):

    img = None

    try:

        img = Image.open(
            image_path
        )

        max_dimension = 2500

        if max(
            img.width,
            img.height
        ) > max_dimension:

            scale = (
                max_dimension
                / max(
                    img.width,
                    img.height
                )
            )

            img = img.resize(
                (
                    int(img.width * scale),
                    int(img.height * scale)
                )
            )

        img = preprocess_for_ocr(
            img
        )

        text = ocr_image(
            img
        )

        return text, "OCR (image)"

    finally:

        if img is not None:

            try:
                img.close()
            except:
                pass


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(
    pdf_path: str
):

    doc = None

    try:

        doc = pymupdf.open(
            pdf_path
        )

        parts = []

        # Try normal PDF text first
        for i in range(
            min(
                PDF_TEXT_PAGES,
                doc.page_count
            )
        ):

            page = doc.load_page(i)

            page_text = page.get_text(
                "text"
            )

            if page_text:

                parts.append(
                    page_text
                )

        text = clean_text(
            "\n".join(parts)
        )

        method = "PDF text layer"


        # =================================================
        # OCR FALLBACK
        # =================================================

        if len(text) < 50:

            print(
                "🔎 Little/no text found."
            )

            print(
                "🔎 Switching to OCR..."
            )

            ocr_parts = []

            for i in range(
                min(
                    PDF_OCR_PAGES,
                    doc.page_count
                )
            ):

                print(
                    f"🔎 OCR page {i + 1}"
                )

                page = doc.load_page(i)

                # Lower DPI = less memory/CPU
                pix = page.get_pixmap(
                    dpi=220,
                    alpha=False
                )

                img = Image.open(
                    io.BytesIO(
                        pix.tobytes("png")
                    )
                )

                img = img.resize(
                    (
                        int(img.width * 1.1),
                        int(img.height * 1.1)
                    )
                )

                img = preprocess_for_ocr(
                    img
                )

                ocr_text = ocr_image(
                    img
                )

                ocr_parts.append(
                    ocr_text
                )

                img.close()

            text = clean_text(
                "\n".join(ocr_parts)
            )

            method = "OCR (PDF scan)"

        return text, method

    finally:

        if doc is not None:

            doc.close()
            # =========================================================
# GEMINI
# =========================================================

def gemini_generate_text(
    prompt: str
) -> str:

    if not gemini_client:

        raise RuntimeError(
            "GEMINI_API_KEY not found."
        )

    models_to_try = [
        GEMINI_MODEL
    ]

    if GEMINI_MODEL_FALLBACK:

        models_to_try.append(
            GEMINI_MODEL_FALLBACK
        )

    last_error = None

    for model in models_to_try:

        for attempt in range(3):

            try:

                print(
                    f"🤖 Calling Gemini: {model}"
                )

                response = (
                    gemini_client
                    .models
                    .generate_content(
                        model=model,
                        contents=prompt
                    )
                )

                return response.text or ""

            except Exception as e:

                last_error = e

                print(
                    f"⚠️ Gemini error: {e}"
                )

                message = str(e)

                if (
                    "503" in message
                    or "UNAVAILABLE" in message
                    or "high demand" in message.lower()
                ):

                    time.sleep(
                        1 + attempt * 2
                    )

                    continue

                if (
                    "404" in message
                    or "NOT_FOUND" in message
                    or "no longer available" in message
                ):

                    break

                raise

    raise RuntimeError(
        f"Gemini failed: {last_error}"
    )


# =========================================================
# JSON
# =========================================================

def extract_json_from_text(
    text: str
) -> dict:

    start = text.find("{")

    end = text.rfind("}")

    if (
        start == -1
        or end == -1
        or end <= start
    ):

        raise ValueError(
            "Could not find JSON in Gemini response."
        )

    return json.loads(
        text[start:end + 1]
    )


# =========================================================
# GEMINI DOCUMENT ANALYSIS
# =========================================================

def analyze_document_with_gemini(
    extracted_text: str
) -> dict:

    doc_text = extracted_text[:12000]

    prompt = f"""
Return ONLY valid JSON.

{{
  "document_type": "string or null",
  "important_information": [
    {{
      "label": "string",
      "value": "string or null",
      "sensitive": true
    }}
  ],
  "simple_explanation": "string",
  "things_to_notice": ["string"],
  "is_medical": true,
  "medical_disclaimer": "string or null"
}}

Rules:
- Do not invent details.
- Only use information present in the document.
- If medical, do NOT diagnose.
- Include a disclaimer for medical documents.
- Mark truly sensitive identifiers as sensitive=true.
- Sensitive examples include Aadhaar, bank account numbers,
  PAN, passport numbers, IDs and certificate numbers.

DOCUMENT TEXT:

\"\"\"
{doc_text}
\"\"\"
""".strip()

    raw = gemini_generate_text(
        prompt
    )

    return extract_json_from_text(
        raw
    )


# =========================================================
# SENSITIVE INFORMATION MASKING
# =========================================================

def mask_sensitive_value(
    value: str | None
) -> str:

    if value is None:
        return ""

    s = str(value).strip()

    if not s:
        return ""

    digits = re.sub(
        r"\D",
        "",
        s
    )

    if len(digits) >= 8:

        last4 = digits[-4:]

        masked_digits = (
            "X" * (len(digits) - 4)
            + last4
        )

        grouped = " ".join(
            [
                masked_digits[i:i + 4]
                for i in range(
                    0,
                    len(masked_digits),
                    4
                )
            ]
        )

        return grouped.strip()

    if len(s) > 4:

        return (
            "X" * (len(s) - 2)
            + s[-2:]
        )

    return "XX"


def should_force_sensitive(
    label: str,
    value: str | None
) -> bool:

    lab = (
        label or ""
    ).lower()

    sensitive_words = [
        "aadhaar",
        "aadhar",
        "account",
        "pan",
        "passport",
        "id",
        "number",
        "certificate no",
        "certificate number"
    ]

    if any(
        word in lab
        for word in sensitive_words
    ):

        return True

    if value:

        digits = re.sub(
            r"\D",
            "",
            value
        )

        if len(digits) == 12:
            return True

        if 9 <= len(digits) <= 18:
            return True

    return False


def postprocess_analysis(
    analysis: dict
) -> dict:

    if (
        not analysis
        or not isinstance(
            analysis,
            dict
        )
    ):

        return analysis

    info = analysis.get(
        "important_information",
        []
    )

    if not isinstance(
        info,
        list
    ):

        info = []

    new_info = []

    for item in info:

        if not isinstance(
            item,
            dict
        ):

            continue

        label = item.get(
            "label",
            ""
        )

        value = item.get(
            "value",
            None
        )

        sensitive = bool(
            item.get(
                "sensitive",
                False
            )
        )

        if should_force_sensitive(
            label,
            value
        ):

            sensitive = True

        masked_value = (
            mask_sensitive_value(
                value
            )
            if sensitive
            else (
                value
                if value is not None
                else ""
            )
        )

        new_info.append(
            {
                "label": label,
                "value": (
                    value
                    if value is not None
                    else ""
                ),
                "sensitive": sensitive,
                "masked_value": masked_value
            }
        )

    analysis[
        "important_information"
    ] = new_info

    return analysis
# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# GEMINI TEST
# =========================================================

@app.route("/gemini-test")
def gemini_test():

    try:

        result = gemini_generate_text(
            "Reply with exactly: Gemini is working"
        )

        return result

    except Exception as e:

        print(
            f"❌ Gemini test failed: {e}"
        )

        return (
            f"Gemini test failed: {e}",
            500
        )


# =========================================================
# UPLOAD
# =========================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload():

    # If someone opens /upload directly
    # don't show a 405 error.
    if request.method == "GET":

        return render_template(
            "index.html"
        )


    if "document" not in request.files:

        return render_template(
            "index.html",
            error="No file found. Please select a document."
        )


    file = request.files["document"]


    if file.filename == "":

        return render_template(
            "index.html",
            error="No file selected. Please choose a file."
        )


    if not allowed_file(
        file.filename
    ):

        return render_template(
            "index.html",
            error=(
                "Invalid file type. "
                "Please upload a PDF or image."
            )
        )


    original_name = secure_filename(
        file.filename
    )

    ext = original_name.rsplit(
        ".",
        1
    )[1].lower()


    temp_name = (
        f"{uuid.uuid4().hex}.{ext}"
    )

    temp_path = os.path.join(
        UPLOAD_FOLDER,
        temp_name
    )


    pdf_page_count = None

    analysis = None

    analysis_error = None

    warn_msg = None

    extraction_method = ""

    extracted_text = ""


    try:

        # =================================================
        # SAVE FILE
        # =================================================

        file.save(
            temp_path
        )

        print(
            f"📄 Uploaded: {original_name}"
        )


        # =================================================
        # PDF
        # =================================================

        if ext == "pdf":

            try:

                doc = pymupdf.open(
                    temp_path
                )

                pdf_page_count = (
                    doc.page_count
                )

                doc.close()

            except Exception as e:

                print(
                    f"❌ PDF open error: {e}"
                )

                return render_template(
                    "index.html",
                    error=(
                        "This PDF could not be opened. "
                        "Please try another PDF."
                    )
                )


            if (
                pdf_page_count
                > MAX_PDF_PAGES_ALLOWED
            ):

                return render_template(
                    "index.html",
                    error=(
                        f"PDF has {pdf_page_count} pages. "
                        f"Please upload up to "
                        f"{MAX_PDF_PAGES_ALLOWED} pages."
                    )
                )


            try:

                (
                    extracted_text,
                    extraction_method
                ) = extract_text_from_pdf(
                    temp_path
                )

            except Exception as e:

                print(
                    "❌ PDF/OCR ERROR:"
                )

                print(
                    f"{type(e).__name__}: {e}"
                )

                return render_template(
                    "index.html",
                    error=(
                        "We couldn't read this PDF. "
                        "If it is scanned or blurry, "
                        "please try a clearer copy."
                    )
                )


        # =================================================
        # IMAGE
        # =================================================

        else:

            try:

                (
                    extracted_text,
                    extraction_method
                ) = extract_text_from_image(
                    temp_path
                )

            except Exception as e:

                print(
                    "❌ IMAGE OCR ERROR:"
                )

                print(
                    f"{type(e).__name__}: {e}"
                )

                return render_template(
                    "index.html",
                    error=(
                        "We couldn't read this image. "
                        "Please upload a clearer image."
                    )
                )


        # =================================================
        # CLEAN TEXT
        # =================================================

        extracted_text = (
            extracted_text or ""
        ).strip()


        if extraction_method.startswith(
            "OCR"
        ):

            warn_msg = (
                "Scanned/compressed document detected. "
                "OCR may be slower or less accurate."
            )


        # =================================================
        # GEMINI
        # =================================================

        if extracted_text:

            try:

                print(
                    "🤖 Sending text to Gemini..."
                )

                analysis = (
                    analyze_document_with_gemini(
                        extracted_text
                    )
                )

                analysis = (
                    postprocess_analysis(
                        analysis
                    )
                )

                print(
                    "✅ Gemini analysis completed."
                )

            except Exception as e:

                print(
                    "❌ GEMINI ERROR:"
                )

                print(
                    f"{type(e).__name__}: {e}"
                )

                analysis_error = (
                    "The document was read, "
                    "but the AI summary could not "
                    "be generated right now."
                )

        else:

            warn_msg = (
                "Could not extract readable text. "
                "Try a clearer PDF or image."
            )


        # =================================================
        # RESULT PAGE
        # =================================================

        word_count = len(
            extracted_text.split()
        )

        preview = extracted_text[
            :2500
        ]


        return render_template(
            "result.html",

            filename=original_name,

            filetype=ext.upper(),

            message="Processed successfully.",

            extraction_method=(
                extraction_method
            ),

            word_count=word_count,

            extracted_preview=preview,

            extraction_error=warn_msg,

            analysis=analysis,

            analysis_error=analysis_error,

            pdf_page_count=pdf_page_count
        )


    except Exception as e:

        print(
            "❌ UNEXPECTED ERROR:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        return render_template(
            "index.html",
            error=(
                "Something went wrong while "
                "processing the document. "
                "Please try again."
            )
        )


    finally:

        # Always delete temporary upload
        if os.path.exists(
            temp_path
        ):

            try:

                os.remove(
                    temp_path
                )

            except Exception as e:

                print(
                    f"⚠️ Could not delete temp file: {e}"
                )


# =========================================================
# FILE TOO LARGE
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        error=(
            "File is too large. "
            "Please upload a file smaller than 10 MB."
        )
    ), 413


# =========================================================
# GENERAL SERVER ERROR
# =========================================================

@app.errorhandler(500)
def internal_server_error(error):

    print(
        f"❌ Flask 500 error: {error}"
    )

    return render_template(
        "index.html",
        error=(
            "Something went wrong while "
            "processing the document. "
            "Please try again with a clearer file."
        )
    ), 500


# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )