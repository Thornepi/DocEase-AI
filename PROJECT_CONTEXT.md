# DocEase AI — PROJECT CONTEXT

## Purpose
Web app that helps users understand complex documents in simple language.
Uploads are processed temporarily and deleted after processing.

## Tech stack
- Backend: Python + Flask
- Frontend: HTML/CSS/JS
- PDF processing: PyMuPDF (pymupdf)
- OCR: Tesseract OCR via pytesseract + Pillow
- AI: Gemini API (google-genai) [next step]
- Secrets: .env (python-dotenv)

## Current progress
- Flask site working
- Modern responsive UI
- Upload works (PDF/image)
- Extract text from PDFs (text layer) or OCR for scanned PDFs
- Shows warning for low-quality scans
- Deletes uploaded files after processing (uploads folder stays empty)

## Important decisions
- Files are temporary; deleted after processing (no database storage for MVP)
- Show a note asking users to upload clear/uncompressed docs for best results

## Next task
- Gemini integration:
  - doc type detection
  - important info extraction
  - simple explanation
  - “things to notice”
  - later: masking + Q&A

## Files
- app.py
- templates/index.html
- templates/result.html
- static/style.css
- static/script.js
- uploads/ (temp)