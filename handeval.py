import streamlit as st
import fitz  # PyMuPDF
import easyocr
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load models once
reader = easyocr.Reader(['en'], gpu=False)
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("📄 HandEval.AI – Answer Sheet Evaluation")

# -------------------------------
# Function: Extract text from PDF
# -------------------------------
def extract_text_from_pdf(pdf_file):
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""

        for page in doc:
            text = page.get_text()
            if not text.strip():
                # If PDF text extraction fails, fallback to OCR
                pix = page.get_pixmap()
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                ocr_result = reader.readtext(img, detail=0)
                text = " ".join(ocr_result)

            full_text += text + "\n"

        return full_text
    except Exception as e:
        st.error(f"Error extracting PDF: {e}")
        return ""

# -------------------------------
# Function: Evaluate similarity
# -------------------------------
def evaluate_answers(key_text, student_text):
    emb1 = model.encode(key_text, convert_to_tensor=True)
    emb2 = model.encode(student_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return round(similarity * 100, 2)

# -------------------------------
# UI Section
# -------------------------------
answer_key_pdf = st.file_uploader("📘 Upload Answer Key (PDF)", type=["pdf"])
student_pdf = st.file_uploader("📝 Upload Student Answer Sheet (PDF)", type=["pdf"])

if st.button("⚡ Evaluate"):
    if answer_key_pdf and student_pdf:
        st.write("⏳ Extracting text from PDFs...")
        key_text = extract_text_from_pdf(answer_key_pdf)
        student_text = extract_text_from_pdf(student_pdf)

        if key_text.strip() == "" or student_text.strip() == "":
            st.error("❌ Unable to extract text from one or both PDFs.")
        else:
            score = evaluate_answers(key_text, student_text)
            st.success(f"✅ Evaluation Complete! Similarity Score: **{score}%**")

    else:
        st.warning("⚠ Please upload both PDFs before clicking Evaluate.")
