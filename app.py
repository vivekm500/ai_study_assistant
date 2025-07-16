import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import PyPDF2
import speech_recognition as sr
import tempfile
import os
from docx import Document
from reportlab.pdfgen import canvas

# App Config
st.set_page_config(page_title="AI Study Assistant", layout="centered")

# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Language map for translation (optional)
lang_map = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "bn": "Helsinki-NLP/opus-mt-en-bn",
    "ta": "Helsinki-NLP/opus-mt-en-ta",
    "te": "Helsinki-NLP/opus-mt-en-te",
    "gu": "Helsinki-NLP/opus-mt-en-gu"
}

def translate(text, target_lang):
    if target_lang == "None" or target_lang not in lang_map:
        return text
    model_name = lang_map[target_lang]
    tokenizer_mt = MarianTokenizer.from_pretrained(model_name)
    model_mt = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer_mt(text, return_tensors="pt", truncation=True, padding=True)
    translated = model_mt.generate(**inputs)
    return tokenizer_mt.decode(translated[0], skip_special_tokens=True)

# PDF reader
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Voice-to-text
def convert_audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

# Export to PDF
def export_to_pdf(content, filename):
    path = f"{filename}.pdf"
    c = canvas.Canvas(path)
    textobject = c.beginText(40, 800)
    for line in content.split('\n'):
        textobject.textLine(line)
    c.drawText(textobject)
    c.save()
    return path

# Export to DOCX
def export_to_docx(content, filename):
    doc = Document()
    doc.add_paragraph(content)
    path = f"{filename}.docx"
    doc.save(path)
    return path

# Title
st.title("📚 AI Study Assistant (Offline + Multilingual)")

# Inputs
with st.expander("📤 Upload or Paste Notes"):
    uploaded_pdf = st.file_uploader("📄 Upload PDF Notes", type="pdf")
    audio_file = st.file_uploader("🎤 Upload WAV voice note", type=["wav"])
    text_input = st.text_area("✍️ Or paste your notes here")
    lang = st.selectbox("🌍 Translate output to:", ["None", "hi", "bn", "ta", "te", "gu"], index=0)

# Extract Input
input_text = ""
if uploaded_pdf:
    input_text = read_pdf(uploaded_pdf)
    st.success("✅ PDF uploaded.")
elif audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        input_text = convert_audio_to_text(tmp.name)
        os.remove(tmp.name)
        st.success("✅ Audio transcribed.")
elif text_input:
    input_text = text_input

# Process Button
if st.button("🤖 Generate Study Materials"):
    if not input_text.strip():
        st.warning("⚠️ Please provide notes first.")
    else:
        chunk = input_text[:2048]

        # Summary
        summary = summarizer(chunk, max_length=512, min_length=150, do_sample=False)[0]["summary_text"]
        summary = translate(summary, lang)
        st.subheader("📝 Summary")
        st.write(summary)

        # Questions
        q_prompt = f"""
Extract all meaningful and non-repetitive descriptive and conceptual questions from the following academic notes.

Notes:
{chunk}
"""
        inputs_q = tokenizer(q_prompt, return_tensors="pt", truncation=True)
        output_q = model.generate(**inputs_q, max_length=1024, num_beams=4)
        questions = tokenizer.decode(output_q[0], skip_special_tokens=True)
        questions = translate(questions, lang)
        st.subheader("❓ Important Questions")
        st.text(questions)

        # MCQs
        mcq_prompt = f"""
Generate as many unique multiple-choice questions (MCQs) as possible from the given notes.
Each MCQ should follow this format:
Question: ...
Options:
A. ...
B. ...
C. ...
D. ...
Answer: <A/B/C/D>

Avoid repeating content.

Text:
{chunk}
"""
        inputs_m = tokenizer(mcq_prompt, return_tensors="pt", truncation=True)
        output_m = model.generate(**inputs_m, max_length=1024, num_beams=4)
        mcqs = tokenizer.decode(output_m[0], skip_special_tokens=True)
        mcqs = translate(mcqs, lang)
        st.subheader("🧠 Multiple Choice Questions (MCQs)")
        st.text(mcqs)

        # Flashcards
        fc_prompt = f"""
Create multiple non-repetitive flashcards from the following academic text.
Each flashcard should be in this format:
Question: ...
Answer: ...

Text:
{chunk}
"""
        inputs_f = tokenizer(fc_prompt, return_tensors="pt", truncation=True)
        output_f = model.generate(**inputs_f, max_length=1024, num_beams=4)
        flashcards = tokenizer.decode(output_f[0], skip_special_tokens=True)
        flashcards = translate(flashcards, lang)
        st.subheader("📇 Flashcards")
        st.text(flashcards)

        # Combine for export
        full_content = f"""📝 Summary:\n{summary}\n\n❓ Questions:\n{questions}\n\n🧠 MCQs:\n{mcqs}\n\n📇 Flashcards:\n{flashcards}"""

        st.download_button("📥 Export as TXT", full_content, file_name="study_output.txt")

        pdf_path = export_to_pdf(full_content, "study_output")
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Export as PDF", f, file_name="study_output.pdf")

        docx_path = export_to_docx(full_content, "study_output")
        with open(docx_path, "rb") as f:
            st.download_button("📝 Export as DOCX", f, file_name="study_output.docx")
