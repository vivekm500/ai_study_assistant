import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import PyPDF2
import speech_recognition as sr
import tempfile
import os
from docx import Document
from reportlab.pdfgen import canvas

# Streamlit config
st.set_page_config(page_title="AI Study Assistant", layout="centered")

# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Language translation map
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

# Read PDF
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Voice to text
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

# Generate output with model
def generate_with_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=1024, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI Header
st.title("📚 AI Study Assistant (All-In-One Generator)")

# Input methods
with st.expander("📤 Upload or Paste Notes"):
    uploaded_pdf = st.file_uploader("📄 Upload PDF Notes", type="pdf")
    audio_file = st.file_uploader("🎤 Upload WAV voice note", type=["wav"])
    text_input = st.text_area("✍️ Or paste your notes here")
    lang = st.selectbox("🌍 Translate output to:", ["None", "hi", "bn", "ta", "te", "gu"], index=0)

# Get input
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

# Process button
if st.button("🚀 Generate Summary, Questions, MCQs, Flashcards"):
    if not input_text.strip():
        st.warning("⚠️ Please provide some notes.")
    else:
        chunk = input_text[:2048]

        # Summary
        summary = summarizer(chunk, max_length=512, min_length=150, do_sample=False)[0]["summary_text"]
        summary = translate(summary, lang)
        st.subheader("📝 Summary")
        st.write(summary)

        # Important Questions
        question_prompt = f"""
From the following text, generate 5 important short questions for revision:

Text:
{chunk}
"""
        questions = generate_with_model(question_prompt)
        questions = translate(questions, lang)
        st.subheader("❓ Important Questions")
        st.text(questions)

        # MCQs
        mcq_prompt = f"""
From the text below, create 3 multiple-choice questions with 4 options (A to D) and mention the correct answer.

Format:
Question:
Options:
A.
B.
C.
D.
Answer:

Text:
{chunk}
"""
        mcqs = generate_with_model(mcq_prompt)
        mcqs = translate(mcqs, lang)
        st.subheader("🧠 MCQs")
        st.text(mcqs)

        # Flashcards
        flashcard_prompt = f"""
Generate 5 flashcards in Q&A format from this text:

Text:
{chunk}
"""
        flashcards = generate_with_model(flashcard_prompt)
        flashcards = translate(flashcards, lang)
        st.subheader("📇 Flashcards")
        st.text(flashcards)

        # Export all
        full_output = f"""📝 Summary:\n{summary}\n\n❓ Important Questions:\n{questions}\n\n🧠 MCQs:\n{mcqs}\n\n📇 Flashcards:\n{flashcards}"""

        st.download_button("📥 Export as TXT", full_output, file_name="study_output.txt")

        pdf_path = export_to_pdf(full_output, "study_output")
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Export as PDF", f, file_name="study_output.pdf")

        docx_path = export_to_docx(full_output, "study_output")
        with open(docx_path, "rb") as f:
            st.download_button("📝 Export as DOCX", f, file_name="study_output.docx")
