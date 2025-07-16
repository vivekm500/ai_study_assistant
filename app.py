import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import speech_recognition as sr
import tempfile
import os
from googletrans import Translator
from docx import Document
from reportlab.pdfgen import canvas

# Page config
st.set_page_config(page_title="AI Study Assistant", layout="centered")

# Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
translator = Translator()

# Voice to Text
def convert_audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

# PDF Reader
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# PDF Export
def export_to_pdf(content, filename):
    temp_path = f"{filename}.pdf"
    c = canvas.Canvas(temp_path)
    textobject = c.beginText(40, 800)
    for line in content.split('\n'):
        textobject.textLine(line)
    c.drawText(textobject)
    c.save()
    return temp_path

# DOCX Export
def export_to_docx(content, filename):
    doc = Document()
    doc.add_paragraph(content)
    path = f"{filename}.docx"
    doc.save(path)
    return path

# Title
st.title("📚 AI Study Assistant")

# User Inputs
with st.expander("📤 Upload or Paste Notes"):
    uploaded_pdf = st.file_uploader("📄 Upload PDF Notes", type="pdf")
    audio_file = st.file_uploader("🎤 Upload WAV voice note", type=["wav"])
    text_input = st.text_area("✍️ Or paste your notes here")
    lang = st.selectbox("🌍 Translate output to:", ["None", "hi", "bn", "ta", "te", "gu"], index=0)

# Load Input Text
input_text = ""
if uploaded_pdf:
    input_text = read_pdf(uploaded_pdf)
    st.success("✅ PDF loaded")
elif audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        input_text = convert_audio_to_text(tmp.name)
        os.remove(tmp.name)
        st.success("✅ Voice note transcribed")
elif text_input:
    input_text = text_input

# Generate
if st.button("🤖 Generate All Content"):
    if not input_text.strip():
        st.warning("⚠️ Please provide notes first.")
    else:
        chunk = input_text[:2048]

        # Summary
        summary = summarizer(chunk, max_length=300, min_length=80, do_sample=False)[0]["summary_text"]
        if lang != "None":
            summary = translator.translate(summary, dest=lang).text
        st.subheader("📝 Summary")
        st.write(summary)

        # Questions
        prompt_q = f"Generate all important short and descriptive questions based on:\n{chunk}"
        input_ids = tokenizer(prompt_q, return_tensors="pt", max_length=1024, truncation=True)
        output_ids = model.generate(**input_ids, max_length=1024, num_beams=4, early_stopping=True)
        questions = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if lang != "None":
            questions = translator.translate(questions, dest=lang).text
        st.subheader("❓ Important Questions")
        st.text(questions)

        # MCQs
        prompt_mcq = f"Generate multiple MCQs with 4 options and correct answers from:\n{chunk}"
        input_ids = tokenizer(prompt_mcq, return_tensors="pt", max_length=1024, truncation=True)
        output_ids = model.generate(**input_ids, max_length=1024, num_beams=4, early_stopping=True)
        mcqs = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if lang != "None":
            mcqs = translator.translate(mcqs, dest=lang).text
        st.subheader("🧠 MCQs")
        st.text(mcqs)

        # Flashcards
        prompt_flashcards = f"Generate flashcards (Q&A pairs) for revision from:\n{chunk}"
        input_ids = tokenizer(prompt_flashcards, return_tensors="pt", max_length=1024, truncation=True)
        output_ids = model.generate(**input_ids, max_length=1024, num_beams=4, early_stopping=True)
        flashcards = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if lang != "None":
            flashcards = translator.translate(flashcards, dest=lang).text
        st.subheader("📇 Flashcards")
        st.text(flashcards)

        # All content
        full_content = f"Summary:\n{summary}\n\nQuestions:\n{questions}\n\nMCQs:\n{mcqs}\n\nFlashcards:\n{flashcards}"

        # Download buttons
        st.download_button("📥 Download as TXT", full_content, file_name="study_output.txt")
        pdf_path = export_to_pdf(full_content, "study_output")
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download as PDF", f, file_name="study_output.pdf")
        docx_path = export_to_docx(full_content, "study_output")
        with open(docx_path, "rb") as f:
            st.download_button("📝 Download as DOCX", f, file_name="study_output.docx")
