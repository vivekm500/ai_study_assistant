import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import speech_recognition as sr
import tempfile
import os
from transformers import MarianMTModel, MarianTokenizer
from docx import Document
from reportlab.pdfgen import canvas

# Set layout
st.set_page_config(page_title="AI Study Assistant", layout="centered")

# Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Translation: Hugging Face model mapping
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

# PDF Reader
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
    temp_path = f"{filename}.pdf"
    c = canvas.Canvas(temp_path)
    textobject = c.beginText(40, 800)
    for line in content.split('\n'):
        textobject.textLine(line)
    c.drawText(textobject)
    c.save()
    return temp_path

# Export to DOCX
def export_to_docx(content, filename):
    doc = Document()
    doc.add_paragraph(content)
    path = f"{filename}.docx"
    doc.save(path)
    return path

# UI Start
st.title("📚 AI Study Assistant (Offline + Multilingual)")

# Input Options
with st.expander("📤 Upload or Paste Notes"):
    uploaded_pdf = st.file_uploader("📄 Upload PDF Notes", type="pdf")
    audio_file = st.file_uploader("🎤 Upload WAV voice note", type=["wav"])
    text_input = st.text_area("✍️ Or paste your notes here")
    lang = st.selectbox("🌍 Translate output to:", ["None", "hi", "bn", "ta", "te", "gu"], index=0)

# Collect Input Text
input_text = ""
if uploaded_pdf:
    input_text = read_pdf(uploaded_pdf)
    st.success("✅ PDF uploaded and processed.")
elif audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        input_text = convert_audio_to_text(tmp.name)
        os.remove(tmp.name)
        st.success("✅ Voice note converted to text.")
elif text_input:
    input_text = text_input

# Main Action
if st.button("🤖 Generate Summary, Questions, MCQs, Flashcards"):
    if not input_text.strip():
        st.warning("⚠️ Please provide notes to proceed.")
    else:
        chunk = input_text[:2048]

        # Summary
        summary = summarizer(chunk, max_length=300, min_length=80, do_sample=False)[0]["summary_text"]
        summary = translate(summary, lang)
        st.subheader("📝 Summary")
        st.write(summary)

        # Questions
        q_prompt = f"Generate all important short and descriptive questions based on:\n{chunk}"
        inputs_q = tokenizer(q_prompt, return_tensors="pt", max_length=1024, truncation=True)
        output_q = model.generate(**inputs_q, max_length=1024, num_beams=4, early_stopping=True)
        questions = tokenizer.decode(output_q[0], skip_special_tokens=True)
        questions = translate(questions, lang)
        st.subheader("❓ Important Questions")
        st.text(questions)

        # MCQs
        mcq_prompt = f"Generate multiple MCQs with 4 options and correct answers from:\n{chunk}"
        inputs_m = tokenizer(mcq_prompt, return_tensors="pt", max_length=1024, truncation=True)
        output_m = model.generate(**inputs_m, max_length=1024, num_beams=4, early_stopping=True)
        mcqs = tokenizer.decode(output_m[0], skip_special_tokens=True)
        mcqs = translate(mcqs, lang)
        st.subheader("🧠 MCQs")
        st.text(mcqs)

        # Flashcards
        fc_prompt = f"Generate flashcards (Q&A pairs) for revision from:\n{chunk}"
        inputs_f = tokenizer(fc_prompt, return_tensors="pt", max_length=1024, truncation=True)
        output_f = model.generate(**inputs_f, max_length=1024, num_beams=4, early_stopping=True)
        flashcards = tokenizer.decode(output_f[0], skip_special_tokens=True)
        flashcards = translate(flashcards, lang)
        st.subheader("📇 Flashcards")
        st.text(flashcards)

        # Combine for export
        output_data = f"📝 Summary:\n{summary}\n\n❓ Questions:\n{questions}\n\n🧠 MCQs:\n{mcqs}\n\n📇 Flashcards:\n{flashcards}"

        # Download buttons
        st.download_button("📥 Export as TXT", output_data, file_name="study_output.txt")
        pdf_path = export_to_pdf(output_data, "study_output")
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Export as PDF", f, file_name="study_output.pdf")
        docx_path = export_to_docx(output_data, "study_output")
        with open(docx_path, "rb") as f:
            st.download_button("📝 Export as DOCX", f, file_name="study_output.docx")
