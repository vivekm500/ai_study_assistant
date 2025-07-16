import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import PyPDF2
import speech_recognition as sr
import tempfile
import os
from docx import Document
from reportlab.pdfgen import canvas
import torch
import textwrap

# Page config
st.set_page_config(page_title="AI Study Assistant", layout="centered")

# Load models
st.spinner("Loading models...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

def generate_text(prompt, max_len=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=max_len, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Translation support
lang_map = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "bn": "Helsinki-NLP/opus-mt-en-bn",
    "ta": "Helsinki-NLP/opus-mt-en-ta",
    "te": "Helsinki-NLP/opus-mt-en-te",
    "gu": "Helsinki-NLP/opus-mt-en-gu"
}

def translate(text, lang):
    if lang == "None" or lang not in lang_map:
        return text
    model_name = lang_map[lang]
    tokenizer_mt = MarianTokenizer.from_pretrained(model_name)
    model_mt = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer_mt(text, return_tensors="pt", truncation=True, padding=True)
    translated = model_mt.generate(**inputs)
    return tokenizer_mt.decode(translated[0], skip_special_tokens=True)

# File reading

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages])

def convert_audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

# Export utilities

def export_to_pdf(content, filename):
    path = f"{filename}.pdf"
    c = canvas.Canvas(path)
    textobject = c.beginText(40, 800)
    for line in content.split('\n'):
        textobject.textLine(line)
    c.drawText(textobject)
    c.save()
    return path

def export_to_docx(content, filename):
    doc = Document()
    for line in content.split('\n'):
        doc.add_paragraph(line)
    path = f"{filename}.docx"
    doc.save(path)
    return path

# UI
st.title("📚 AI Study Assistant - flan-t5-large powered")
uploaded_pdf = st.file_uploader("📄 Upload PDF", type="pdf")
audio_file = st.file_uploader("🎤 Upload Voice Note (WAV)", type="wav")
text_input = st.text_area("✍️ Or Paste Your Notes Here")
lang = st.selectbox("🌍 Translate Output To", ["None", "hi", "bn", "ta", "te", "gu"])

input_text = ""
if uploaded_pdf:
    input_text = read_pdf(uploaded_pdf)
elif audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        input_text = convert_audio_to_text(tmp.name)
        os.remove(tmp.name)
elif text_input:
    input_text = text_input

if st.button("🚀 Generate Study Material") and input_text.strip():
    text_chunks = textwrap.wrap(input_text, 1000)
    full_summary, full_questions, full_mcqs, full_flashcards = [], [], [], []

    for chunk in text_chunks:
        # Summary
        summary = summarizer(chunk, max_length=512, min_length=150, do_sample=False)[0]['summary_text']
        full_summary.append(summary)

        # Important Questions
        q_prompt = f"""
Extract all meaningful descriptive and conceptual questions from this academic text. Avoid duplicates.
Text:
{chunk}
"""
        questions = generate_text(q_prompt)
        full_questions.append(questions)

        # MCQs
        mcq_prompt = f"""
Generate all possible multiple-choice questions from this academic text. Each MCQ must have this format:

Question: ...
Options:
A. ...
B. ...
C. ...
D. ...
Answer: B

Avoid repeated questions or repeated options.
Text:
{chunk}
"""
        mcqs = generate_text(mcq_prompt)
        full_mcqs.append(mcqs)

        # Flashcards
        fc_prompt = f"""
Create all possible non-repetitive flashcards from the text. Format:

Question: ...
Answer: ...

Text:
{chunk}
"""
        flashcards = generate_text(fc_prompt)
        full_flashcards.append(flashcards)

    # Combine + Translate
    combined = {
        "📝 Summary": "\n".join(full_summary),
        "❓ Important Questions": "\n".join(full_questions),
        "🧠 MCQs": "\n".join(full_mcqs),
        "📇 Flashcards": "\n".join(full_flashcards)
    }

    final_output = "\n\n".join([f"{key}\n{translate(value, lang)}" for key, value in combined.items()])

    # Display
    for section, text in combined.items():
        st.subheader(section)
        st.text(translate(text, lang))

    # Export
    st.download_button("📥 Download as TXT", final_output, file_name="study_material.txt")
    with open(export_to_pdf(final_output, "study_material"), "rb") as f:
        st.download_button("📄 Download PDF", f, file_name="study_material.pdf")
    with open(export_to_docx(final_output, "study_material"), "rb") as f:
        st.download_button("📝 Download DOCX", f, file_name="study_material.docx")
