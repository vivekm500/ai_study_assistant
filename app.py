import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile
from docx import Document
from fpdf import FPDF
import torch
from pydub import AudioSegment
import speech_recognition as sr

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load flan-t5-base
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Title
st.title("📚 AI Study Assistant - flan-t5-base (Fast)")

# Input
text = st.text_area("Enter text to analyze")

# Audio file upload
audio_file = st.file_uploader("🎤 Upload a WAV audio note", type=["wav"])
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio_file.read())
        tmp_audio.flush()
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_audio.name) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success("✅ Transcribed audio successfully")
            st.text_area("Transcribed Text", text)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Generate if text provided
if text:
    if st.button("🔍 Analyze"):
        with st.spinner("Generating summary and questions..."):
            def generate(prompt, max_length=256):
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                output_ids = model.generate(input_ids, max_length=max_length, num_beams=5, early_stopping=True)
                return tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Generate Summary
            summary_prompt = f"Summarize in detail:\n{text}"
            summary = generate(summary_prompt, max_length=300)

            # Important Questions
            q_prompt = f"Generate all important questions from this text:\n{text}"
            questions = generate(q_prompt)

            # MCQs
            mcq_prompt = f"Generate all unique MCQs with 4 options and correct answers clearly mentioned:\n{text}"
            mcqs = generate(mcq_prompt)

            # Flashcards
            flashcard_prompt = f"Generate multiple flashcards as Question and Answer pairs:\n{text}"
            flashcards = generate(flashcard_prompt)

            # Display output
            st.subheader("📝 Summary")
            st.write(summary)

            st.subheader("❓ Important Questions")
            st.markdown(questions.replace("\n", "\n- "))

            st.subheader("🧠 Multiple Choice Questions (MCQs)")
            st.markdown(mcqs.replace("A)", "**A)**").replace("B)", "**B)**").replace("C)", "**C)**").replace("D)", "**D)**"))

            st.subheader("📇 Flashcards")
            st.markdown(flashcards.replace("Answer:", "\n**Answer:**"))

            # Export options
            export_option = st.selectbox("📤 Export as", ["None", "PDF", "DOCX"])
            if export_option != "None":
                if export_option == "PDF":
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, f"Summary:\n{summary}\n\nQuestions:\n{questions}\n\nMCQs:\n{mcqs}\n\nFlashcards:\n{flashcards}")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                        pdf.output(tmp_pdf.name)
                        st.download_button("📥 Download PDF", data=open(tmp_pdf.name, "rb").read(), file_name="study_assistant.pdf")
                else:
                    doc = Document()
                    doc.add_heading("AI Study Assistant Output", level=1)
                    doc.add_heading("Summary", level=2)
                    doc.add_paragraph(summary)
                    doc.add_heading("Important Questions", level=2)
                    doc.add_paragraph(questions)
                    doc.add_heading("MCQs", level=2)
                    doc.add_paragraph(mcqs)
                    doc.add_heading("Flashcards", level=2)
                    doc.add_paragraph(flashcards)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_doc:
                        doc.save(tmp_doc.name)
                        st.download_button("📥 Download DOCX", data=open(tmp_doc.name, "rb").read(), file_name="study_assistant.docx")
