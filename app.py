import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import speech_recognition as sr
import tempfile
import os

# Title
st.title("📚 AI Study Assistant (with Voice Input, Summary, Questions & MCQs)")

# ✅ Use CPU-friendly summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ✅ Use smaller FLAN-T5 model to avoid memory issues on Streamlit Cloud
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# PDF reading function
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Speech to text
def convert_audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

# Inputs
uploaded_file = st.file_uploader("📄 Upload PDF Notes", type="pdf")
audio_file = st.file_uploader("🎤 Upload voice note (WAV only)", type=["wav"])
text_input = st.text_area("✍️ Or paste your notes here")

# Final content source
final_text = ""
if uploaded_file is not None:
    final_text = read_pdf(uploaded_file)
    st.success("✅ PDF uploaded successfully!")
elif audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio_file.read())
        final_text = convert_audio_to_text(tmp_audio.name)
        os.remove(tmp_audio.name)
        st.success("✅ Voice note converted to text!")
elif text_input:
    final_text = text_input

# Main logic
if st.button("🤖 Generate Summary, Questions & MCQs"):
    if final_text.strip():
        with st.spinner("Generating Detailed Summary..."):
            input_chunk = final_text[:2048]
            summary_parts = summarizer(input_chunk, max_length=300, min_length=80, do_sample=False)
            full_summary = " ".join([part['summary_text'] for part in summary_parts])
            st.subheader("📝 Summary")
            st.write(full_summary)

        with st.spinner("Generating Important Questions..."):
            prompt_q = f"""
Generate as many important short-answer or descriptive questions as possible from the following content.
Questions should cover all aspects and topics discussed.

Text:
{input_chunk}
"""
            inputs_q = tokenizer(prompt_q, return_tensors="pt", max_length=1024, truncation=True)
            output_q = model.generate(**inputs_q, max_length=1024, num_beams=4, early_stopping=True)
            questions = tokenizer.decode(output_q[0], skip_special_tokens=True)
            st.subheader("❓ Important Questions")
            st.markdown("```text\n" + questions + "\n```")

        with st.spinner("Generating MCQs..."):
            prompt = f"""
You are a helpful education assistant. Based on the following passage, generate as many high-quality multiple choice questions (MCQs) as possible.

Each question must:
- Be based on the passage
- Have 4 options labeled A, B, C, D
- Include a final line like: "Answer: B"

Here is the passage:
{input_chunk}
"""
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            output = model.generate(
                **inputs,
                max_length=1024,
                num_beams=4,
                early_stopping=True
            )
            mcqs = tokenizer.decode(output[0], skip_special_tokens=True)
            st.subheader("🧠 MCQs")
            st.markdown("```text\n" + mcqs + "\n```")
    else:
        st.warning("Please upload a PDF, voice note, or paste your notes.")
