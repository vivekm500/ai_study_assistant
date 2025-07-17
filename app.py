import streamlit as st
from PyPDF2 import PdfReader
import speech_recognition as sr
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load FLAN-T5-Base model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

st.set_page_config(page_title="📚 AI Study Assistant", layout="centered")
st.title("📚 AI Study Assistant")

# --- Input Section ---
st.subheader("✍️ Enter text to analyze")
text_input = st.text_area("Paste your study content here")

# --- Upload PDF ---
st.subheader("📄 Or upload a PDF file")
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
if pdf_file is not None:
    reader = PdfReader(pdf_file)
    text_input = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    st.success("✅ Extracted text from PDF!")

# --- Upload WAV ---
st.subheader("🎤 Or upload a WAV voice note")
audio_file = st.file_uploader("Upload a WAV file", type=["wav"])
if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    st.audio("temp_audio.wav")

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile("temp_audio.wav") as source:
            audio_data = recognizer.record(source)
            voice_text = recognizer.recognize_google(audio_data)
            st.success("✅ Transcribed voice note!")
            text_input = voice_text
    except Exception as e:
        st.error(f"Speech recognition failed: {e}")

# --- Processing Function ---
def generate_output(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Generate Output ---
if st.button("🚀 Generate Study Material"):
    if not text_input.strip():
        st.warning("Please provide input via text, PDF, or audio.")
    else:
        with st.spinner("Generating summary..."):
            summary = generate_output("summarize in detail: " + text_input)

        with st.spinner("Generating important questions..."):
            questions = generate_output("generate all possible important questions: " + text_input)

        with st.spinner("Generating MCQs..."):
            mcqs = generate_output("generate unique and well-formatted mcqs with no repeated options and 4 choices on new lines: " + text_input)

        with st.spinner("Generating flashcards..."):
            flashcards = generate_output("generate well-formatted flashcards in question and answer format without duplication: " + text_input)

        st.subheader("📝 Summary")
        st.write(summary)

        st.subheader("❓ Important Questions")
        st.markdown(f"```\n{questions}\n```")

        st.subheader("🧠 Multiple Choice Questions (MCQs)")
        st.markdown(f"```\n{mcqs}\n```")

        st.subheader("📇 Flashcards")
        st.markdown(f"```\n{flashcards}\n```")
