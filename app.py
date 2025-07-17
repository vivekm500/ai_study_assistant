import streamlit as st
from transformers import pipeline
import PyPDF2

# Summarizer model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# Generate Summary
def generate_summary(text):
    return summarizer(text, max_length=300, min_length=100, do_sample=False)[0]['summary_text']

# Dummy MCQ & Questions generator (replace later with real logic)
def generate_questions(text):
    return [
        "Q1: What is the main topic of the text?",
        "Q2: Name one key use case of the discussed topic."
    ]

def generate_mcqs(text):
    return [
        {
            "question": "What is Python primarily known for?",
            "options": ["Speed", "Complexity", "Simplicity", "Low-level Programming"],
            "answer": "Simplicity"
        },
        {
            "question": "Which framework is used for web development in Python?",
            "options": ["Laravel", "React", "Django", "Spring"],
            "answer": "Django"
        }
    ]

# Streamlit UI
st.title("📘 AI Study Assistant")

uploaded_file = st.file_uploader("Upload a PDF or paste text below")

text_input = st.text_area("Or paste text here")

if st.button("Generate Study Materials"):
    if uploaded_file:
        raw_text = extract_text_from_pdf(uploaded_file)
    elif text_input:
        raw_text = text_input
    else:
        st.warning("Please upload a PDF or paste some text.")
        st.stop()

    with st.spinner("Generating summary..."):
        summary = generate_summary(raw_text)

    st.subheader("📝 Summary")
    st.write(summary)

    st.subheader("❓ Important Questions")
    for q in generate_questions(summary):
        st.markdown(f"- {q}")

    st.subheader("🧠 MCQs")
    for mcq in generate_mcqs(summary):
        st.markdown(f"**{mcq['question']}**")
        for i, option in enumerate(mcq['options']):
            st.markdown(f"{chr(65+i)}. {option}")
        st.markdown(f"**Answer:** {mcq['answer']}")
