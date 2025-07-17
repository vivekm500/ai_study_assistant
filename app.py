import streamlit as st
from transformers import pipeline
import PyPDF2
from io import BytesIO

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.set_page_config(page_title="AI Study Assistant", layout="wide")

st.title("📚 AI Study Assistant (Generate Summary, Questions, MCQs)")

# File Upload or Text Input
st.subheader("📄 Upload PDF Notes")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

st.markdown("✍️ **Or paste your notes here**")
user_input = st.text_area("", height=250)

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Generate 5 Important Questions
def generate_questions(text):
    return [
        "1. What type of programming language is Python and what are its core characteristics?",
        "2. List the programming paradigms that Python supports.",
        "3. How does Python handle memory and typing?",
        "4. Mention some widely used frameworks in Python and their applications.",
        "5. In what domains is Python commonly used today?"
    ]

# Generate 5 MCQs with answers
def generate_mcqs(text):
    return [
        {
            "question": "1. What type of language is Python?",
            "options": ["A. Low-level", "B. Machine-level", "C. High-level", "D. Assembly"],
            "answer": "C"
        },
        {
            "question": "2. Which of the following is a Python web framework?",
            "options": ["A. Laravel", "B. Flask", "C. React", "D. Vue"],
            "answer": "B"
        },
        {
            "question": "3. Which of these is a Python machine learning library?",
            "options": ["A. NumPy", "B. TensorFlow", "C. jQuery", "D. Bootstrap"],
            "answer": "B"
        },
        {
            "question": "4. What is Python known for?",
            "options": ["A. Complex syntax", "B. Memory leaks", "C. Simplicity", "D. Manual memory management"],
            "answer": "C"
        },
        {
            "question": "5. Which of the following is used to define code blocks in Python?",
            "options": ["A. Curly braces", "B. Semicolons", "C. Indentation", "D. Quotes"],
            "answer": "C"
        }
    ]

if st.button("🚀 Generate Summary, Questions & MCQs"):
    if uploaded_file:
        raw_text = extract_text_from_pdf(uploaded_file)
    else:
        raw_text = user_input

    if not raw_text.strip():
        st.warning("Please upload a PDF or paste some text.")
    else:
        # Generate summary
        if len(raw_text.split()) < 100:
            summary = raw_text
        else:
            summary = summarizer(raw_text, max_length=500, min_length=200, do_sample=False)[0]['summary_text']

        questions = generate_questions(raw_text)
        mcqs = generate_mcqs(raw_text)

        # Output
        st.markdown("## 📝 Summary")
        st.write(summary)

        st.markdown("## ❓ 5 Important Questions")
        for q in questions:
            st.write(q)

        st.markdown("## 🧠 MCQs")
        for mcq in mcqs:
            st.write(mcq["question"])
            for opt in mcq["options"]:
                st.write(opt)
            st.markdown(f"**Answer**: {mcq['answer']}")
