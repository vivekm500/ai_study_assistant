import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
from googletrans import Translator

# Title
st.title("📚 AI Study Assistant (No Login)")

# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
translator = Translator()

# File/Text Input
uploaded_file = st.file_uploader("📄 Upload PDF Notes", type="pdf")
text_input = st.text_area("✍️ Or paste your notes here")
target_lang = st.selectbox("🌍 Translate output to:", ["None", "hi", "bn", "ta", "te", "gu"], index=0)

# Read PDF
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Final text to use
final_text = ""
if uploaded_file:
    final_text = read_pdf(uploaded_file)
    st.success("✅ PDF uploaded!")
elif text_input:
    final_text = text_input

# Main Action Button
if st.button("🤖 Generate Summary, Questions, MCQs, Flashcards"):
    if final_text.strip():
        input_chunk = final_text[:2048]

        # Summary
        summary_parts = summarizer(input_chunk, max_length=300, min_length=80, do_sample=False)
        full_summary = " ".join([part['summary_text'] for part in summary_parts])
        if target_lang != "None":
            full_summary = translator.translate(full_summary, dest=target_lang).text
        st.subheader("📝 Summary")
        st.write(full_summary)

        # Questions
        prompt_q = f"Generate all important short and descriptive questions based on:\n{input_chunk}"
        inputs_q = tokenizer(prompt_q, return_tensors="pt", max_length=1024, truncation=True)
        output_q = model.generate(**inputs_q, max_length=1024, num_beams=4, early_stopping=True)
        questions = tokenizer.decode(output_q[0], skip_special_tokens=True)
        if target_lang != "None":
            questions = translator.translate(questions, dest=target_lang).text
        st.subheader("❓ Important Questions")
        st.markdown("```text\n" + questions + "\n```")

        # MCQs
        prompt_mcq = f"Generate multiple MCQs with 4 options and correct answers from:\n{input_chunk}"
        inputs = tokenizer(prompt_mcq, return_tensors="pt", max_length=1024, truncation=True)
        output = model.generate(**inputs, max_length=1024, num_beams=4, early_stopping=True)
        mcqs = tokenizer.decode(output[0], skip_special_tokens=True)
        if target_lang != "None":
            mcqs = translator.translate(mcqs, dest=target_lang).text
        st.subheader("🧠 MCQs")
        st.markdown("```text\n" + mcqs + "\n```")

        # Flashcards
        prompt_fc = f"Generate flashcards (Q&A pairs) for revision from:\n{input_chunk}"
        inputs_fc = tokenizer(prompt_fc, return_tensors="pt", max_length=1024, truncation=True)
        output_fc = model.generate(**inputs_fc, max_length=1024, num_beams=4, early_stopping=True)
        flashcards = tokenizer.decode(output_fc[0], skip_special_tokens=True)
        if target_lang != "None":
            flashcards = translator.translate(flashcards, dest=target_lang).text
        st.subheader("📇 Flashcards")
        st.markdown("```text\n" + flashcards + "\n```")

        # Export
        st.download_button("📥 Export All as TXT", data=full_summary + "\n\n" + questions + "\n\n" + mcqs + "\n\n" + flashcards, file_name="study_output.txt")

    else:
        st.warning("⚠️ Please upload a PDF or paste your notes.")
