import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2

# Title
st.title("📚 AI Study Assistant (Generate Summary, Questions, MCQs)")

# Load Hugging Face models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# MCQ model setup
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# PDF reading function
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Inputs
uploaded_file = st.file_uploader("📄 Upload PDF Notes", type="pdf")
text_input = st.text_area("✍️ Or paste your notes here")

# Final content source
final_text = ""
if uploaded_file is not None:
    final_text = read_pdf(uploaded_file)
    st.success("✅ PDF uploaded successfully!")
elif text_input:
    final_text = text_input

if st.button("🤖 Generate Summary, Questions & MCQs"):
    if final_text.strip():
        with st.spinner("Generating Summary..."):
            input_chunk = final_text[:1024]
            summary = summarizer(input_chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            st.subheader("📝 Summary")
            st.write(summary)

        st.subheader("❓ 5 Important Questions")
        st.write("""
        1. What is the main idea of the topic?
        2. Explain the key concepts mentioned.
        3. What are the uses or applications of the topic?
        4. Describe one example mentioned in the notes.
        5. What are the challenges or limitations of this topic?
        """)

        with st.spinner("Generating MCQs..."):
            prompt = f"""
            Generate 5 multiple choice questions (MCQs) from the following text.
            Each question must include:
            - A question
            - 4 options labeled A, B, C, D
            - One correct answer labeled like: Answer: A or Answer: B

            Text: {input_chunk}
            """
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            output = model.generate(
                **inputs,
                max_length=800,
                num_beams=4,
                early_stopping=True
            )
            mcqs = tokenizer.decode(output[0], skip_special_tokens=True)

            st.subheader("🧠 MCQs")
            st.text(mcqs)
    else:
        st.warning("Please upload a PDF or paste your notes.")
