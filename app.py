import streamlit as st
from question_gen import extract_text_from_pdf, generate_questions_from_text
import base64
import io

st.set_page_config(page_title="PDF to Questions", layout="centered")

st.title("📄 PDF to Questions Generator")
st.markdown("Upload a **PDF** file and generate questions from the content automatically.")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    st.success("📄 PDF uploaded successfully!")

    if st.button("Generate Questions"):
        with st.spinner("Extracting text and generating questions..."):
            text = extract_text_from_pdf(uploaded_file)
            questions = generate_questions_from_text(text)

            if questions:
                st.subheader("📝 Generated Questions:")
                for idx, q in enumerate(questions, 1):
                    st.markdown(f"**{idx}.** {q}")

                # Allow download
                output = io.StringIO()
                for q in questions:
                    output.write(q + "\n")
                b64 = base64.b64encode(output.getvalue().encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="questions.txt">📥 Download Questions</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("No questions generated. Please check if the PDF contains valid English content.")
