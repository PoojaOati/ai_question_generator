import fasttext
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
from deep_translator import GoogleTranslator
from huggingface_hub import hf_hub_download

# Define a function to download the model from Hugging Face
def download_model_from_huggingface(model_name: str, file_name: str):
    # This will download the model file from your Hugging Face repo
    file_path = hf_hub_download(repo_id=model_name, filename=file_name)
    return file_path

# Load language detection model from Hugging Face
lang_model_path = download_model_from_huggingface("Pooja1218/ai-quiz-generator", "lid.176.bin")
lang_model = fasttext.load_model(lang_model_path)

# Load question generation model
model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, max_chunk_size=500):
    sentences = text.split(".")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence.strip() + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence.strip() + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_questions_from_text(text):
    text_chunks = split_text(text)
    questions = []

    for chunk in text_chunks:
        lang_code = lang_model.predict(chunk.strip().replace("\n", " "))[0][0].replace("__label__", "")
        if lang_code != "en":
            try:
                chunk = GoogleTranslator(source=lang_code, target="en").translate(chunk)
            except Exception as e:
                questions.append(f"Translation failed: {e}")
                continue

        prompt = f"generate questions: {chunk}"
        try:
            output = qa_pipeline(prompt, max_length=64, do_sample=True, top_k=50, num_return_sequences=1)
            generated_text = output[0]['generated_text']
            if not generated_text.endswith("?"):
                generated_text += "?"
            questions.append(generated_text)
        except Exception as e:
            questions.append(f"Error generating question: {e}")

    return questions
