#!pip install -q gradio sentence-transformers faiss-cpu langchain transformers torch

import gradio as gr
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Configuration - UPDATED MODELS
EMBEDDING_MODEL = "dbmdz/bert-base-turkish-uncased"
LLM_MODEL = "lserinol/bert-turkish-uncased"  # Verified working Turkish model
TEXT_FILE = "5_g.pdf"

# Load models - UPDATED MODEL LOADING
embedder = SentenceTransformer(EMBEDDING_MODEL)
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Prepare vector store (same as before)
def prepare_vector_store():
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)
    
    vectors = embedder.encode(chunks)
    vector_store = FAISS.from_embeddings(
        list(zip(chunks, vectors)),
        embedder
    )
    return vector_store

vector_store = prepare_vector_store()

# UPDATED RAG PIPELINE
def turkish_rag(query):
    relevant_docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Improved prompt for Turkish
    prompt = f"""tr: Soru: {query}
    Bağlam: {context}
    Cevap:"""
    
    result = generator(
        prompt,
        max_length=500,
        num_beams=5,
        early_stopping=True
    )
    
    return result[0]['generated_text']

# Gradio Interface (same as before)
with gr.Blocks(theme=gr.themes.Soft(), title="Türkçe RAG Sistemi") as demo:
    gr.Markdown("# 🇹🇷 Türkçe Soru-Cevap Sistemi")
    gr.Markdown("Belgelerinizden akıllı cevaplar alın!")
    
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Sorunuzu yazın", placeholder="Türkiye'nin başkenti neresidir?")
            submit_btn = gr.Button("Cevapla")
        
        answer = gr.Textbox(label="Cevap", interactive=False)
    
    examples = gr.Examples(
        examples=["Yapay zeka nedir?", "Makine öğrenmesi nasıl çalışır?"],
        inputs=question
    )
    
    submit_btn.click(
        fn=turkish_rag,
        inputs=question,
        outputs=answer
    )

demo.launch(server_name="0.0.0.0", server_port=7860)