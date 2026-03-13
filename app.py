import gradio as gr
import os
import tempfile
import logging
from pdf_loader import load_and_chunk_pdfs
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from evaluation import evaluate, model as eval_model

logging.basicConfig(level=logging.INFO)

vector_store = VectorStore()
rag = RAGPipeline()

UPLOAD_DIR = tempfile.gettempdir()

def upload_pdfs(files):
    if not files:
        return gr.update(value="No files uploaded."), None
    file_paths = []
    for file in files:
        # Gradio with type='filepath' gives file as a string path
        if isinstance(file, str) and os.path.exists(file):
            file_paths.append(file)
        else:
            logging.warning(f"File {file} is not a valid path.")
    chunks = load_and_chunk_pdfs(file_paths)
    if not chunks:
        return gr.update(value="No text extracted from PDFs."), None
    vector_store.add_documents(chunks)
    return gr.update(value=f"Uploaded and indexed {len(chunks)} chunks from {len(files)} PDFs."), None

def answer_question_interface(question):
    if vector_store.is_empty():
        return "Database is empty. Please upload PDFs first.", None, None, None
    retrieved, question_emb = vector_store.similarity_search(question, top_k=5)
    context_chunks = [r["text"] for r in retrieved]
    context_embs = [vector_store._embed_texts([c])[0] for c in context_chunks]
    answer = rag.answer_question(context_chunks, question)
    answer_emb = eval_model.encode([answer], convert_to_numpy=True)[0]
    metrics = evaluate(
        question,
        answer,
        context_chunks,
        question_emb,
        answer_emb,
        context_embs
    )
    sources = [f"{r['metadata']['source']} (page {r['metadata']['page']}), score: {1-r['distance']:.2f}" for r in retrieved]
    return answer, metrics, sources, context_chunks

def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Local PDF RAG System with ChromaDB")
        with gr.Row():
            pdf_upload = gr.File(label="Upload PDFs", file_count="multiple", type="filepath")
            upload_btn = gr.Button("Build/Update Knowledge Base")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
        upload_btn.click(upload_pdfs, inputs=[pdf_upload], outputs=[upload_status, gr.State()])
        gr.Markdown("## Ask a Question about your PDFs")
        question = gr.Textbox(label="Your Question")
        ask_btn = gr.Button("Ask")
        answer = gr.Textbox(label="Answer", interactive=False)
        metrics = gr.JSON(label="Evaluation Metrics")
        sources = gr.JSON(label="Retrieved Sources")
        context = gr.JSON(label="Retrieved Chunks")
        ask_btn.click(answer_question_interface, inputs=[question], outputs=[answer, metrics, sources, context])
    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
