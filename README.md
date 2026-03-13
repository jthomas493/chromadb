# Local PDF RAG System with ChromaDB

This application provides a local Retrieval-Augmented Generation (RAG) system using ChromaDB and a Gradio interface. Users can upload multiple PDF files, which are automatically chunked and indexed for semantic search and question answering.

## Features
- Drag-and-drop PDF upload
- Automatic text extraction and chunking (chunk size ~800, overlap ~200)
- Embedding generation with `all-MiniLM-L6-v2` (sentence-transformers)
- Persistent ChromaDB vector store at `./chromadb_store`
- Metadata includes source file and page number
- Gradio UI for uploading, updating, and querying
- Answers questions using only retrieved context (RAG)
- Evaluation metrics: groundedness, relevance, answer similarity, confidence
- Clear display of answer, metrics, and retrieved sources

## File Structure
- `app.py`: Gradio UI and main app logic
- `pdf_loader.py`: PDF extraction and chunking
- `vector_store.py`: Embedding and ChromaDB management
- `rag_pipeline.py`: LLM prompt and answer logic
- `evaluation.py`: Evaluation metrics
- `requirements.txt`: All dependencies

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your OpenAI API key (for LLM):
   ```bash
   export OPENAI_API_KEY=your-key
   ```
3. Run the app:
   ```bash
   python app.py
   ```

## Notes
- The vector store is persistent in `./chromadb_store`.
- The app uses `gpt-3.5-turbo` by default; override with `OPENAI_MODEL` env variable.
- All processing is local except for LLM calls.
- Evaluation metrics are shown after each answer.

## Requirements
See `requirements.txt` for all required packages.

**Update:** Now uses a local LLM via Ollama (no OpenAI API key required). Make sure Ollama is installed and running (https://ollama.com/). Start a model, e.g.:

    ollama run llama2

To use a different model, set:

    export OLLAMA_MODEL=your-model-name

All LLM inference is local.