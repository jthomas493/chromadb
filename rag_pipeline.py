import os
from typing import List, Dict, Any
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging

logging.basicConfig(level=logging.INFO)

LLM_MODEL = os.environ.get("HF_MODEL", "distilgpt2")


class RAGPipeline:
    def __init__(self):
        # Uses HuggingFace Transformers pipeline for local LLM inference
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)

    def build_prompt(self, context_chunks: List[str], question: str) -> str:
        context = "\n\n".join(context_chunks)
        prompt = (
            "You are an expert assistant. Answer the following question strictly using ONLY the provided context. "
            "If the answer is not in the context, say 'I don't know.'\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer: "
        )
        return prompt

    def answer_question(self, context_chunks: List[str], question: str) -> str:
        prompt = self.build_prompt(context_chunks, question)
        response = self.llm.invoke(prompt)
        if hasattr(response, 'content'):
            return response.content.strip()
        if isinstance(response, list) and len(response) > 0:
            # HuggingFacePipeline returns a list of dicts
            return response[0].get('generated_text', '').strip()
        return str(response).strip()
