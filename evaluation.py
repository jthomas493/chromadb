import numpy as np
from sentence_transformers import util, SentenceTransformer
import re
import logging

logging.basicConfig(level=logging.INFO)

model = SentenceTransformer("all-MiniLM-L6-v2")

def token_overlap(answer: str, context: str) -> float:
    answer_tokens = set(re.findall(r"\w+", answer.lower()))
    context_tokens = set(re.findall(r"\w+", context.lower()))
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)

def cosine_similarity(a_emb, b_emb) -> float:
    a_arr = np.array(a_emb, dtype=np.float32)
    b_arr = np.array(b_emb, dtype=np.float32)
    return float(util.cos_sim(a_arr, b_arr).item())

def evaluate(
    question: str,
    answer: str,
    context_chunks: list,
    question_emb: list,
    answer_emb: list,
    context_embs: list
) -> dict:
    context_text = " ".join(context_chunks)
    groundedness = token_overlap(answer, context_text)
    relevance = np.mean([
        cosine_similarity(question_emb, c_emb) for c_emb in context_embs
    ])
    answer_similarity = np.mean([
        cosine_similarity(answer_emb, c_emb) for c_emb in context_embs
    ])
    confidence = 0.4 * relevance + 0.4 * groundedness + 0.2 * answer_similarity
    logging.info(f"Groundedness: {groundedness:.2f}, Relevance: {relevance:.2f}, Answer Sim: {answer_similarity:.2f}, Confidence: {confidence:.2f}")
    return {
        "groundedness": groundedness,
        "relevance": relevance,
        "answer_similarity": answer_similarity,
        "confidence": confidence
    }
