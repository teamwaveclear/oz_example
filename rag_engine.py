"""
RAG engine: document loading, Hugging Face embeddings, FAISS search, Gemini LLM.

This module is a refactor of the original `rag_example.py` so it can be reused
by the Flask web app. State (chunks + FAISS index) is held inside a single
`RAGEngine` instance that is initialised once at app startup.
"""

import os
import time
import threading

import faiss
import numpy as np
import nltk

from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from nltk.tokenize import sent_tokenize


# ==========================================================
# HARD-CODED TOKENS (kept as-is per requirements)
# ==========================================================

GEMINI_API_KEY = ""
HF_TOKEN = ""


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_FOLDER = "data"

HF_EMBEDDING_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
GEMINI_MODEL = "gemini-3-flash-preview"

TOP_K = 3
BATCH_SIZE = 8


def _log(msg: str) -> None:
    """Flushed print so progress is visible behind the Flask request log."""
    print(f"[rag] {msg}", flush=True)


# ==========================================================
# NLTK SETUP
# ==========================================================

def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# ==========================================================
# DOCUMENT LOADING
# ==========================================================

def load_documents(folder=DATA_FOLDER):
    """Load .txt files from the data folder and split them into sentences."""
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Folder '{folder}' does not exist. Create it and put .txt files inside."
        )

    chunks = []
    sources = []

    for file_name in sorted(os.listdir(folder)):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(folder, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        for sentence in sent_tokenize(text):
            sentence = sentence.strip()
            if sentence:
                chunks.append(sentence)
                sources.append(file_name)

    if not chunks:
        raise ValueError(
            f"No text found. Make sure the '{folder}' folder contains .txt files."
        )

    return chunks, sources


# ==========================================================
# HUGGING FACE EMBEDDINGS
# ==========================================================

def _normalize_embedding_output(raw_output, expected_count):
    """Convert raw HF embedding output to a clean 2D float32 numpy array."""
    arr = np.array(raw_output, dtype="float32")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        if arr.shape[0] == expected_count:
            pass
        elif expected_count == 1:
            arr = arr.mean(axis=0, keepdims=True)
        else:
            raise ValueError(
                f"Unexpected 2D embedding shape: {arr.shape}, "
                f"expected_count={expected_count}"
            )
    elif arr.ndim == 3:
        arr = arr.mean(axis=1)
    else:
        raise ValueError(f"Unexpected embedding dimensions: {arr.ndim}")

    if arr.shape[0] != expected_count:
        raise ValueError(
            f"Embedding count mismatch. Expected {expected_count}, got {arr.shape[0]}"
        )

    return arr.astype("float32")


# ==========================================================
# RAG ENGINE
# ==========================================================

class RAGEngine:
    """Encapsulates the whole RAG pipeline as a single in-memory object."""

    def __init__(self, data_folder=DATA_FOLDER):
        self.data_folder = data_folder
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.index = None
        self.ready = False
        self.status = "not_initialised"
        self.progress = {"current": 0, "total": 0}

        self._lock = threading.Lock()

        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        self.hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

    # ---------- initialisation ----------

    def initialise(self):
        """Load docs, build embeddings, build FAISS index. Idempotent."""
        with self._lock:
            if self.ready:
                return

            self.status = "downloading_nltk"
            _log("Setting up NLTK tokenizer...")
            setup_nltk()

            self.status = "loading_documents"
            _log(f"Loading documents from '{self.data_folder}'...")
            self.chunks, self.sources = load_documents(self.data_folder)
            _log(f"Loaded {len(self.chunks)} sentences from "
                 f"{len(set(self.sources))} file(s).")

            self.status = "embedding_documents"
            _log("Embedding documents via Hugging Face...")
            embeddings = self._embed_texts(self.chunks)

            self.status = "building_index"
            _log("Building FAISS index...")
            self.index = self._create_faiss_index(embeddings)

            self.ready = True
            self.status = "ready"
            _log(f"Ready. {self.index.ntotal} vectors indexed.")

    # ---------- embeddings ----------

    def _hf_feature_extraction_with_retries(self, inputs, expected_count, max_retries=5):
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                result = self.hf_client.feature_extraction(
                    inputs, model=HF_EMBEDDING_MODEL
                )
                return _normalize_embedding_output(
                    raw_output=result, expected_count=expected_count
                )
            except Exception as exc:
                last_error = exc
                _log(f"Embedding call failed (attempt {attempt}/{max_retries}): {exc}")
                if attempt == max_retries:
                    raise
                wait = attempt * 3
                _log(f"Retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"Embedding failed: {last_error}")

    def _embed_texts(self, texts, batch_size=BATCH_SIZE):
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        self.progress = {"current": 0, "total": total_batches}
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            batch_num = start // batch_size + 1
            _log(f"  batch {batch_num}/{total_batches} ({len(batch)} items)...")
            embeddings = self._hf_feature_extraction_with_retries(
                inputs=batch, expected_count=len(batch)
            )
            all_embeddings.append(embeddings)
            self.progress = {"current": batch_num, "total": total_batches}
        return np.vstack(all_embeddings).astype("float32")

    def _embed_query(self, query):
        embedding = self._hf_feature_extraction_with_retries(
            inputs=query, expected_count=1
        )
        return embedding.astype("float32")

    # ---------- FAISS ----------

    @staticmethod
    def _create_faiss_index(embeddings):
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def retrieve(self, query, k=TOP_K):
        """Return top-k retrieved chunks together with score + source filename."""
        if not self.ready:
            raise RuntimeError("RAG engine is not ready yet.")

        query_embedding = self._embed_query(query)
        faiss.normalize_L2(query_embedding)

        scores, indexes = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indexes[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "source": self.sources[idx] if idx < len(self.sources) else None,
                "score": float(score),
            })
        return results

    # ---------- Gemini ----------

    def ask_gemini(self, context, question, history=None):
        """
        Call Gemini with the retrieved context, conversation history and
        the user's current question.

        `history` is a list of {"role": "user"|"assistant", "content": str}
        entries (already excluding the current question).
        """
        history_text = ""
        if history:
            lines = []
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            history_text = "\n".join(lines)

        prompt = f"""You are a helpful RAG assistant.

Use the provided context and the previous conversation to answer the user's
latest question.

Rules:
1. First answer using only the provided context.
2. If the context does not contain enough information, say:
   "I do not have enough information in the documents, but based on general knowledge..."
3. Keep the answer simple and clear.
4. Do not invent facts from the documents.
5. Use the conversation history only to resolve references like "it" or
   "the previous one" - never invent earlier turns.

Conversation so far:
{history_text if history_text else "(no previous messages)"}

Context retrieved from documents:
{context}

User's latest question:
{question}

Answer:
"""

        response = self.gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=500,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()

    # ---------- high level helper ----------

    def answer(self, question, history=None, k=TOP_K):
        """Run the full retrieve + generate flow and return a structured dict."""
        retrieved = self.retrieve(question, k=k)
        context = "\n".join(item["text"] for item in retrieved)
        answer_text = self.ask_gemini(
            context=context, question=question, history=history or []
        )
        return {
            "answer": answer_text,
            "context": retrieved,
        }
