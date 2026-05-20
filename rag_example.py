import os
import time
import faiss
import numpy as np
import nltk

from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from nltk.tokenize import sent_tokenize


# ==========================================================
# HARD-CODED TOKENS
# ==========================================================

GEMINI_API_KEY = ""
HF_TOKEN = ""


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_FOLDER = "data"

# Hugging Face cloud embedding model
HF_EMBEDDING_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"

# You can also try:
# HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini cloud LLM
GEMINI_MODEL = "gemini-3-flash-preview"

TOP_K = 3

# Start with 1 to avoid connection problems.
# Later you can try 4 or 8.
BATCH_SIZE = 1


# ==========================================================
# CLIENTS
# ==========================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)


# ==========================================================
# NLTK SETUP
# ==========================================================

def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# ==========================================================
# LOAD DOCUMENTS
# ==========================================================

def load_documents(folder=DATA_FOLDER):
    """
    Load .txt files from the data folder and split them into text chunks.
    """

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Folder '{folder}' does not exist. Create it and put .txt files inside."
        )

    chunks = []

    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            sentences = sent_tokenize(text)
            print("Sentences:", sentences)
            for sentence in sentences:
                sentence = sentence.strip()

                if sentence:
                    chunks.append(sentence)

    if not chunks:
        raise ValueError(
            f"No text found. Make sure the '{folder}' folder contains .txt files."
        )

    print(f"Loaded {len(chunks)} text chunks.")
    return chunks


# ==========================================================
# HUGGING FACE CLOUD EMBEDDINGS
# ==========================================================

def normalize_embedding_output(raw_output, expected_count):
    """
    Converts Hugging Face embedding output into a clean 2D numpy array.

    Final shape:
        [number_of_texts, embedding_dimension]
    """

    arr = np.array(raw_output, dtype="float32")

    # Case 1:
    # Single embedding:
    # [embedding_dimension]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Case 2:
    # Batch embeddings:
    # [batch_size, embedding_dimension]
    elif arr.ndim == 2:
        if arr.shape[0] == expected_count:
            pass

        # Token embeddings for one input:
        # [tokens, embedding_dimension]
        elif expected_count == 1:
            arr = arr.mean(axis=0, keepdims=True)

        else:
            raise ValueError(
                f"Unexpected 2D embedding shape: {arr.shape}, "
                f"expected_count={expected_count}"
            )

    # Case 3:
    # Token embeddings for batch:
    # [batch_size, tokens, embedding_dimension]
    elif arr.ndim == 3:
        arr = arr.mean(axis=1)

    else:
        raise ValueError(f"Unexpected embedding dimensions: {arr.ndim}")

    if arr.shape[0] != expected_count:
        raise ValueError(
            f"Embedding count mismatch. Expected {expected_count}, got {arr.shape[0]}"
        )

    return arr.astype("float32")


def hf_feature_extraction_with_retries(inputs, expected_count, max_retries=5):
    """
    Calls Hugging Face cloud embedding model with retries.

    inputs can be:
    - string
    - list of strings
    """

    for attempt in range(1, max_retries + 1):
        try:
            result = hf_client.feature_extraction(
                inputs,
                model=HF_EMBEDDING_MODEL
            )

            embeddings = normalize_embedding_output(
                raw_output=result,
                expected_count=expected_count
            )

            return embeddings

        except Exception as e:
            print(f"Hugging Face embedding failed. Attempt {attempt}/{max_retries}")
            print("Error:", e)

            if attempt == max_retries:
                raise

            wait_seconds = attempt * 3
            print(f"Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)


def embed_texts_with_huggingface(texts, batch_size=BATCH_SIZE):
    """
    Creates document embeddings using Hugging Face cloud inference.
    """

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        current_batch = start // batch_size + 1
        print(f"Embedding batch {current_batch}/{total_batches}...")

        embeddings = hf_feature_extraction_with_retries(
            inputs=batch,
            expected_count=len(batch)
        )

        all_embeddings.append(embeddings)

    final_embeddings = np.vstack(all_embeddings).astype("float32")
    print("Final embeddings:", final_embeddings)
    print(f"Created document embeddings. Shape: {final_embeddings.shape}")

    return final_embeddings


def embed_query_with_huggingface(query):
    """
    Creates one query embedding using Hugging Face cloud inference.
    """

    embedding = hf_feature_extraction_with_retries(
        inputs=query,
        expected_count=1
    )

    return embedding.astype("float32")


# ==========================================================
# FAISS VECTOR SEARCH
# ==========================================================

def create_faiss_index(embeddings):
    """
    Creates FAISS index.

    We normalize vectors and use inner product.
    This behaves like cosine similarity.
    """

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"FAISS index created with {index.ntotal} vectors.")

    return index


def retrieve(query, index, chunks, k=TOP_K):
    """
    Embeds the user question with Hugging Face and searches FAISS.
    """

    query_embedding = embed_query_with_huggingface(query)

    faiss.normalize_L2(query_embedding)

    scores, indexes = index.search(query_embedding, k)

    print("\nFAISS scores:", scores)
    print("FAISS indexes:", indexes)

    retrieved_chunks = []

    for idx in indexes[0]:
        if idx != -1:
            retrieved_chunks.append(chunks[idx])

    return retrieved_chunks


# ==========================================================
# GEMINI LLM
# ==========================================================

def ask_gemini(context, question):
    """
    Gemini is the LLM.
    Hugging Face is only used for embeddings.
    """

    prompt = f"""
You are a helpful RAG assistant.

Use the provided context to answer the user's question.

Rules:
1. First answer using only the provided context.
2. If the context does not contain enough information, say:
   "I do not have enough information in the documents, but based on general knowledge..."
3. Keep the answer simple and clear.
4. Do not invent facts from the documents.

Context:
{context}

Question:
{question}

Answer:
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=500,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            )
        )
    )

    return response.text.strip()


# ==========================================================
# MAIN APP
# ==========================================================

def main():
    setup_nltk()

    print("Loading documents...")
    chunks = load_documents(DATA_FOLDER)

    print("\nCreating Hugging Face cloud embeddings...")
    document_embeddings = embed_texts_with_huggingface(chunks)

    print("\nCreating FAISS index...")
    index = create_faiss_index(document_embeddings)

    print("\nRAG system is ready.")
    print("Embeddings: Hugging Face cloud")
    print("Vector search: FAISS local")
    print("LLM: Gemini cloud")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nAsk something: ").strip()

        if question.lower() == "exit":
            print("Goodbye.")
            break

        if not question:
            print("Please enter a real question.")
            continue

        top_chunks = retrieve(
            query=question,
            index=index,
            chunks=chunks,
            k=TOP_K
        )

        context = "\n".join(top_chunks)

        print("\nRetrieved Context:")
        print(context)

        answer = ask_gemini(context, question)

        print("\nGemini Answer:")
        print(answer)


if __name__ == "__main__":
    main()