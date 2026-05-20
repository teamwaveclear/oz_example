# RAG Assistant (Flask)
Alexk, Ramat Gan, 88391839
A Flask web application that wraps the original `rag_example.py` RAG pipeline
in a modern chat UI with per-session conversation memory.

- **Embeddings**: Hugging Face Inference API (`granite-embedding-97m-multilingual-r2`)
- **Vector search**: FAISS (in-memory, cosine via normalized inner product)
- **LLM**: Google Gemini (`gemini-3-flash-preview`)
- **Memory**: SQLite (`chat.db`) - sessions and messages persist across restarts
- **Frontend**: Plain HTML / CSS / vanilla JS (no build step)

## Project layout

```
oz_vibe/
├── app.py                  # Flask app + REST API
├── rag_engine.py           # RAG pipeline (refactored from rag_example.py)
├── database.py             # SQLite memory layer
├── rag_example.py          # Original script (kept for reference)
├── requirements.txt
├── data/                   # .txt knowledge base
│   └── Risk Analysis Report.txt
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

## Setup

> Requires Python 3.10+ (uses `list[dict]` / `X | None` typing).

```powershell
# from project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The API tokens are kept hard-coded in `rag_engine.py` as requested.

## Run

```powershell
python app.py
```

Then open <http://127.0.0.1:5000>.

The engine boots in the background:

1. NLTK tokenizer downloads (first run only)
2. `.txt` files in `data/` are sentence-tokenized
3. Each sentence is embedded via Hugging Face (this is the slow step)
4. A FAISS index is built in memory

The sidebar status dot turns green when the engine is ready. The input box
unlocks automatically. You can browse the UI and create conversations while
indexing is in progress; sending messages is blocked until the index is ready.

## Conversation memory

Every conversation is stored in `chat.db` (created next to `app.py`):

- `sessions(id, title, created_at, updated_at)`
- `messages(id, session_id, role, content, context_json, created_at)`

When you send a question, the last 20 messages of the current session are
passed to Gemini as conversation history together with the freshly retrieved
context, so the assistant can resolve references like *"summarize the
previous one"*.

To wipe memory, delete `chat.db`.

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/api/status` | Engine readiness + indexing status |
| `GET`  | `/api/sessions` | List sessions |
| `POST` | `/api/sessions` | Create a new session |
| `GET`  | `/api/sessions/<id>` | Session details + full message history |
| `PATCH`| `/api/sessions/<id>` | Rename session (`{ "title": "..." }`) |
| `DELETE`| `/api/sessions/<id>` | Delete session and its messages |
| `POST` | `/api/sessions/<id>/messages` | Send a question (`{ "content": "..." }`) and get an answer |

## Adding documents

Drop more `.txt` files into `data/` and restart `app.py`. The whole folder is
re-indexed on startup.
