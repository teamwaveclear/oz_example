"""
Flask web app exposing the RAG pipeline as a chat interface with persistent
SQLite-backed conversation memory.
"""

import threading
import traceback

from flask import Flask, jsonify, render_template, request

import database
from rag_engine import RAGEngine


app = Flask(__name__)
engine = RAGEngine()


# ==========================================================
# BACKGROUND INITIALISATION
# ==========================================================
# Loading documents + computing embeddings can take a while (especially
# the first time NLTK has to download data). We do it in a background
# thread so the UI can render immediately and poll /api/status.

_init_error: dict | None = None


def _initialise_engine_background():
    global _init_error
    try:
        engine.initialise()
    except Exception as exc:
        _init_error = {
            "message": str(exc),
            "type": exc.__class__.__name__,
        }
        traceback.print_exc()


def _start_background_init():
    database.init_db()
    thread = threading.Thread(
        target=_initialise_engine_background, daemon=True, name="rag-init"
    )
    thread.start()


# ==========================================================
# ROUTES
# ==========================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "ready": engine.ready,
        "status": engine.status,
        "chunks": len(engine.chunks),
        "progress": engine.progress,
        "error": _init_error,
    })


# ---------- sessions ----------

@app.route("/api/sessions", methods=["GET"])
def api_list_sessions():
    return jsonify({"sessions": database.list_sessions()})


@app.route("/api/sessions", methods=["POST"])
def api_create_session():
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "New conversation").strip() or "New conversation"
    session = database.create_session(title=title)
    return jsonify(session), 201


@app.route("/api/sessions/<session_id>", methods=["GET"])
def api_get_session(session_id):
    session = database.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    session["messages"] = database.get_messages(session_id)
    return jsonify(session)


@app.route("/api/sessions/<session_id>", methods=["PATCH"])
def api_update_session(session_id):
    if not database.get_session(session_id):
        return jsonify({"error": "Session not found"}), 404

    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400

    database.update_session_title(session_id, title)
    return jsonify(database.get_session(session_id))


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def api_delete_session(session_id):
    if not database.get_session(session_id):
        return jsonify({"error": "Session not found"}), 404
    database.delete_session(session_id)
    return jsonify({"ok": True})


# ---------- chat ----------

@app.route("/api/sessions/<session_id>/messages", methods=["POST"])
def api_send_message(session_id):
    if not engine.ready:
        return jsonify({
            "error": "RAG engine is still initialising.",
            "status": engine.status,
        }), 503

    session = database.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    payload = request.get_json(silent=True) or {}
    question = (payload.get("content") or "").strip()
    if not question:
        return jsonify({"error": "content is required"}), 400

    history = database.get_history_for_llm(session_id, limit=20)
    user_msg = database.add_message(session_id, "user", question)

    try:
        result = engine.answer(question=question, history=history)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "error": "Failed to generate answer.",
            "detail": str(exc),
            "user_message": user_msg,
        }), 500

    assistant_msg = database.add_message(
        session_id,
        "assistant",
        result["answer"],
        context=result["context"],
    )

    if session["title"] == "New conversation":
        new_title = question[:60] + ("..." if len(question) > 60 else "")
        database.update_session_title(session_id, new_title)

    return jsonify({
        "user_message": user_msg,
        "assistant_message": assistant_msg,
    })


# ==========================================================
# ENTRY POINT
# ==========================================================

_start_background_init()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
