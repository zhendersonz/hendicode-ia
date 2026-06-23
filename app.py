from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from agent.core import process_stream, handle_command
from agent.model import hendi_model
from agent.memory import hendi_memory
import json
import os
import sys
import threading
import time

app = Flask(__name__)

# ── CORS ─────────────────────────────────────────────────────────────────────
try:
    CORS(app)
except Exception:
    @app.after_request
    def _cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        return response

# ── Estatísticas em memória ─────────────────────────────────────────────────
_stats = {
    "started_at": time.time(),
    "total_interactions": 0,
    "total_tokens_generated": 0,
    "feedback_received": 0,
}

# ── Pré-carregamento ────────────────────────────────────────────────────────
def preload():
    print("[HendiCode] Iniciando pré-carregamento do cérebro...")
    hendi_model.load()
    print("[HendiCode] Cérebro carregado e pronto!")

threading.Thread(target=preload).start()

# ── Rota principal ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        print(f"[ERRO] /: {e}")
        return jsonify({"erro": str(e)}), 500

# ── Chat com streaming ─────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        timeout = 60
        waited = 0
        while not hendi_model.loaded and waited < timeout:
            time.sleep(1)
            waited += 1

        if not hendi_model.loaded:
            return jsonify({"erro": "Modelo ainda carregando. Aguarde e tente novamente."}), 503

        data = request.get_json()
        message = data.get("message", "").strip()
        history = data.get("history", [])

        if not message:
            return jsonify({"response": "Digite um comando."})

        _stats["total_interactions"] += 1

        def generate():
            tokens = 0
            for chunk in process_stream(message, history):
                tokens += 1
                yield chunk
            _stats["total_tokens_generated"] += tokens

        return Response(stream_with_context(generate()), mimetype="text/plain")
    except Exception as e:
        print(f"[ERRO] /api/chat: {e}")
        return jsonify({"erro": str(e)}), 500

# ── Health check ────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    try:
        return jsonify({
            "status": "ok",
            "modelo_carregado": hendi_model.loaded,
            "device": getattr(hendi_model, "device", "cpu"),
        })
    except Exception as e:
        print(f"[ERRO] /api/health: {e}")
        return jsonify({"erro": str(e)}), 500

# ── Feedback ────────────────────────────────────────────────────────────────
@app.route("/api/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json()
        user_msg = data.get("message") or data.get("user_msg", "")
        assistant_msg = data.get("response") or data.get("assistant_msg", "")
        positive = data.get("positive", True)

        if not user_msg or not assistant_msg:
            return jsonify({"erro": "Campos 'message' e 'response' são obrigatórios"}), 400

        hendi_memory.save_feedback(user_msg, assistant_msg, positive)
        if not positive:
            palavras = set(assistant_msg.lower().split()[:5])
            for p in palavras:
                if len(p) > 3:
                    hendi_memory.save_blacklist(p)

        _stats["feedback_received"] += 1
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"[ERRO] /api/feedback: {e}")
        return jsonify({"erro": str(e)}), 500

# ── Resumo da memória ──────────────────────────────────────────────────────
@app.route("/api/memory", methods=["GET"])
def memory_summary():
    try:
        summary = hendi_memory.get_context_summary()
        facts = hendi_memory.data.get("facts", [])
        preferences = hendi_memory.data.get("preferences", [])
        feedback_count = hendi_memory.data.get("feedback_count", 0)
        user_name = hendi_memory.data.get("user_name", "Usuário")

        return jsonify({
            "user_name": user_name,
            "facts": facts,
            "preferences": preferences,
            "feedback_count": feedback_count,
            "summary": summary,
        })
    except Exception as e:
        print(f"[ERRO] /api/memory: {e}")
        return jsonify({"erro": str(e)}), 500

# ── Limpar memória ─────────────────────────────────────────────────────────
@app.route("/api/memory/clear", methods=["POST"])
def memory_clear():
    try:
        data = request.get_json() or {}
        confirm = data.get("confirm", False)

        if not confirm:
            return jsonify({"erro": "Confirmação necessária. Envie {\"confirm\": true}."}), 400

        hendi_memory.data["facts"] = []
        hendi_memory.data["preferences"] = []
        hendi_memory.data["feedback_count"] = 0
        hendi_memory._save_memory()

        if os.path.exists(hendi_memory.history_file):
            open(hendi_memory.history_file, "w").close()

        return jsonify({"status": "ok", "mensagem": "Memória limpa com sucesso."})
    except Exception as e:
        print(f"[ERRO] /api/memory/clear: {e}")
        return jsonify({"erro": str(e)}), 500

# ── Estatísticas ────────────────────────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
def stats():
    try:
        uptime = time.time() - _stats["started_at"]
        return jsonify({
            "total_interactions": _stats["total_interactions"],
            "total_tokens_generated": _stats["total_tokens_generated"],
            "feedback_received": _stats["feedback_received"],
            "uptime_seconds": round(uptime, 2),
            "modelo_carregado": hendi_model.loaded,
            "device": getattr(hendi_model, "device", "cpu"),
        })
    except Exception as e:
        print(f"[ERRO] /api/stats: {e}")
        return jsonify({"erro": str(e)}), 500

# ── Servidor ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug, threaded=True, use_reloader=False)
