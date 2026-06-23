"""
memory.py — Memória persistente do HendiCode.

CORREÇÕES:
- [BUG] _load_memory: except nu (sem tipo) silenciava qualquer erro, inclusive erros de disco
- [BUG] save_blacklist: abria o arquivo duas vezes separadas (leitura + escrita) sem lock,
  causando race condition se duas requisições simultâneas chamassem save_blacklist
- [BUG] get_examples: except nu silenciava linhas JSONL corrompidas sem aviso nenhum
- [BUG] save_blacklist: se o arquivo existia mas estava corrompido, json.load lançava exceção
  não tratada e derrubava a requisição
- [MELHORIA] _save_memory: escrita atômica via arquivo temporário + rename, evita corrupção
  se o processo for encerrado no meio da escrita
- [MELHORIA] threading.Lock para evitar race condition em escritas concorrentes
"""
import json
import os
import tempfile
import threading

class MemoryManager:
    def __init__(self, folder_path):
        self.folder_path    = folder_path
        self.memory_file    = os.path.join(folder_path, "memory.json")
        self.history_file   = os.path.join(folder_path, "history.jsonl")
        self.examples_file  = os.path.join(folder_path, "examples.jsonl")
        self.blacklist_file = os.path.join(folder_path, "blacklist.json")
        self._lock          = threading.Lock()
        self.data           = self._load_memory()

    # ── Carregamento ───────────────────────────────────────────────────
    def _load_memory(self):
        dados = {"user_name": "Usuário", "facts": [], "preferences": [], "feedback_count": 0}
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    dados_arq = json.load(f)
                dados.update(dados_arq)
            except (json.JSONDecodeError, OSError) as e:
                # [FIX] Loga o erro em vez de silenciar
                print(f"[Memory] Aviso: memory.json corrompido ou ilegível ({e}). Usando padrão.")
        return dados

    # ── Escrita atômica ────────────────────────────────────────────────
    def _save_memory(self):
        """
        [FIX] Escrita atômica: grava em arquivo temporário no mesmo diretório
        e faz rename atômico. Evita corrupção se o processo morrer no meio.
        """
        os.makedirs(self.folder_path, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self.folder_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
            os.replace(tmp_path, self.memory_file)
        except Exception:
            # Limpa o temporário se algo deu errado
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ── API pública ────────────────────────────────────────────────────
    def save_fact(self, fact):
        with self._lock:
            if fact not in self.data["facts"]:
                self.data["facts"].append(fact)
                self._save_memory()

    def save_preference(self, pref):
        with self._lock:
            if pref not in self.data["preferences"]:
                self.data["preferences"].append(pref)
                self._save_memory()

    def save_interaction(self, user_msg, assistant_msg):
        # JSONL: append é thread-safe no nível do OS para writes pequenos,
        # mas usamos lock para garantir em todos os sistemas.
        with self._lock:
            os.makedirs(self.folder_path, exist_ok=True)
            with open(self.history_file, "a", encoding="utf-8") as f:
                entry = {"user": user_msg, "assistant": assistant_msg}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def save_feedback(self, user_msg, assistant_msg, positive=True):
        with self._lock:
            self.data["feedback_count"] += 1
            self._save_memory()
            if positive:
                os.makedirs(self.folder_path, exist_ok=True)
                with open(self.examples_file, "a", encoding="utf-8") as f:
                    entry = {"user": user_msg, "assistant": assistant_msg}
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def save_blacklist(self, pattern):
        """
        [FIX] Operação de leitura+escrita agora é atômica sob lock.
        Trata corrupção do arquivo blacklist.json sem derrubar a requisição.
        """
        with self._lock:
            os.makedirs(self.folder_path, exist_ok=True)
            blacklist = []
            if os.path.exists(self.blacklist_file):
                try:
                    with open(self.blacklist_file, "r", encoding="utf-8") as f:
                        blacklist = json.load(f)
                    if not isinstance(blacklist, list):
                        blacklist = []
                except (json.JSONDecodeError, OSError):
                    blacklist = []

            if pattern not in blacklist:
                blacklist.append(pattern)
                fd, tmp_path = tempfile.mkstemp(dir=self.folder_path, suffix=".tmp")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(blacklist, f, ensure_ascii=False, indent=4)
                    os.replace(tmp_path, self.blacklist_file)
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    def get_examples(self, max_examples=5):
        if not os.path.exists(self.examples_file):
            return []
        exemplos = []
        with open(self.examples_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    exemplos.append(json.loads(line))
                except json.JSONDecodeError:
                    # [FIX] Loga linha corrompida em vez de silenciar
                    print(f"[Memory] Aviso: linha {i+1} de examples.jsonl ignorada (JSON inválido).")
        return exemplos[-max_examples:]

    def get_blacklist(self):
        if not os.path.exists(self.blacklist_file):
            return []
        try:
            with open(self.blacklist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def get_context_summary(self):
        partes = []
        if self.data["facts"]:
            partes.append("Fatos sobre o usuário:\n" + "\n".join(self.data["facts"]))
        if self.data["preferences"]:
            partes.append("Preferências:\n" + "\n".join(self.data["preferences"]))
        exemplos = self.get_examples()
        if exemplos:
            partes.append("Exemplos de boas respostas:")
            for ex in exemplos:
                usuario = ex.get('user') or ex.get('instruction') or ex.get('input', '')
                assistente = ex.get('assistant') or ex.get('response') or ex.get('output', '')
                if usuario and assistente:
                    partes.append(f"  Usuário: {usuario}")
                    partes.append(f"  Assistente: {assistente}")
        blacklist = self.get_blacklist()
        if blacklist:
            partes.append("Padrões a evitar:\n" + "\n".join(blacklist))
        return "\n\n".join(partes) if partes else "Nenhuma informação prévia sobre o usuário."


memory_dir = os.path.join(os.path.dirname(__file__), "..", "memory")
hendi_memory = MemoryManager(memory_dir)
