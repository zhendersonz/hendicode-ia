"""
core.py — Núcleo do HendiCode.
Formato: ### Instruction: / ### Response:
"""
import re
from . import tools
from .memory import hendi_memory
from .model import hendi_model
from .retriever import retriever

MAX_HISTORY_TURNS = 10
MAX_RETRIES = 3

SISTEMA = """Você é o HendiCode, uma inteligência artificial especializada em programação e lógica.
Você foi treinado do zero com dados em português e código pelo seu criador Henderson.
Regras importantes:
1. Responda SEMPRE em português, independente do idioma da pergunta
2. Seja preciso, direto e útil — nada de rodeios
3. Quando der exemplos de código, use markdown com ```linguagem
4. Se não souber a resposta, admita — não invente
5. Mantenha respostas concisas (2-4 parágrafos no máximo, a menos que peçam mais detalhes)
6. Prefira exemplos práticos a explicações teóricas longas
7. Você possui memória persistente e lembra fatos sobre o usuário"""


def _detectar_idioma(texto: str) -> str:
    palavras_pt = r"\b(?:é|uma|como|para|com|dos|das|mais|mas|por|que|são|tem|está|muito|isso|esse|aquele|entre|sobre|depois|antes|sempre|nunca|já|ainda|também|onde|quando|quem|qual|porque|portanto|então|assim|pode|deve|ficar|fazer|dizer|saber|haver|ter|ser|estar|ir|vir|pôr|seu|sua|meu|minha|dele|dela|nosso|deste|dessa|naquele|naquela)\b"
    if re.search(palavras_pt, texto, re.IGNORECASE):
        return "pt"
    return "outro"


def _detectar_repeticao(texto: str) -> bool:
    if len(texto) < 50:
        return False
    linhas = texto.strip().split("\n")
    if len(linhas) >= 5:
        linhas_unicas = set(l.strip() for l in linhas if l.strip())
        if len(linhas_unicas) <= 2:
            return True
    chunks = [texto[i:i+30] for i in range(0, len(texto), 30)]
    if len(chunks) >= 5:
        from collections import Counter
        contagem = Counter(chunks)
        mais_comum = contagem.most_common(1)[0]
        if mais_comum[1] >= 3 and len(chunks) >= 6:
            return True
    return False


def _resposta_valida(pergunta: str, resposta: str) -> bool:
    if not resposta or len(resposta.strip()) < 5:
        return False
    if len(pergunta) > 30 and len(resposta.strip()) < 15:
        return False
    if _detectar_repeticao(resposta):
        return False
    if resposta.strip().lower() == pergunta.strip().lower():
        return False
    return True


def process_stream(message, history=None):
    if history is None:
        history = []

    if not hendi_model.loaded:
        yield "⏳ Modelo ainda carregando, aguarde..."
        return

    msg = message.strip()
    yield " "

    if msg.startswith("/"):
        yield handle_command(msg)
        return

    if msg.startswith("!"):
        cmd = msg[1:]
        result = tools.execute_command(cmd)
        output = result.get("stdout") or result.get("stderr") or "(sem saída)"
        if not result.get("success"):
            output = result.get("error", "erro")
        yield f"HendiCode executou:\n```\n{output[:2000]}\n```"
        return

    if msg.startswith("read:"):
        path = msg[5:].strip()
        result = tools.read_file(path)
        if result.get("success"):
            yield f"HendiCode leu {path}:\n```\n{result['content'][:3000]}\n```"
        else:
            yield f"Erro: {result['error']}"
        return

    user_context = hendi_memory.get_context_summary()
    trechos_relevantes = retriever.buscar(msg, top_n=2)

    idioma = _detectar_idioma(msg)
    modo = "code" if re.search(
        r"(?:c[oó]digo|programa|fun[cç][ãa]o|classe|import |def |class |python|java|javascript|"
        r"sql|html|css|docker|git |api |rest|plugin|minecraft|erro|bug|algoritmo|loop|"
        r"heranç[aã]|polimorfismo|requisiç[ãa]o|regex|thread|async|await|banco de dados|"
        r"sql|query|join|teste|pytest|jest|terminal|comando|instalar|pip|npm|vari[aá]vel)",
        msg, re.IGNORECASE,
    ) else "chat"

    # ── Montagem do prompt ─────────────────────────────────────────────
    partes_prompt = [SISTEMA]

    if user_context and user_context != "Nenhuma informação prévia sobre o usuário.":
        partes_prompt.append(f"Contexto do usuário:\n{user_context}")

    exemplos = hendi_memory.get_examples(max_examples=3)
    if exemplos:
        bloco_exemplos = "Exemplos de respostas ideais:"
        for ex in exemplos:
            usuario = ex.get('user') or ex.get('instruction') or ex.get('input', '')
            assistente = ex.get('assistant') or ex.get('response') or ex.get('output', '')
            if usuario and assistente:
                bloco_exemplos += f"\n  Usuário: {usuario}\n  Assistente: {assistente}"
        if bloco_exemplos != "Exemplos de respostas ideais:":
            partes_prompt.append(bloco_exemplos)

    if trechos_relevantes:
        partes = []
        for t in trechos_relevantes:
            partes.append(f"[Fonte: {t['arquivo']}]\n{t['conteudo'][:1000]}")
        partes_prompt.append("Contexto adicional:\n" + "\n\n".join(partes))

    full_prompt = "\n\n".join(partes_prompt) + "\n\n"

    # Histórico limitado
    turnos_usados = 0
    for i in range(len(history) - 1, -1, -2):
        if i - 1 >= 0:
            turnos_usados += 1
            if turnos_usados > MAX_HISTORY_TURNS:
                break
    historio_limitado = history[-MAX_HISTORY_TURNS * 2:] if len(history) > MAX_HISTORY_TURNS * 2 else history

    for i in range(0, len(historio_limitado) - 1, 2):
        if i + 1 < len(historio_limitado):
            user_turn = historio_limitado[i]
            asst_turn = historio_limitado[i + 1]
            if user_turn.get("role") == "user" and asst_turn.get("role") == "assistant":
                full_prompt += (
                    f"### Instruction:\n{user_turn['content']}\n"
                    f"### Response:\n{asst_turn['content']}\n<|EOT|>\n"
                )

    full_prompt += f"### Instruction:\n{msg}\n### Response:\n"

    # ── Geração com retry ──────────────────────────────────────────────
    tentativas = 0
    full_response = ""
    while tentativas < MAX_RETRIES:
        tentativas += 1
        full_response = ""
        for chunk in hendi_model.generate_stream(
            full_prompt,
            max_new_tokens=512,
            top_p=0.9,
            repetition_penalty=1.2,
            modo=modo,
        ):
            full_response += chunk
            yield chunk

        full_response = full_response.strip()

        if _resposta_valida(msg, full_response):
            break

        if tentativas < MAX_RETRIES:
            full_response_regenerada = ""
            for chunk in hendi_model.generate_stream(
                full_prompt,
                max_new_tokens=512,
                top_p=0.95,
                repetition_penalty=1.3,
                modo=modo,
            ):
                full_response_regenerada += chunk

            full_response = full_response_regenerada.strip()

            if _resposta_valida(msg, full_response):
                yield "\n\n" + full_response
                break

            if tentativas == MAX_RETRIES - 1:
                yield "\n\nDesculpe, não consegui gerar uma resposta adequada. Pode reformular a pergunta?"

    # ── Pós-processamento ──────────────────────────────────────────────
    if full_response.strip():
        hendi_memory.save_interaction(msg, full_response)

    msg_lower = msg.lower()
    if re.search(r"meu nome (é|e)|me chamo|sou o|eu sou", msg_lower):
        hendi_memory.save_fact(msg)
    if re.search(r"gosto de|prefiro|quero|não gosto|odeio", msg_lower):
        hendi_memory.save_preference(msg)


def handle_command(cmd):
    cmd_lower = cmd.lower().strip()
    if cmd_lower == "/help":
        return (
            "**HendiCode - Painel de Controle**\n\n"
            "`/help` - Ajuda | `/clear` - Limpa chat\n"
            "`/like` - Marca última resposta como boa\n"
            "`/dislike` - Marca última resposta como ruim\n"
            "**Ferramentas:** `!cmd` (Terminal) | `read: arquivo` | Canvas (Automático para códigos longos)"
        )
    if cmd_lower == "/clear":
        return "__clear__"
    if cmd_lower == "/like" or cmd_lower.startswith("/like "):
        return "__feedback_positive__"
    if cmd_lower == "/dislike" or cmd_lower.startswith("/dislike "):
        return "__feedback_negative__"
    return f"Comando desconhecido: `{cmd}`"
