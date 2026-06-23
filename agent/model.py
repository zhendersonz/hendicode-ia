"""
model.py — Carregador do modelo HendiCode com suporte a LoRA.
"""
import os
import sys
import json
import threading
import torch
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modelo_original import HendiCodeModel

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

STOP_MARKERS = [
    "\n### Instruction:",
    "\n### Response:",
    "### Instruction:",
    "### Response:",
    "<|EOT|>",
    "<｜end▁of▁sentence｜>",
]

PADRONIZACAO_MARCADOR = re.compile(
    r"(?:\n)?(?:### Instruction:|### Response:|<\|EOT\|>)"
)

def _prefixo_de_marcador(texto):
    """Retorna True se texto termina com prefixo parcial de algum marcador."""
    for marcador in STOP_MARKERS:
        for n in range(1, len(marcador) + 1):
            if texto.endswith(marcador[:n]):
                return True
    return False

def _detectar_modo(pergunta: str) -> str:
    """Detecta se a pergunta é sobre código ou conversa geral."""
    padroes_codigo = [
        r"(?:c[oó]digo|programa|fun[cç][ãa]o|classe|m[eé]todo|import|def |class |print|return)",
        r"(?:python|java|javascript|typescript|sql|html|css|c\+\+|rust|golang|php|ruby|c#)",
        r"(?:docker|git |api |rest|json|yaml|xml|http|tcp|udp|flask|django|react|vue)",
        r"(?:plugin|minecraft|spigot|bukkit|mod|forge)",
        r"(?:erro|bug|debug|exception|traceback|stack trace|crash)",
        r"(?:comando terminal|shell|bash|powershell|cmd)",
        r"(?:algoritmo|estrutura de dados|complexidade|big o)",
        r"(?:loop|for |while |if |else |elif|switch|case)",
        r"(?:lista|array|dicionario|tupla|conjunto|set|map|hash)",
        r"(?:heranç[aã]|polimorfismo|encapsulamento|abstraç[ãa]o|poo|oop)",
        r"(?:requisiç[ãa]o|resposta|endpoint|rota|m[eé]todo http)",
        r"(?:regex|express[ãa]o regular|pattern|match)",
        r"(?:thread|processo|assincrono|async|await|concorr[êe]ncia)",
        r"(?:rede|network|socket|conex[ãa]o|cliente|servidor)",
        r"(?:compile|compilei|instalar|instalei|pip|npm|yarn|brew|apt|choco)",
        r"(?:vari[aá]vel|constante|par[âa]metro|argumento|retorno)",
        r"(?:banco de dados|sql|query|join|select|insert|update|delete|where)",
        r"(?:tabela|coluna|linha|registro|schema|índice|index)",
        r"(?:teste|test|unit[aá]rio|integraç[ãa]o|pytest|jest|mocha)",
        r"(?:machine learning|ia|inteligencia artificial|rede neural|deep learning)",
        r"(?:matriz|vetor|tensor|numpy|pandas|matplotlib|scikit)",
    ]
    for padrao in padroes_codigo:
        if re.search(padrao, pergunta, re.IGNORECASE):
            return "code"
    return "chat"


class HendiCodeLoaderOriginal:
    def __init__(self):
        self.model      = None
        self.tokenizer  = None
        self.loaded     = False
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "models", "hendicode")
        self.config     = None
        self._load_lock = threading.Lock()

    def load(self):
        with self._load_lock:
            if self.loaded:
                return True
            try:
                print(f"[HendiCode] Carregando modelo em {self.device}...")

                config_path = os.path.join(self.model_path, "model_config.json")
                if not os.path.exists(config_path):
                    print("[HendiCode] ERRO: model_config.json não encontrado!")
                    return False
                with open(config_path, encoding="utf-8") as f:
                    self.config = json.load(f)
                print(
                    f"[HendiCode] Config: {self.config['num_hidden_layers']} camadas, "
                    f"{self.config['hidden_size']} hidden, {self.config['vocab_size']} vocab"
                )

                from tokenizers import Tokenizer
                tokenizer_path = os.path.join(self.model_path, "tokenizer.json")
                if not os.path.exists(tokenizer_path):
                    print(f"[HendiCode] Tokenizer não encontrado em {tokenizer_path}")
                    return False
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
                print(f"[HendiCode] Tokenizer: {self.tokenizer.get_vocab_size()} tokens")

                self.model = HendiCodeModel(self.config)
                n_params = sum(p.numel() for p in self.model.parameters())
                print(f"[HendiCode] Arquitetura: {n_params:,} parâmetros")

                pesos_path = os.path.join(self.model_path, "pesos_finais.pt")
                if not os.path.exists(pesos_path):
                    print("[HendiCode] ERRO: pesos_finais.pt não encontrado.")
                    return False
                state_dict = torch.load(pesos_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state_dict, strict=False)
                carregados = sum(v.numel() for v in state_dict.values())
                print(f"[HendiCode] Pesos: {carregados:,} parâmetros carregados")

                lora_path = os.path.join(self.model_path, "lora")
                if os.path.isdir(lora_path) and PEFT_AVAILABLE:
                    try:
                        print(f"[HendiCode] Aplicando adaptadores LoRA de: {lora_path}")
                        self.model = PeftModel.from_pretrained(self.model, lora_path)
                        print(f"[HendiCode] LoRA aplicado com sucesso!")
                    except Exception as e:
                        print(f"[HendiCode] Aviso: falha ao carregar LoRA ({e}). Usando modelo base.")
                elif os.path.isdir(lora_path) and not PEFT_AVAILABLE:
                    print("[HendiCode] Aviso: adaptadores LoRA encontrados, mas PEFT não está instalado.")

                self.model = self.model.to(self.device)
                self.model.eval()

                if self.device == "cpu":
                    torch.set_num_threads(os.cpu_count())
                    print(f"[HendiCode] Threads CPU: {os.cpu_count()}")

                self.loaded = True
                print("[HendiCode] Modelo pronto!")
                return True

            except Exception as e:
                import traceback
                print(f"[HendiCode] Erro ao carregar: {e}")
                traceback.print_exc()
                return False

    def generate_stream(
        self,
        prompt,
        max_new_tokens=512,
        temperature=None,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.15,
        modo=None,
    ):
        if not self.loaded:
            if not self.load():
                yield "[HendiCode] Erro: modelo não carregado."
                return

        try:
            # ── Detecção dinâmica de temperatura ──
            if temperature is None:
                if modo is None:
                    modo = _detectar_modo(prompt)
                temperature = 0.2 if modo == "code" else 0.7

            encoded  = self.tokenizer.encode(prompt)
            bos_id   = self.config.get("bos_token_id", 32013)
            stop_ids = {
                self.config.get("pad_token_id", 32014),
                self.config.get("eos_token_id", 32021),
            }

            input_ids = torch.tensor(
                [[bos_id] + encoded.ids],
                dtype=torch.long,
                device=self.device,
            )

            id_buffer    = []
            texto_total  = ""
            buffer_seg   = ""

            for token_tensor in self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_token_ids=stop_ids,
            ):
                token_id = token_tensor.item() if hasattr(token_tensor, "item") else int(token_tensor)
                if token_id in stop_ids:
                    break

                id_buffer.append(token_id)

                novo_total = self.tokenizer.decode(id_buffer, skip_special_tokens=True)
                incremento = novo_total[len(texto_total):]
                texto_total = novo_total

                if not incremento:
                    continue

                buffer_seg += incremento

                marcador_achado = None
                for marcador in STOP_MARKERS:
                    if marcador in buffer_seg:
                        marcador_achado = marcador
                        break

                if marcador_achado:
                    ate = buffer_seg.split(marcador_achado)[0]
                    if ate:
                        yield ate
                    return

                if _prefixo_de_marcador(buffer_seg):
                    continue

                yield buffer_seg
                buffer_seg = ""

            if buffer_seg:
                marcador_achado = None
                for marcador in STOP_MARKERS:
                    if marcador in buffer_seg:
                        marcador_achado = marcador
                        break
                if marcador_achado:
                    ate = buffer_seg.split(marcador_achado)[0]
                    if ate:
                        yield ate
                else:
                    yield buffer_seg

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"[HendiCode] Erro na geração: {e}"


hendi_model = HendiCodeLoaderOriginal()
