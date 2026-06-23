"""
treinar_tokenizer.py
Treina um tokenizer BPE do zero para o HendiCode original.
Treinado em português + código de programação.
"""
import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC

def treinar_tokenizer(
    arquivos_txt=None,
    vocab_size=32768,
    output_dir="models/hendicode"
):
    if arquivos_txt is None:
        pasta_dados = "dados_brutos"
        if os.path.exists(pasta_dados):
            arquivos_txt = [
                os.path.join(pasta_dados, f)
                for f in os.listdir(pasta_dados)
                if f.endswith(".txt")
            ]
        else:
            arquivos_txt = []

    if not arquivos_txt:
        print("[AVISO] Nenhum arquivo .txt encontrado em dados_brutos/")
        print("[AVISO] O tokenizer será treinado com dados de exemplo mínimos")
        os.makedirs("dados_brutos", exist_ok=True)
        exemplo = """def hello():
    print("Olá, mundo!")

print("Bem-vindo ao HendiCode!")
for i in range(10):
    print(i)
"""
        with open("dados_brutos/exemplo_base.txt", "w", encoding="utf-8") as f:
            f.write(exemplo)
        arquivos_txt = ["dados_brutos/exemplo_base.txt"]

    print(f"[HendiCode] Iniciando treinamento do tokenizer próprio...")
    print(f"[HendiCode] Vocabulário alvo: {vocab_size} tokens")
    print(f"[HendiCode] Arquivos fonte: {arquivos_txt}")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<|pad|>",
            "<|bos|>",
            "<|eos|>",
            "<|unk|>",
        ],
        min_frequency=3,
        show_progress=True,
    )

    def ler_arquivos(arquivos):
        for arq in arquivos:
            with open(arq, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    yield line

    tokenizer.train_from_iterator(ler_arquivos(arquivos_txt), trainer)

    os.makedirs(output_dir, exist_ok=True)

    caminho_tokenizer = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(caminho_tokenizer)
    print(f"[HendiCode] Tokenizer salvo em: {caminho_tokenizer}")

    config = {
        "vocab_size": vocab_size,
        "pad_token": "<|pad|>",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
        "model_max_length": 2048,
        "tokenizer_class": "HendiCodeTokenizer"
    }

    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[HendiCode] Config salva em: {config_path}")

    # Teste básico
    test_text = "def hello():\n    print('Olá, mundo!')\n"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"[HendiCode] Teste de tokenização:")
    print(f"  Original: {test_text!r}")
    print(f"  Tokens: {encoded.ids[:20]}... ({len(encoded.ids)} tokens)")
    print(f"  Decodificado: {decoded!r}")

    print(f"[HendiCode] Tokenizer pronto! Use-o com:")
    print(f"  from tokenizers import Tokenizer")
    print(f"  tok = Tokenizer.from_file('{caminho_tokenizer}')")

    return tokenizer

if __name__ == "__main__":
    treinar_tokenizer()
