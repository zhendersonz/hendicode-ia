"""
processar_dados_brutos.py
Tokeniza todos os .txt de dados_brutos/ usando o tokenizer treinado
e gera um único arquivo .bin em dados_processados/.
"""
import os
import glob
from tokenizers import Tokenizer

def processar():
    tokenizer = Tokenizer.from_file("models/hendicode/tokenizer.json")

    arquivos = sorted(glob.glob("dados_brutos/*.txt"))
    print(f"[HendiCode] Processando {len(arquivos)} arquivos...")

    os.makedirs("dados_processados", exist_ok=True)

    total_tokens = 0
    buffer = bytearray()
    caminho_saida = "dados_processados/tudo.bin"

    with open(caminho_saida, "wb") as out:
        for i, arq in enumerate(arquivos):
            nome = os.path.basename(arq)
            try:
                tam = os.path.getsize(arq)
                with open(arq, "r", encoding="utf-8", errors="ignore") as f:
                    if tam > 5_000_000:
                        # Arquivos grandes: processa em chunks
                        for line in f:
                            tokens = tokenizer.encode(line).ids
                            for tid in tokens:
                                buffer.extend(tid.to_bytes(4, "little"))
                            total_tokens += len(tokens)
                    else:
                        texto = f.read()
                        tokens = tokenizer.encode(texto).ids
                        for tid in tokens:
                            buffer.extend(tid.to_bytes(4, "little"))
                        total_tokens += len(tokens)

                if len(buffer) >= 5_000_000:
                    out.write(buffer)
                    buffer.clear()

                if (i + 1) % 30 == 0:
                    pct = (i + 1) / len(arquivos) * 100
                    print(f"  [{i+1}/{len(arquivos)}] {pct:.0f}% | {total_tokens:,} tokens")

            except Exception as e:
                print(f"  [ERRO] {nome}: {e}")

        if buffer:
            out.write(buffer)

    tamanho_mb = os.path.getsize(caminho_saida) / 1_000_000
    print(f"\n[HendiCode] Processamento concluído!")
    print(f"  Total: {total_tokens:,} tokens")
    print(f"  Arquivo: {caminho_saida} ({tamanho_mb:.1f} MB)")

if __name__ == "__main__":
    processar()
