"""
processar_dados.py
Baixa datasets públicos e processa em arquivos .bin para treino.
"""
import os
import glob
import math
from tokenizers import Tokenizer

def processar_tudo(tokenizer_path="models/hendicode/tokenizer.json"):
    os.makedirs("dados_processados", exist_ok=True)
    
    print("[HendiCode] Carregando tokenizer...")
    tok = Tokenizer.from_file(tokenizer_path)
    
    def tokenizar_e_salvar(texto, nome_arquivo, max_length=2048):
        """Tokeniza um texto e salva em .bin"""
        tokens = tok.encode(texto).ids
        caminho = f"dados_processados/{nome_arquivo}"
        with open(caminho, "wb") as f:
            for tid in tokens:
                f.write(tid.to_bytes(4, "little"))
        print(f"  → {nome_arquivo}: {len(tokens)} tokens")
        return len(tokens)
    
    def baixar_hf_dataset(nome, split="train", amostra=None):
        """Baixa dataset do HuggingFace"""
        from datasets import load_dataset
        print(f"[HendiCode] Baixando {nome}...")
        ds = load_dataset(nome, split=split, streaming=True)
        if amostra:
            ds = ds.take(amostra)
        return ds
    
    total_tokens = 0
    
    # 1. The Stack (código)
    try:
        ds_stack = baixar_hf_dataset("bigcode/the-stack-smol", amostra=5000)
        textos_code = []
        for i, item in enumerate(ds_stack):
            textos_code.append(item.get("content", ""))
        texto_completo = "\n".join(textos_code)
        total_tokens += tokenizar_e_salvar(texto_completo, "the_stack.bin")
    except Exception as e:
        print(f"  [AVISO] Erro ao baixar The Stack: {e}")
    
    # 2. CodeAlpaca
    try:
        ds_alpaca = baixar_hf_dataset("sahil2801/CodeAlpaca-20K", amostra=2000)
        textos_alpaca = []
        for item in ds_alpaca:
            instrucao = item.get("instruction", "")
            entrada = item.get("input", "")
            saida = item.get("output", "")
            texto = f"USER: {instrucao} {entrada}\nASSISTANT: {saida}\n"
            textos_alpaca.append(texto)
        texto_completo = "\n".join(textos_alpaca)
        total_tokens += tokenizar_e_salvar(texto_completo, "codealpaca.bin")
    except Exception as e:
        print(f"  [AVISO] Erro ao baixar CodeAlpaca: {e}")
    
    # 3. Wikipedia PT
    try:
        ds_wiki = baixar_hf_dataset("wikipedia", "20220301.pt", amostra=2000)
        textos_wiki = []
        for item in ds_wiki:
            textos_wiki.append(item.get("text", ""))
        texto_completo = "\n".join(textos_wiki)
        total_tokens += tokenizar_e_salvar(texto_completo, "wikipedia_pt.bin")
    except Exception as e:
        print(f"  [AVISO] Erro ao baixar Wikipedia: {e}")
    
    # 4. CC100 Portuguese
    try:
        ds_cc100 = baixar_hf_dataset("cc100", lang="pt", amostra=5000)
        textos_cc100 = []
        for item in ds_cc100:
            textos_cc100.append(item.get("text", ""))
        texto_completo = "\n".join(textos_cc100)
        total_tokens += tokenizar_e_salvar(texto_completo, "cc100_pt.bin")
    except Exception as e:
        print(f"  [AVISO] Erro ao baixar CC100: {e}")
    
    print(f"\n[HendiCode] Total processado: {total_tokens} tokens")
    print(f"[HendiCode] Arquivos em dados_processados/:")
    for f in sorted(glob.glob("dados_processados/*.bin")):
        tamanho = os.path.getsize(f) // 4  # cada token é 4 bytes
        print(f"  {f}: {tamanho} tokens")
    
    return total_tokens

if __name__ == "__main__":
    processar_tudo()
