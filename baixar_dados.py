"""
baixar_dados.py
Baixa datasets reais para treino do HendiCode e salva como .txt.
"""
import os
import json
import urllib.request
import gzip
import io

def baixar_wikipedia_pt(limite=2000):
    """Baixa artigos da Wikipedia PT via API (sem depender do datasets)."""
    print("[1/3] Baixando Wikipedia PT via API...")
    textos = []
    url_base = "https://pt.wikipedia.org/w/api.php"
    params = "?action=query&list=random&rnnamespace=0&rnlimit={}&format=json&utf8=1"
    
    tentativas = 0
    while len(textos) < limite and tentativas < limite * 2:
        try:
            req = urllib.request.Request(
                url_base + params.format(min(50, limite - len(textos))),
                headers={"User-Agent": "HendiCode/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                dados = json.loads(resp.read().decode("utf-8"))
            
            pages = [p["id"] for p in dados["query"]["random"]]
            
            # Busca o conteúdo de cada página
            content_url = url_base + "?action=query&pageids={}&prop=extracts&exintro&explaintext&format=json&utf8=1".format("|".join(map(str, pages)))
            req2 = urllib.request.Request(content_url, headers={"User-Agent": "HendiCode/1.0"})
            with urllib.request.urlopen(req2, timeout=30) as resp2:
                dados2 = json.loads(resp2.read().decode("utf-8"))
            
            for pid in pages:
                page = dados2["query"]["pages"].get(str(pid), {})
                extract = page.get("extract", "").strip()
                title = page.get("title", "")
                if extract and len(extract) > 100:
                    textos.append(f"Título: {title}\n\n{extract}")
                    if len(textos) % 100 == 0:
                        print(f"  Wikipedia: {len(textos)}/{limite} artigos")
            
            tentativas += 1
        except Exception as e:
            tentativas += 1
            continue
    
    print(f"  Wikipedia: {len(textos)} artigos baixados")
    return "\n\n---\n\n".join(textos[:limite])


def baixar_codealpaca(limite=5000):
    """Baixa CodeAlpaca-20K diretamente do Hugging Face."""
    print("[2/3] Baixando CodeAlpaca-20K...")
    from datasets import load_dataset
    ds = load_dataset("sahil2801/CodeAlpaca-20K", split="train", streaming=True)
    textos = []
    for i, item in enumerate(ds):
        if i >= limite:
            break
        texto = f"USER: {item.get('instruction', '')}\n{item.get('input', '')}\nASSISTANT: {item.get('output', '')}\n"
        textos.append(texto)
        if (i + 1) % 1000 == 0:
            print(f"  CodeAlpaca: {i+1}/{limite}")
    print(f"  CodeAlpaca: {len(textos)} exemplos")
    return "\n".join(textos)


def baixar_python_code(limite=2000):
    """Baixa código Python público do GitHub via API."""
    print("[3/3] Baixando código Python público...")
    from datasets import load_dataset
    
    # Usa dataset público de código (não gated)
    try:
        ds = load_dataset("bigcode/the-stack-dedup", split="train", streaming=True)
        textos = []
        linguas = {}
        for i, item in enumerate(ds):
            if i >= limite:
                break
            lang = item.get("lang", "unknown")
            linguas[lang] = linguas.get(lang, 0) + 1
            textos.append(f"// Language: {lang}\n{item.get('content', '')}")
            if (i + 1) % 500 == 0:
                print(f"  The Stack: {i+1}/{limite} ({len(linguas)} linguas)")
        print(f"  The Stack: {len(textos)} arquivos, linguas: {list(linguas.keys())[:10]}...")
        return "\n\n".join(textos)
    except Exception:
        print("  The Stack-dedup indisponível, tentando alternativa...")
    
    # Fallback: usa dataset público de Python
    try:
        ds = load_dataset("transformersbook/codeparrot-train", split="train", streaming=True)
        textos = []
        for i, item in enumerate(ds):
            if i >= limite:
                break
            textos.append(f"// Python code\n{item.get('content', '')}")
            if (i + 1) % 500 == 0:
                print(f"  CodeParrot: {i+1}/{limite}")
        print(f"  CodeParrot: {len(textos)} arquivos")
        return "\n\n".join(textos)
    except Exception as e:
        print(f"  [ERRO] Fallback: {e}")
    
    return ""


def baixar_tudo():
    desktop = r"C:\Users\HENDI\Desktop"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dados_brutos = os.path.join(script_dir, "dados_brutos")
    os.makedirs(dados_brutos, exist_ok=True)

    destinos = [desktop, dados_brutos]
    stats = []
    erros = []

    # 1. Wikipedia PT
    print("=" * 60)
    try:
        texto = baixar_wikipedia_pt(2000)
        if texto:
            for d in destinos:
                path = os.path.join(d, "wikipedia_pt.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(texto)
                print(f"  → {path}")
            stats.append(("Wikipedia PT", len(texto)))
    except Exception as e:
        erros.append(f"Wikipedia: {e}")

    # 2. CodeAlpaca
    print("=" * 60)
    try:
        texto = baixar_codealpaca(5000)
        if texto:
            for d in destinos:
                path = os.path.join(d, "codealpaca.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(texto)
                print(f"  → {path}")
            stats.append(("CodeAlpaca", len(texto)))
    except Exception as e:
        erros.append(f"CodeAlpaca: {e}")

    # 3. Código (Python)
    print("=" * 60)
    try:
        texto = baixar_python_code(2000)
        if texto:
            for d in destinos:
                path = os.path.join(d, "python_code.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(texto)
                print(f"  → {path}")
            stats.append(("Python Code", len(texto)))
    except Exception as e:
        erros.append(f"Python Code: {e}")

    # Resumo
    print("=" * 60)
    print("RESUMO:")
    print(f"{'Dataset':<20} {'Caracteres':>15}")
    print("-" * 35)
    for nome, chars in stats:
        print(f"{nome:<20} {chars:>15,}")
    total = sum(s[1] for s in stats)
    print("-" * 35)
    print(f"{'TOTAL':<20} {total:>15,}")
    
    if erros:
        print(f"\nERROS ({len(erros)}):")
        for e in erros:
            print(f"  ⚠ {e}")
    
    print(f"\nArquivos salvos em:")
    print(f"  {desktop}")
    print(f"  {dados_brutos}")

if __name__ == "__main__":
    baixar_tudo()
