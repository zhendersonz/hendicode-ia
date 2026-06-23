"""
baixar_wikipedia_completo.py
Baixa TODOS os artigos de programação da Wikipedia PT
via categorias recursivas + action=raw.
Salva em dados_wikipedia_programacao/ no Desktop.
"""
import os
import subprocess
import json
import re
import time

OUTPUT_DIR = r"C:\Users\HENDI\Desktop\dados_wikipedia_programacao"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USER_AGENT = "HendiCode/1.0"

def api_get(params):
    """Chama Wikipedia API via curl."""
    url = "https://pt.wikipedia.org/w/api.php?" + params
    cmd = ["curl", "-s", "-m", "30", "-H", f"User-Agent: {USER_AGENT}", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
    except:
        pass
    return None


def get_raw(titulo):
    """Obtém artigo completo via action=raw."""
    import urllib.parse
    url = f"https://pt.wikipedia.org/w/index.php?title={urllib.parse.quote(titulo)}&action=raw"
    cmd = ["curl", "-s", "-m", "30", "-H", f"User-Agent: {USER_AGENT}", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and len(r.stdout) > 100:
            return r.stdout
    except:
        pass
    return None


def limpar_wikitext(texto):
    """Remove marcações wiki."""
    texto = re.sub(r'<ref[^>]*>.*?</ref>', '', texto, flags=re.DOTALL)
    texto = re.sub(r'<ref[^>]*/>', '', texto)
    texto = re.sub(r'<!--.*?-->', '', texto, flags=re.DOTALL)
    texto = re.sub(r'\{\{[^}]*\}\}', '', texto)
    texto = re.sub(r'\[\[([^\]|]*)\|([^\]]*)\]\]', r'\2', texto)
    texto = re.sub(r'\[\[([^\]]*)\]\]', r'\1', texto)
    texto = re.sub(r"''+", '', texto)
    texto = re.sub(r'={2,}.*?={2,}', '', texto)
    texto = re.sub(r'<[^>]+>', '', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto.strip()


def get_subcats(categoria, visitadas):
    """Retorna subcategorias recursivamente."""
    subs = []
    cmcontinue = None
    while True:
        params = (
            f"action=query&list=categorymembers"
            f"&cmtitle=Categoria:{categoria}"
            f"&cmtype=subcat&cmlimit=max"
            f"&format=json&utf8=1"
        )
        if cmcontinue:
            params += f"&cmcontinue={cmcontinue}"
        
        dados = api_get(params)
        if not dados:
            break
        
        for m in dados.get("query", {}).get("categorymembers", []):
            nome = m["title"].replace("Categoria:", "")
            if nome not in visitadas:
                visitadas.add(nome)
                subs.append(nome)
        
        cont = dados.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue:
            break
    
    return subs


def get_pages(categoria):
    """Retorna todas as páginas de uma categoria."""
    pages = []
    cmcontinue = None
    while True:
        params = (
            f"action=query&list=categorymembers"
            f"&cmtitle=Categoria:{categoria}"
            f"&cmtype=page&cmlimit=max"
            f"&format=json&utf8=1"
        )
        if cmcontinue:
            params += f"&cmcontinue={cmcontinue}"
        
        dados = api_get(params)
        if not dados:
            break
        
        for m in dados.get("query", {}).get("categorymembers", []):
            pages.append(m["title"])
        
        cont = dados.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue:
            break
    
    return pages


def baixar_tudo():
    print("=" * 60)
    print("Wikipedia PT - Download completo de programação")
    print(f"Salvando em: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Categorias raiz
    RAIZ = [
        "Programação", "Ciência da computação", "Engenharia de software",
        "Computação", "Informática", "Redes de computadores",
        "Banco de dados", "Segurança computacional",
        "Inteligência artificial", "Sistemas operacionais",
        "Internet", "Software", "Hardware",
        "Criptografia", "Linguagens de programação",
        "Algoritmos", "Estruturas de dados",
        "Compiladores", "Teoria da computação",
        "Aprendizado de máquina", "Visão computacional",
        "Robótica", "Telecomunicações",
        "Matemática computacional", "Multimédia",
    ]
    
    # Fase 1: Coletar TODAS as categorias recursivamente
    print("\n[Fase 1] Coletando categorias...")
    todas_cats = set()
    para_processar = list(RAIZ)
    
    while para_processar:
        cat = para_processar.pop(0)
        if cat in todas_cats:
            continue
        todas_cats.add(cat)
        print(f"  Categoria: {cat}")
        
        subs = get_subcats(cat, set())
        novas = [s for s in subs if s not in todas_cats]
        para_processar.extend(novas)
        
        if subs:
            print(f"    → {len(subs)} subcategorias encontradas")
        
        time.sleep(0.3)  # rate limit
    
    print(f"\n  Total de categorias: {len(todas_cats)}")
    
    # Salva lista de categorias
    with open(os.path.join(OUTPUT_DIR, "categorias.txt"), "w", encoding="utf-8") as f:
        for c in sorted(todas_cats):
            f.write(c + "\n")
    
    # Fase 2: Coletar páginas de cada categoria
    print(f"\n[Fase 2] Coletando páginas das categorias...")
    todas_paginas = set()
    
    for i, cat in enumerate(sorted(todas_cats)):
        pages = get_pages(cat)
        novas = [p for p in pages if p not in todas_paginas]
        todas_paginas.update(novas)
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(todas_cats)}] {len(todas_paginas)} páginas únicas até agora")
        
        time.sleep(0.2)
    
    print(f"\n  Total de páginas únicas: {len(todas_paginas)}")
    
    with open(os.path.join(OUTPUT_DIR, "paginas.txt"), "w", encoding="utf-8") as f:
        for p in sorted(todas_paginas):
            f.write(p + "\n")
    
    # Fase 3: Baixar conteúdo de cada página
    print(f"\n[Fase 3] Baixando conteúdo ({len(todas_paginas)} páginas)...")
    
    paginas_baixadas = 0
    paginas_erro = 0
    todas_ordenadas = sorted(todas_paginas)
    
    for i, titulo in enumerate(todas_ordenadas):
        print(f"  [{i+1}/{len(todas_ordenadas)}] {titulo}...", end=" ", flush=True)
        
        raw = get_raw(titulo)
        if raw and len(raw) > 100:
            limpo = limpar_wikitext(raw)
            if len(limpo) > 100:
                # Salva individualmente
                nome_arquivo = re.sub(r'[\\/*?:"<>|]', '_', titulo) + ".txt"
                caminho = os.path.join(OUTPUT_DIR, nome_arquivo)
                with open(caminho, "w", encoding="utf-8") as f:
                    f.write(f"Título: {titulo}\n\n{limpo}")
                paginas_baixadas += 1
                print(f"✓ {len(limpo):,} chars")
            else:
                paginas_erro += 1
                print(f"✗ (pouco texto)")
        else:
            paginas_erro += 1
            print(f"✗ (vazio)")
        
        # Rate limit
        time.sleep(0.3)
    
    # Fase 4: Juntar tudo em um arquivo único
    print(f"\n[Fase 4] Juntando em arquivo único...")
    todos_textos = []
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.endswith(".txt") and fname not in ("categorias.txt", "paginas.txt", "wikipedia_programacao_completo.txt", "relatorio.txt"):
            path = os.path.join(OUTPUT_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    todos_textos.append(f.read().strip())
            except:
                pass
    
    completo = "\n\n---\n\n".join(todos_textos)
    caminho_completo = os.path.join(OUTPUT_DIR, "wikipedia_programacao_completo.txt")
    with open(caminho_completo, "w", encoding="utf-8") as f:
        f.write(completo)
    
    total_chars = len(completo)
    
    # Relatório final
    print(f"\n{'=' * 60}")
    print("RESUMO FINAL")
    print(f"{'=' * 60}")
    print(f"Categorias: {len(todas_cats)}")
    print(f"Páginas únicas: {len(todas_paginas)}")
    print(f"Baixadas com sucesso: {paginas_baixadas}")
    print(f"Erros/vazios: {paginas_erro}")
    print(f"Total de caracteres: {total_chars:,}")
    print(f"Arquivo único: {caminho_completo}")
    print(f"Pasta: {OUTPUT_DIR}")
    
    relatorio = f"""RELATÓRIO - Wikipedia Programação Completo
====================================
Categorias processadas: {len(todas_cats)}
Páginas únicas: {len(todas_paginas)}
Baixadas com sucesso: {paginas_baixadas}
Erros: {paginas_erro}
Total caracteres: {total_chars:,}

Para usar no HendiCode:
1. Copie o arquivo para dados_brutos/:
   copy "{caminho_completo}" "C:\\Users\\HENDI\\Desktop\\HendiCode\\dados_brutos\\"

2. Retreine o tokenizer:
   python treinar_tokenizer.py

3. Processe os dados:
   python processar_dados.py
"""
    with open(os.path.join(OUTPUT_DIR, "relatorio.txt"), "w", encoding="utf-8") as f:
        f.write(relatorio)
    print("\n" + relatorio)


if __name__ == "__main__":
    baixar_tudo()
