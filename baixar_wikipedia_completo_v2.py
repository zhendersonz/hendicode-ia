"""
baixar_wikipedia_completo_v2.py
Versao corrigida - usa bytes e decode UTF-8 explicito
para evitar corrompimento cp1252.
"""
import os, subprocess, json, re, time, urllib.parse

OUTPUT_DIR = r"C:\Users\HENDI\Desktop\dados_wikipedia_programacao"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USER_AGENT = "HendiCode/1.0"

def api_get(params):
    """Chama API via curl, decodifica como UTF-8."""
    url = "https://pt.wikipedia.org/w/api.php?" + params
    cmd = ["curl", "-s", "-m", "30", "-H", f"User-Agent: {USER_AGENT}", url]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        text = r.stdout.decode("utf-8", errors="replace")
        if text.strip():
            return json.loads(text)
    except:
        pass
    return None

def get_raw(titulo):
    """Obtem artigo via action=raw."""
    url = f"https://pt.wikipedia.org/w/index.php?title={urllib.parse.quote(titulo)}&action=raw"
    cmd = ["curl", "-s", "-m", "30", "-H", f"User-Agent: {USER_AGENT}", url]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        text = r.stdout.decode("utf-8", errors="replace")
        if len(text) > 100:
            return text
    except:
        pass
    return None

def limpar(texto):
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

def get_subcats(categoria):
    """Retorna subcategorias."""
    subs = []
    cont = None
    while True:
        p = f"action=query&list=categorymembers&cmtitle={urllib.parse.quote('Categoria:'+categoria)}&cmtype=subcat&cmlimit=max&format=json&utf8=1"
        if cont: p += f"&cmcontinue={cont}"
        d = api_get(p)
        if not d: break
        for m in d.get("query",{}).get("categorymembers",[]):
            subs.append(m["title"].replace("Categoria:",""))
        cont = (d.get("continue") or {}).get("cmcontinue")
        if not cont: break
    return subs

def get_pages(categoria):
    """Retorna paginas de uma categoria."""
    pages = []
    cont = None
    while True:
        p = f"action=query&list=categorymembers&cmtitle={urllib.parse.quote('Categoria:'+categoria)}&cmtype=page&cmlimit=max&format=json&utf8=1"
        if cont: p += f"&cmcontinue={cont}"
        d = api_get(p)
        if not d: break
        for m in d.get("query",{}).get("categorymembers",[]):
            pages.append(m["title"])
        cont = (d.get("continue") or {}).get("cmcontinue")
        if not cont: break
    return pages

print("="*60)
print("Wikipedia PT - Download completo (v2 - encoding fix)")
print("="*60)

ROOT_CATS = [
    "Programação", "Ciência da computação", "Engenharia de software",
    "Computação", "Informática", "Redes de computadores",
    "Banco de dados", "Segurança computacional",
    "Inteligência artificial", "Sistemas operacionais",
    "Internet", "Software", "Hardware",
    "Criptografia", "Linguagens de programação",
    "Algoritmos", "Estruturas de dados", "Compiladores",
    "Teoria da computação", "Aprendizado de máquina",
    "Visão computacional", "Robótica",
    "Telecomunicações", "Matemática computacional",
]

# Fase 1: Categorias
print("\n[1/3] Coletando categorias...")
todas = set()
fila = list(ROOT_CATS)
while fila:
    c = fila.pop(0)
    if c in todas: continue
    todas.add(c)
    subs = get_subcats(c)
    novas = [s for s in subs if s not in todas]
    fila.extend(novas)
    print(f"  {c}: {len(subs)} subs", flush=True)
    time.sleep(0.2)

print(f"\n  Total: {len(todas)} categorias")

# Fase 2: Paginas
print("\n[2/3] Coletando paginas...")
todas_pags = set()
for i, c in enumerate(sorted(todas)):
    pags = get_pages(c)
    novas = [p for p in pags if p not in todas_pags]
    # Filtra paginas nao-uteis
    novas = [p for p in novas if not p.startswith("!") and not p.startswith("Predefinição") and not p.startswith("Wikipédia") and not p.startswith("Anexo:") and not p.startswith("Usuário") and not p.startswith("Ficheiro") and not p.startswith("Categoria")]
    todas_pags.update(novas)
    if (i+1)%20==0:
        print(f"  [{i+1}/{len(todas)}] {len(todas_pags)} paginas", flush=True)
    time.sleep(0.15)

print(f"\n  Total: {len(todas_pags)} paginas")

# Fase 3: Download
print("\n[3/3] Baixando conteudo...")
baixadas = 0
erros = 0
for i, tit in enumerate(sorted(todas_pags)):
    print(f"  [{i+1}/{len(todas_pags)}] {tit[:50]}...", end=" ", flush=True)
    raw = get_raw(tit)
    if raw and len(raw) > 100:
        limpo = limpar(raw)
        if len(limpo) > 100:
            fname = re.sub(r'[\\/*?:"<>|]', '_', tit) + ".txt"
            with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
                f.write(f"Título: {tit}\n\n{limpo}")
            baixadas += 1
            print(f"OK {len(limpo)}c")
        else:
            erros += 1
            print("curto")
    else:
        erros += 1
        print("vazio")
    time.sleep(0.3)

# Juntar tudo
print("\nJuntando arquivos...")
todos_txt = []
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith(".txt") and f not in ("categorias.txt","paginas.txt","wikipedia_programacao_completo.txt","relatorio.txt"):
        try:
            with open(os.path.join(OUTPUT_DIR, f), encoding="utf-8") as fp:
                todos_txt.append(fp.read().strip())
        except: pass

completo = "\n\n===\n\n".join(todos_txt)
with open(os.path.join(OUTPUT_DIR, "wikipedia_programacao_completo.txt"), "w", encoding="utf-8") as f:
    f.write(completo)

print(f"\n{'='*60}")
print(f"Categorias: {len(todas)}")
print(f"Paginas: {len(todas_pags)}")
print(f"Baixadas: {baixadas}, Erros: {erros}")
print(f"Total chars: {len(completo):,}")
print(f"Pasta: {OUTPUT_DIR}")
print(f"Arquivo: wikipedia_programacao_completo.txt")
print(f"{'='*60}")
