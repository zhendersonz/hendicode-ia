"""
baixar_wikipedia_programacao.py
Baixa artigos completos da Wikipedia PT sobre programação
usando action=raw (sem limite de caracteres).
"""
import os
import urllib.request
import urllib.parse
import time

ARTIGOS_PROGRAMACAO = [
    # Linguagens
    "Python", "JavaScript", "Java (linguagem de programação)",
    "C++", "C Sharp", "PHP", "Ruby", "Rust", "Go (linguagem de programação)",
    "TypeScript", "Kotlin", "Swift", "HTML", "CSS", "SQL",
    "Assembly", "Lua (linguagem de programação)", "R (linguagem de programação)",
    "Perl", "Dart", "Linguagem de programação",
    "Linguagem de script", "Linguagem compilada", "Linguagem interpretada",
    "Paradigma de programação", "Programação orientada a objetos",
    "Programação funcional", "Programação declarativa", "Programação estruturada",
    "Programação lógica",
    # Conceitos
    "Algoritmo", "Estrutura de dados", "Complexidade computacional",
    "Notação Big O", "Recursividade", "Árvore binária",
    "Lista ligada", "Tabela hash", "Pilha (estrutura de dados)", "Fila (estrutura de dados)",
    "Grafo", "Compilador", "Interpretador",
    "Árvore AVL", "Árvore rubro-negra", "Ordenação (computação)",
    # Banco de dados
    "Banco de dados", "Sistema de gerenciamento de banco de dados",
    "Modelo relacional", "MySQL", "PostgreSQL", "SQLite", "MongoDB",
    "NoSQL",
    # Sistemas
    "Sistema operacional", "Kernel", "Thread (computação)", "Processo (informática)",
    "Gerenciamento de memória", "Linux", "Microsoft Windows", "Android",
    "Sistema de arquivos", "Shell", "Chamada de sistema",
    # Redes
    "Rede de computadores", "TCP/IP", "DNS", "HTTP",
    "Criptografia", "Firewall (computação)", "Segurança da informação",
    "Proxy", "Rede privada virtual", "Endereço IP",
    # IA
    "Inteligência artificial", "Aprendizado de máquina", "Aprendizado profundo",
    "Rede neural artificial", "Processamento de linguagem natural",
    "Visão computacional", "Sistema especialista",
    # Engenharia
    "Engenharia de software", "Metodologia ágil",
    "Git", "Teste de software", "DevOps", "Arquitetura de software",
    "Microsserviços", "Computação em nuvem",
    "API", "REST", "WebSocket",
    # Frameworks
    "React (JavaScript)", "Angular", "Vue.js", "Node.js",
    "Django", "Ruby on Rails", "Laravel",
    # Web
    "Navegador web", "Servidor web",
    "Design responsivo",
    # Profissões
    "Programador", "Desenvolvedor web", "Engenheiro de software",
    "Cientista de dados", "Administrador de banco de dados",
    # Gerais
    "Software", "Hardware", "Firmware",
    "Código-fonte", "Depuração", "Controle de versão",
    "Microsoft", "Google", "GitHub",
]


def obter_raw(titulo):
    """Obtém artigo completo via action=raw (sem limite de tamanho)."""
    url = (f"https://pt.wikipedia.org/w/index.php"
           f"?title={urllib.parse.quote(titulo)}"
           f"&action=raw")
    
    req = urllib.request.Request(url, headers={
        "User-Agent": "HendiCode/1.0",
        "Accept-Encoding": "gzip,deflate"
    })
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            # Tenta detectar encoding
            content_type = resp.headers.get("content-type", "")
            if "charset=" in content_type:
                enc = content_type.split("charset=")[-1]
            else:
                enc = "utf-8"
            return raw.decode(enc, errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("  [RATE LIMIT] ", end="", flush=True)
        else:
            print(f"  [HTTP {e.code}] ", end="", flush=True)
        return None
    except Exception as e:
        return None


def limpar_wikitext(texto):
    """Remove marcações wiki mais comuns para deixar texto mais limpo."""
    import re
    # Remove referências
    texto = re.sub(r'<ref[^>]*>.*?</ref>', '', texto, flags=re.DOTALL)
    texto = re.sub(r'<ref[^>]*/>', '', texto)
    # Remove comentários HTML
    texto = re.sub(r'<!--.*?-->', '', texto, flags=re.DOTALL)
    # Remove templates {{...}}
    texto = re.sub(r'\{\{[^}]*\}\}', '', texto)
    # Remove links [[...]]
    texto = re.sub(r'\[\[([^\]|]*)\|([^\]]*)\]\]', r'\2', texto)
    texto = re.sub(r'\[\[([^\]]*)\]\]', r'\1', texto)
    # Remove headings
    texto = re.sub(r'={2,}.*?={2,}', '', texto)
    # Remove tags HTML
    texto = re.sub(r'<[^>]+>', '', texto)
    # Remove múltiplas linhas em branco
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto.strip()


def baixar_wikipedia_programacao():
    print("=" * 60)
    print(f"Baixando {len(ARTIGOS_PROGRAMACAO)} artigos (action=raw, sem limite)")
    print("=" * 60)

    sucessos = []
    erros = 0
    rate_limit_count = 0

    for i, titulo in enumerate(ARTIGOS_PROGRAMACAO):
        # Aguarda 1.5s entre requests para evitar rate limit
        if i > 0:
            time.sleep(1.5)
        
        print(f"[{i+1}/{len(ARTIGOS_PROGRAMACAO)}] {titulo}...", end=" ", flush=True)
        
        raw = obter_raw(titulo)
        if raw and len(raw) > 100:
            limpo = limpar_wikitext(raw)
            if len(limpo) > 100:
                sucessos.append((titulo, limpo))
                print(f"✓ {len(limpo):,} chars")
            else:
                erros += 1
                print(f"✗ (pouco texto: {len(limpo)} chars)")
        elif raw is None:
            rate_limit_count += 1
            if rate_limit_count >= 5:
                print("\n  Muitos rate limits! Aumentando pausa para 5s...")
                rate_limit_count = 0
            erros += 1
            print("✗ (rate limit)")
        else:
            erros += 1
            print(f"✗ (vazio, {len(raw) if raw else 0} chars)")

    # Remove duplicatas
    seen = set()
    unique = []
    for titulo, texto in sucessos:
        if titulo not in seen:
            seen.add(titulo)
            unique.append((titulo, texto))

    texto_final = "\n\n---\n\n".join(
        f"Título: {t}\n\n{texto}" for t, texto in unique
    )

    print(f"\n{'=' * 60}")
    print(f"Resumo: {len(unique)} artigos de {len(ARTIGOS_PROGRAMACAO)}")
    print(f"  Total: {len(texto_final):,} caracteres")
    print(f"  Média: {len(texto_final)//max(len(unique),1):,} chars/artigo")

    return texto_final, unique


def salvar_dados(texto_final, artigos):
    desktop = r"C:\Users\HENDI\Desktop"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dados_brutos = os.path.join(script_dir, "dados_brutos")
    
    for destino in [desktop, dados_brutos]:
        path = os.path.join(destino, "wikipedia_programacao.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(texto_final)
        print(f"  → {path}")
    
    print(f"\nArtigos ({len(artigos)}):")
    for t, _ in sorted(artigos, key=lambda x: -len(x[1]))[:30]:
        print(f"  {t}: {len(_):,} chars")
    print(f"\nTotal: {sum(len(t) for _, t in artigos):,} caracteres")
    print(f"Tokenizer pode ser retreinado com: python treinar_tokenizer.py")


if __name__ == "__main__":
    texto_final, artigos = baixar_wikipedia_programacao()
    salvar_dados(texto_final, artigos)
