"""
orquestrador_mestre.py
Baixa TODOS os datasets em paralelo usando ThreadPoolExecutor
"""
import sys, os, json, subprocess, time, shutil, glob
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = r"C:\Users\HENDI\Desktop\dados_5gb"
BRUTOS_DIR = r"C:\Users\HENDI\Desktop\HendiCode\dados_brutos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BRUTOS_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_DIR, "log_completo.txt")

def log(msg):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")
    print(msg)

def curl_download(url, dest, timeout=600):
    for attempt in range(3):
        try:
            r = subprocess.run(["curl", "-sL", "-m", str(timeout), "-o", dest, url],
                             capture_output=True, timeout=timeout+60)
            if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 100:
                return True
        except:
            pass
        time.sleep(5)
    return False

def baixar_um_parquet(url, rfilename, dataset_id, colunas, saida_nome, max_mb=500):
    """Baixa UM arquivo parquet e extrai texto."""
    log(f"  [>] {rfilename}")
    temp_path = os.path.join(OUTPUT_DIR, f"_temp_{saida_nome}.parquet")
    
    ok = curl_download(url, temp_path)
    if not ok:
        log(f"  [X] Falha download {rfilename}")
        return 0
    
    file_mb = os.path.getsize(temp_path) / 1e6
    log(f"      Baixado: {file_mb:.0f} MB")
    
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(temp_path)
        nrows = len(table)
        
        chars = 0
        rows = 0
        textos = []
        saida_path = os.path.join(OUTPUT_DIR, f"{saida_nome}.txt")
        
        for col in colunas:
            if col not in table.column_names:
                continue
            col_data = table.column(col)
            for val in col_data.to_pylist():
                if val and isinstance(val, str) and len(val) > 30:
                    textos.append(val)
                    chars += len(val)
                    rows += 1
                    
                    if len(textos) >= 500:
                        with open(saida_path, "a", encoding="utf-8") as f:
                            f.write("\n\n".join(textos) + "\n")
                        textos = []
        
        if textos:
            with open(saida_path, "a", encoding="utf-8") as f:
                f.write("\n\n".join(textos) + "\n")
        
        log(f"      Extraido: {chars:,} chars, {rows:,} linhas")
        
        # Copia
        try:
            shutil.copy2(saida_path, os.path.join(BRUTOS_DIR, f"{saida_nome}.txt"))
        except:
            pass
        
        return chars
    
    except Exception as e:
        log(f"  [E] Erro processando {rfilename}: {e}")
        return 0
    finally:
        try:
            os.remove(temp_path)
        except:
            pass


def baixar_dataset_parquets(dataset_id, config_filter, colunas, limite_mb, nome_saida, max_files=20):
    """Lista parquets do dataset e baixa sequencialmente."""
    log(f"\n{'='*60}")
    log(f"Dataset: {dataset_id}")
    
    info_url = f"https://huggingface.co/api/datasets/{dataset_id}"
    r = subprocess.run(["curl", "-sL", "-m", "60", info_url], capture_output=True, timeout=90)
    info = json.loads(r.stdout.decode())
    
    siblings = info.get("siblings", [])
    
    if config_filter:
        parqs = [s for s in siblings if s["rfilename"].startswith(f"{config_filter}/") and s["rfilename"].endswith(".parquet")]
    else:
        parqs = [s for s in siblings if s["rfilename"].endswith(".parquet")]
    
    log(f"  Parquets: {len(parqs)} (max: {max_files})")
    parqs = parqs[:max_files]
    
    total_chars = 0
    limite_bytes = limite_mb * 1024 * 1024
    bytes_baixados = 0
    
    for i, arq in enumerate(parqs):
        if bytes_baixados >= limite_bytes:
            log(f"  Limite {limite_mb}MB atingido. Parando.")
            break
        
        rfilename = arq["rfilename"]
        url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{rfilename}"
        
        # Para datasets multi-file, cria nome único
        nome_part = nome_saida if len(parqs) <= 3 else f"{nome_saida}_part{i}"
        
        chars = baixar_um_parquet(url, rfilename, dataset_id, colunas, nome_part, limite_mb)
        total_chars += chars
        
        # Estima bytes pelo chars (texto ~= 1 byte/char)
        bytes_baixados += chars
    
    log(f"  TOTAL: {total_chars:,} chars para {dataset_id}")
    return total_chars


def processar_cc100():
    """Processa cc100 lang=pt que tem estrutura diferente."""
    log(f"\n{'='*60}")
    log("cc100 lang=pt")
    
    # Lista arquivos
    info_url = "https://huggingface.co/api/datasets/statmt/cc100"
    r = subprocess.run(["curl", "-sL", "-m", "60", info_url], capture_output=True, timeout=90)
    info = json.loads(r.stdout.decode())
    
    # Procura arquivos lang=pt
    pt_files = [s for s in info["siblings"] if "lang=pt" in s["rfilename"] and s["rfilename"].endswith(".parquet")]
    log(f"  Parquets PT: {len(pt_files)}")
    
    total_chars = 0
    for arq in pt_files[:2]:  # Download ~2 arquivos
        rfilename = arq["rfilename"]
        url = f"https://huggingface.co/datasets/statmt/cc100/resolve/main/{rfilename}"
        chars = baixar_um_parquet(url, rfilename, "statmt/cc100", ["text"], "cc100_portugues")
        total_chars += chars
    
    return total_chars


def processar_instrucoes():
    """Processa datasets de instrucoes (json)."""
    log(f"\n{'='*60}")
    log("Datasets de instrucoes")
    total_chars = 0
    
    instrucoes = [
        ("sahil2801/CodeAlpaca-20k", "", "instruction,output", "codealpaca_20k"),
        ("nickrosh/Evol-Instruct-Code-80k-v1", "", "instruction,output", "evol_instruct_80k"),
        ("m-a-p/Code-Feedback", "", "prompt,answer", "code_feedback"),
    ]
    
    for ds_id, cfg, cols, nome in instrucoes:
        info_url = f"https://huggingface.co/api/datasets/{ds_id}"
        r = subprocess.run(["curl", "-sL", "-m", "30", info_url], capture_output=True, timeout=45)
        info = json.loads(r.stdout.decode())
        
        jsons = [s for s in info["siblings"] if s["rfilename"].endswith(".json")]
        if not jsons:
            # Tenta parquet
            parqs = [s for s in info["siblings"] if s["rfilename"].endswith(".parquet")]
            if parqs:
                for arq in parqs[:3]:
                    url = f"https://huggingface.co/datasets/{ds_id}/resolve/main/{arq['rfilename']}"
                    chars = baixar_um_parquet(url, arq['rfilename'], ds_id, cols.split(","), nome)
                    total_chars += chars
            continue
        
        log(f"\n  {ds_id}: {len(jsons)} json files")
        
        temp_path = os.path.join(OUTPUT_DIR, "_temp_instrucoes.json")
        saida_path = os.path.join(OUTPUT_DIR, f"{nome}.txt")
        
        for arq in jsons[:2]:
            url = f"https://huggingface.co/datasets/{ds_id}/resolve/main/{arq['rfilename']}"
            
            ok = curl_download(url, temp_path)
            if not ok:
                continue
            
            try:
                import json as jm
                with open(temp_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                colunas_lista = cols.split(",")
                
                # Tenta JSONL
                lines = content.strip().split("\n")
                textos = []
                for line in lines[:10000]:
                    try:
                        obj = jm.loads(line)
                        for col in colunas_lista:
                            val = obj.get(col, "")
                            if val and isinstance(val, str) and len(val) > 20:
                                textos.append(val)
                                if len(textos) >= 500:
                                    with open(saida_path, "a", encoding="utf-8") as f:
                                        f.write("\n\n".join(textos) + "\n")
                                    textos = []
                    except:
                        pass
                
                if textos:
                    with open(saida_path, "a", encoding="utf-8") as f:
                        f.write("\n\n".join(textos) + "\n")
                
                log(f"    +{len(open(saida_path, 'r', encoding='utf-8').read()):,} chars")
            except Exception as e:
                log(f"    Erro: {e}")
            
            try:
                os.remove(temp_path)
            except:
                pass
        
        try:
            shutil.copy2(saida_path, os.path.join(BRUTOS_DIR, f"{nome}.txt"))
        except:
            pass
        
        if os.path.exists(saida_path):
            total_chars += os.path.getsize(saida_path)
    
    return total_chars


def main():
    inicio = time.time()
    log(f"Inicio: {time.ctime()}")
    log("="*60)
    log("ORQUESTRADOR MESTRE - DOWNLOAD 5GB")
    log("="*60)
    
    # ======= TAREFAS =======
    # Vou usar ThreadPoolExecutor para baixar varios datasets em paralelo
    resultados = {}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futuros = {}
        
        # Agent 1+2: github-code-2025 (2 arquivos, ~5GB cada)
        log("\n--- Enfileirando github-code-2025 ---")
        
        def gh_worker(n):
            return baixar_dataset_parquets(
                "nick007x/github-code-2025", "above-2-stars",
                ["content"], 1000, f"github_code_2025_{n}", max_files=1
            )
        
        futuros[executor.submit(gh_worker, 1)] = "github_code_2025_1"
        futuros[executor.submit(gh_worker, 2)] = "github_code_2025_2"
        
        # Agent 3: codeparrot-clean-train
        futuros[executor.submit(baixar_dataset_parquets,
            "codeparrot/codeparrot-clean-train", "", ["content"], 800, "codeparrot_clean", 5)] = "codeparrot_clean"
        
        # Agent 4: code_search_net python
        futuros[executor.submit(baixar_dataset_parquets,
            "code-search-net/code_search_net", "python", ["func_code_string", "func_documentation_string"], 600, "csn_python", 3)] = "csn_python"
        
        # Agent 5: code_search_net java+js
        futuros[executor.submit(baixar_dataset_parquets,
            "code-search-net/code_search_net", "java", ["func_code_string", "func_documentation_string"], 300, "csn_java", 2)] = "csn_java"
        futuros[executor.submit(baixar_dataset_parquets,
            "code-search-net/code_search_net", "javascript", ["func_code_string", "func_documentation_string"], 300, "csn_javascript", 2)] = "csn_javascript"
        
        # Agent 6: rStar-Coder
        futuros[executor.submit(baixar_dataset_parquets,
            "microsoft/rStar-Coder", "", ["prompt", "response"], 300, "rstar_coder", 5)] = "rstar_coder"
        
        # Agent 7: cc100 portugues
        futuros[executor.submit(processar_cc100)] = "cc100"
        
        # Agent 8: hasankursun language split
        futuros[executor.submit(baixar_dataset_parquets,
            "hasankursun/github-code-2025-language-split", "", ["content"], 400, "github_split", 3)] = "github_split"
        
        # Agent 9: instrucoes
        futuros[executor.submit(processar_instrucoes)] = "instrucoes"
        
        # Agent 10: wikipedia que ja temos + merge
        # (ja baixado, so copia)
        def copiar_wikipedia():
            log("\n--- Copiando Wikipedia existente ---")
            for f in glob.glob(os.path.join(BRUTOS_DIR, "wikipedia_programacao*.txt")):
                try:
                    shutil.copy2(f, OUTPUT_DIR)
                    log(f"  Copiado: {os.path.basename(f)}")
                except:
                    pass
            return 0
        
        futuros[executor.submit(copiar_wikipedia)] = "wikipedia"
        
        # Aguarda todos
        for future in as_completed(futuros):
            nome = futuros[future]
            try:
                chars = future.result()
                resultados[nome] = chars
                log(f"\n✓ {nome}: {chars:,} chars")
            except Exception as e:
                log(f"\n✗ {nome}: ERRO - {e}")
                resultados[nome] = 0
    
    # ======= RELATORIO FINAL =======
    log("\n\n" + "="*60)
    log("RELATORIO FINAL")
    log("="*60)
    
    total_chars = 0
    total_mb = 0
    
    # Lista todos os .txt
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.txt"))):
        if os.path.basename(f) == "log_completo.txt":
            continue
        sz = os.path.getsize(f)
        sz_mb = sz / 1e6
        total_mb += sz_mb
        total_chars += sz
        log(f"  {os.path.basename(f)}: {sz_mb:.2f} MB")
    
    duracao = time.time() - inicio
    log(f"\n{'='*60}")
    log(f"TOTAL: {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")
    log(f"Arquivos: {len(glob.glob(os.path.join(OUTPUT_DIR, '*.txt')))}")
    log(f"Duracao: {duracao/60:.1f} minutos")
    log(f"Fim: {time.ctime()}")
    log("="*60)
    
    return total_mb


if __name__ == "__main__":
    main()
