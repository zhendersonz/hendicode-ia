# -*- coding: utf-8 -*-
"""Script final de download - lida com parquet, json.gz, json"""
import sys, os, json, subprocess, time, shutil, glob, gzip

OUTPUT_DIR = r"C:\Users\HENDI\Desktop\dados_5gb"
BRUTOS_DIR = r"C:\Users\HENDI\Desktop\HendiCode\dados_brutos"
for d in [OUTPUT_DIR, BRUTOS_DIR]:
    os.makedirs(d, exist_ok=True)

LOG = os.path.join(OUTPUT_DIR, "log_final.txt")

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
    # Tenta imprimir seguramente
    try:
        print(str(msg))
    except:
        pass

def curl_dl(url, dest, timeout=600):
    for _ in range(3):
        try:
            r = subprocess.run(
                ["curl", "-sL", "-m", str(timeout), "-o", dest, url],
                capture_output=True, timeout=timeout+60
            )
            if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 100:
                return True
        except:
            pass
        time.sleep(5)
    return False

def get_siblings(dataset_id):
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    r = subprocess.run(["curl", "-sL", "-m", "60", url], capture_output=True, timeout=90)
    d = json.loads(r.stdout.decode("utf-8"))
    return d.get("siblings", [])

def extract_parquet(url, nome_saida, colunas, max_mb=500):
    """Baixa um parquet e extrai colunas de texto."""
    temp = os.path.join(OUTPUT_DIR, "_temp.parquet")
    ok = curl_dl(url, temp)
    if not ok:
        return 0
    
    fsize = os.path.getsize(temp)
    if fsize > max_mb * 1024 * 1024:
        log(f"  Arquivo {fsize/1e6:.0f}MB > limite {max_mb}MB, ignorando")
        os.remove(temp)
        return 0
    
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(temp)
        textos = []
        out_path = os.path.join(OUTPUT_DIR, f"{nome_saida}.txt")
        
        for col in colunas:
            if col not in t.column_names:
                continue
            for val in t.column(col).to_pylist():
                if val and isinstance(val, str) and len(val) > 30:
                    textos.append(val)
                    if len(textos) >= 500:
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write("\n\n".join(textos) + "\n")
                        textos = []
        
        if textos:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n\n".join(textos) + "\n")
        
        chars = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        log(f"    OK: +{chars:,} chars, {fsize/1e6:.0f}MB")
        return chars
    except Exception as e:
        log(f"    ERRO: {e}")
        return 0
    finally:
        try:
            os.remove(temp)
        except:
            pass

def extract_gzip(url, nome_saida, colunas):
    """Baixa um .json.gz e extrai colunas."""
    temp = os.path.join(OUTPUT_DIR, "_temp.json.gz")
    ok = curl_dl(url, temp)
    if not ok:
        return 0
    
    try:
        with gzip.open(temp, "rt", encoding="utf-8", errors="replace") as gz:
            content = gz.read(10*1024*1024)  # Le primeiros 10MB
        
        import json as jm
        out_path = os.path.join(OUTPUT_DIR, f"{nome_saida}.txt")
        textos = []
        
        lines = content.strip().split("\n")
        for line in lines:
            try:
                obj = jm.loads(line)
                for col in colunas:
                    val = obj.get(col, "")
                    if val and isinstance(val, str) and len(val) > 30:
                        textos.append(val)
                        if len(textos) >= 500:
                            with open(out_path, "a", encoding="utf-8") as f:
                                f.write("\n\n".join(textos) + "\n")
                            textos = []
            except:
                pass
        
        if textos:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n\n".join(textos) + "\n")
        
        chars = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        log(f"    OK: +{chars:,} chars")
        return chars
    except Exception as e:
        log(f"    ERRO: {e}")
        return 0
    finally:
        try:
            os.remove(temp)
        except:
            pass

def copiar_para_brutos():
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*.txt")):
        if os.path.basename(f).startswith("log"):
            continue
        try:
            shutil.copy2(f, os.path.join(BRUTOS_DIR, os.path.basename(f)))
        except:
            pass

# ==================== TAREFAS ====================

log("INICIO DO DOWNLOAD FINAL")
log("="*60)

total_chars = 0

# 1. CODE_SEARCH_NET - Python (ja temos 600MB do teste)
log("\n[1] code_search_net python (ja baixado)")

# 2. CODE_SEARCH_NET - Java (ja temos 55MB)
log("\n[2] code_search_net java (ja baixado)")

# 3. CODE_SEARCH_NET - Outras linguagens
for lang in ["javascript", "go", "ruby", "php"]:
    log(f"\n[3] code_search_net {lang}")
    sis = get_siblings("code-search-net/code_search_net")
    parqs = [s for s in sis if s["rfilename"].startswith(f"{lang}/") and s["rfilename"].endswith(".parquet")]
    log(f"  {len(parqs)} parquets")
    for arq in parqs[:2]:
        url = f"https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/{arq['rfilename']}"
        c = extract_parquet(url, f"csn_{lang}", ["func_code_string", "func_documentation_string"], 300)
        total_chars += c

# 4. CODE SEARCH NET - All file (test split)
log("\n[4] code_search_net all (test)")
sis = get_siblings("code-search-net/code_search_net")
parqs = [s for s in sis if s["rfilename"].startswith("all/") and s["rfilename"].endswith(".parquet")]
log(f"  {len(parqs)} parquets")
for arq in parqs[:3]:
    url = f"https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/{arq['rfilename']}"
    c = extract_parquet(url, "csn_all", ["func_code_string", "func_documentation_string", "whole_func_string"], 400)
    total_chars += c

# 5. CODEPARROT (.json.gz)
log("\n[5] codeparrot-clean-train")
sis = get_siblings("codeparrot/codeparrot-clean-train")
gz_files = [s for s in sis if s["rfilename"].endswith(".json.gz")]
log(f"  {len(gz_files)} .json.gz files")
for arq in gz_files[:20]:
    url = f"https://huggingface.co/datasets/codeparrot/codeparrot-clean-train/resolve/main/{arq['rfilename']}"
    c = extract_gzip(url, "codeparrot_clean", ["content"])
    total_chars += c

# 6. RSTAR-CODER
log("\n[6] rStar-Coder")
sis = get_siblings("microsoft/rStar-Coder")
parqs = [s for s in sis if s["rfilename"].endswith(".parquet")]
log(f"  {len(parqs)} parquets")
for arq in parqs[:10]:
    url = f"https://huggingface.co/datasets/microsoft/rStar-Coder/resolve/main/{arq['rfilename']}"
    c = extract_parquet(url, "rstar_coder", ["prompt", "response"], 200)
    total_chars += c

# 7. CC100 PORTUGUES
log("\n[7] cc100 lang=pt")
sis = get_siblings("statmt/cc100")
pt_files = [s for s in sis if "lang=pt" in s["rfilename"] and s["rfilename"].endswith(".parquet")]
log(f"  {len(pt_files)} parquets")
for arq in pt_files[:5]:
    url = f"https://huggingface.co/datasets/statmt/cc100/resolve/main/{arq['rfilename']}"
    c = extract_parquet(url, "cc100_portugues", ["text"], 300)
    total_chars += c

# 8. INSTRUCTION DATASETS (CodeAlpaca, Evol-Instruct, Code-Feedback)
log("\n[8] Instruction datasets")
inst_datasets = [
    ("sahil2801/CodeAlpaca-20k", "instruction,output", "codealpaca_20k"),
    ("nickrosh/Evol-Instruct-Code-80k-v1", "instruction,output", "evol_instruct_80k"),
    ("m-a-p/Code-Feedback", "prompt,answer", "code_feedback"),
]
for ds_id, cols, nome in inst_datasets:
    log(f"\n  {ds_id}")
    sis = get_siblings(ds_id)
    
    # Tenta parquet
    parqs = [s for s in sis if s["rfilename"].endswith(".parquet")]
    if parqs:
        for arq in parqs[:2]:
            url = f"https://huggingface.co/datasets/{ds_id}/resolve/main/{arq['rfilename']}"
            c = extract_parquet(url, nome, cols.split(","), 100)
            total_chars += c
        continue
    
    # Tenta json
    jsons = [s for s in sis if s["rfilename"].endswith(".json")]
    if not jsons:
        log(f"  Nenhum arquivo encontrado")
        continue
    
    for arq in jsons[:2]:
        url = f"https://huggingface.co/datasets/{ds_id}/resolve/main/{arq['rfilename']}"
        temp = os.path.join(OUTPUT_DIR, "_temp_inst.json")
        ok = curl_dl(url, temp)
        if not ok:
            continue
        
        try:
            import json as jm
            texts = []
            out_path = os.path.join(OUTPUT_DIR, f"{nome}.txt")
            cols_list = cols.split(",")
            
            with open(temp, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            # Tenta JSONL
            lines = content.strip().split("\n")
            for line in lines[:20000]:
                try:
                    obj = jm.loads(line)
                    for col in cols_list:
                        val = obj.get(col, "")
                        if val and isinstance(val, str) and len(val) > 20:
                            texts.append(val)
                            if len(texts) >= 500:
                                with open(out_path, "a", encoding="utf-8") as f:
                                    f.write("\n\n".join(texts) + "\n")
                                texts = []
                except:
                    pass
            
            if texts:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write("\n\n".join(texts) + "\n")
            
            sz = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            log(f"    OK: +{sz:,} chars")
            total_chars += sz
        except Exception as e:
            log(f"    ERRO: {e}")
        finally:
            try:
                os.remove(temp)
            except:
                pass

# 9. GITHUB-CODE-2025 (1 arquivo, ate 3GB)
log("\n[9] github-code-2025 (1 arquivo grande)")
url = "https://huggingface.co/datasets/nick007x/github-code-2025/resolve/main/above-2-stars/train_000.parquet"
c = extract_parquet(url, "github_code_2025_1", ["content"], 3000)
total_chars += c

# 10. GITHUB-CODE-2025 mais 1 arquivo
log("\n[10] github-code-2025 (arquivo 2)")
url2 = "https://huggingface.co/datasets/nick007x/github-code-2025/resolve/main/above-2-stars/train_001.parquet"
c = extract_parquet(url2, "github_code_2025_2", ["content"], 3000)
total_chars += c

# Copia tudo para dados_brutos
copiar_para_brutos()

# RELATORIO
log("\n\n" + "="*60)
log("RELATORIO FINAL")
log("="*60)
total_mb = 0
for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.txt"))):
    if os.path.basename(f).startswith("log"):
        continue
    sz_mb = os.path.getsize(f) / 1e6
    total_mb += sz_mb
    log(f"  {os.path.basename(f)}: {sz_mb:.2f} MB")

log(f"\nTOTAL: {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")
log("FIM")
