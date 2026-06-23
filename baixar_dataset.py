"""
baixar_dataset.py
Download universal de datasets HF para .txt
Uso: python baixar_dataset.py <dataset_id> [config] [colunas] [limite_mb] [nome_saida]
"""
import sys, os, json, subprocess, re, time

# Config
OUTPUT_DIR = r"C:\Users\HENDI\Desktop\dados_5gb"
os.makedirs(OUTPUT_DIR, exist_ok=True)
USER_AGENT = "HendiCode/1.0"
MAX_RETRIES = 3

def curl_get(url):
    for attempt in range(MAX_RETRIES):
        try:
            r = subprocess.run(
                ["curl", "-sL", "-m", "60", "-H", f"User-Agent: {USER_AGENT}", url],
                capture_output=True, timeout=120
            )
            out = r.stdout.decode("utf-8", errors="replace")
            if out.strip():
                return out
        except:
            pass
        time.sleep(2)
    return None

def baixar_parquet(url, destino_temp):
    """Baixa arquivo parquet com retry."""
    for attempt in range(MAX_RETRIES):
        try:
            r = subprocess.run(
                ["curl", "-sL", "-m", "120", "-H", f"User-Agent: {USER_AGENT}", "-o", destino_temp, url],
                capture_output=True, timeout=180
            )
            if r.returncode == 0 and os.path.exists(destino_temp) and os.path.getsize(destino_temp) > 100:
                return True
        except:
            pass
        time.sleep(3)
    return False

def baixar_json(url, destino_temp):
    """Baixa arquivo json com retry."""
    for attempt in range(MAX_RETRIES):
        try:
            r = subprocess.run(
                ["curl", "-sL", "-m", "120", "-H", f"User-Agent: {USER_AGENT}", "-o", destino_temp, url],
                capture_output=True, timeout=180
            )
            if r.returncode == 0 and os.path.exists(destino_temp) and os.path.getsize(destino_temp) > 100:
                return True
        except:
            pass
        time.sleep(3)
    return False

def processar_dataset(dataset_id, config_name, colunas_texto, limite_mb=500, nome_saida=None):
    """Baixa e processa um dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_id}")
    if config_name:
        print(f"Config: {config_name}")
    print(f"Colunas: {colunas_texto}")
    print(f"Limite: {limite_mb} MB")
    print(f"{'='*60}")
    
    if not nome_saida:
        nome_saida = dataset_id.replace("/", "_").replace("-", "_") + (f"_{config_name}" if config_name else "")
    
    # Pega info do dataset
    info_url = f"https://huggingface.co/api/datasets/{dataset_id}"
    info_json = curl_get(info_url)
    if not info_json:
        print(f"[ERRO] Não foi possível obter info do dataset {dataset_id}")
        return 0
    
    info = json.loads(info_json)
    siblings = info.get("siblings", [])
    
    # Filtra arquivos do config específico
    if config_name:
        arquivos = [s for s in siblings if s["rfilename"].startswith(f"{config_name}/") and s["rfilename"].endswith(".parquet")]
    else:
        arquivos = [s for s in siblings if s["rfilename"].endswith(".parquet") and not s["rfilename"].startswith(".git")]
    
    # Se não achou parquet, tenta json
    if not arquivos:
        if config_name:
            arquivos = [s for s in siblings if s["rfilename"].startswith(f"{config_name}/") and s["rfilename"].endswith(".json")]
        else:
            arquivos = [s for s in siblings if s["rfilename"].endswith(".json") and not s["rfilename"].startswith(".git")]
        formato = "json"
    else:
        formato = "parquet"
    
    if not arquivos:
        print(f"[ERRO] Nenhum arquivo encontrado para {dataset_id}/{config_name}")
        return 0
    
    print(f"Arquivos encontrados: {len(arquivos)} ({formato})")
    
    total_bytes = 0
    total_chars = 0
    total_linhas = 0
    limite_bytes = limite_mb * 1024 * 1024
    
    textos = []
    
    for i, arq in enumerate(arquivos):
        if total_bytes >= limite_bytes:
            print(f"  Limite de {limite_mb}MB atingido. Parando.")
            break
        
        rfilename = arq["rfilename"]
        url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{rfilename}"
        temp_path = os.path.join(OUTPUT_DIR, f"_{nome_saida}_temp_{i}.{formato}")
        
        print(f"  [{i+1}/{len(arquivos)}] {rfilename}", end=" ", flush=True)
        
        # Download
        if formato == "parquet":
            ok = baixar_parquet(url, temp_path)
        else:
            ok = baixar_json(url, temp_path)
        
        if not ok:
            print(f"[FALHA]")
            continue
        
        file_size = os.path.getsize(temp_path)
        print(f"({file_size/1e6:.1f}MB)", end=" ", flush=True)
        
        try:
            if formato == "parquet":
                import pyarrow.parquet as pq
                import pyarrow as pa
                table = pq.read_table(temp_path)
                linhas = len(table)
                
                for col in colunas_texto:
                    if col in table.column_names:
                        textos_col = table.column(col).to_pylist()
                        for val in textos_col:
                            if val and isinstance(val, str) and len(val) > 20:
                                textos.append(val)
                                total_chars += len(val)
                                total_linhas += 1
            else:
                # JSON - lê linha por linha
                import json as jmod
                with open(temp_path, "r", encoding="utf-8") as f:
                    conteudo = f.read()
                
                # Tenta como JSONL (uma linha por objeto)
                linhas_raw = conteudo.strip().split("\n")
                for linha in linhas_raw[:50000]:  # Limite por segurança
                    try:
                        obj = jmod.loads(linha)
                        for col in colunas_texto:
                            val = obj.get(col, "")
                            if val and isinstance(val, str) and len(val) > 20:
                                textos.append(val)
                                total_chars += len(val)
                                total_linhas += 1
                    except:
                        pass
            
            total_bytes += file_size
            print(f"✓ (+{total_chars:,} chars)")
            
            # Salva incrementalmente
            if len(textos) >= 1000:
                with open(os.path.join(OUTPUT_DIR, f"{nome_saida}.txt"), "a", encoding="utf-8") as f:
                    f.write("\n\n".join(textos) + "\n")
                textos = []
        
        except Exception as e:
            print(f"[ERRO] {e}")
        
        finally:
            # Limpa temp
            try:
                os.remove(temp_path)
            except:
                pass
    
    # Salva resto
    if textos:
        with open(os.path.join(OUTPUT_DIR, f"{nome_saida}.txt"), "a", encoding="utf-8") as f:
            f.write("\n\n".join(textos))
    
    print(f"\n  Final: {total_chars:,} caracteres, {total_linhas:,} linhas, {total_bytes/1e6:.1f}MB")
    
    # Copia para dados_brutos
    origem = os.path.join(OUTPUT_DIR, f"{nome_saida}.txt")
    if os.path.exists(origem):
        destino = os.path.join(r"C:\Users\HENDI\Desktop\HendiCode\dados_brutos", f"{nome_saida}.txt")
        import shutil
        shutil.copy2(origem, destino)
        print(f"  Copiado para: {destino}")
    
    return total_chars

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python baixar_dataset.py <dataset_id> [config] [colunas] [limite_mb] [nome_saida]")
        print("Ex:  python baixar_dataset.py code-search-net/code_search_net python func_code_string,func_documentation_string 200 python_code_docs")
        sys.exit(1)
    
    ds = sys.argv[1]
    cfg = sys.argv[2] if len(sys.argv) > 2 else None
    cols = sys.argv[3].split(",") if len(sys.argv) > 3 else ["content", "text", "code"]
    lim = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    nome = sys.argv[5] if len(sys.argv) > 5 else None
    
    processar_dataset(ds, cfg, cols, lim, nome)
