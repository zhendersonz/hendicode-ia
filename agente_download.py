"""
agente_download.py
Agente de download especializado para um dataset específico.
Uso: python agente_download.py <agente_id> <dataset_id> [config] [colunas] [limite_mb] [nome_saida] [max_files]
"""
import sys, os, json, subprocess, time, shutil, glob

OUTPUT_DIR = r"C:\Users\HENDI\Desktop\dados_5gb"
BRUTOS_DIR = r"C:\Users\HENDI\Desktop\HendiCode\dados_brutos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BRUTOS_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, f"log_{sys.argv[1] if len(sys.argv)>1 else 'unknown'}.txt")

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")
    print(msg)

def curl_list(url, timeout=120):
    for attempt in range(3):
        try:
            r = subprocess.run(["curl", "-sL", "-m", str(timeout), url],
                             capture_output=True, timeout=timeout+30)
            out = r.stdout.decode("utf-8", errors="replace")
            if out.strip():
                return out
        except Exception as e:
            log(f"  curl_list attempt {attempt+1}: {e}")
        time.sleep(5)
    return None

def curl_download(url, dest, timeout=600):
    for attempt in range(3):
        try:
            r = subprocess.run(["curl", "-sL", "-m", str(timeout), "-o", dest, url],
                             capture_output=True, timeout=timeout+60)
            if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 100:
                return True
        except Exception as e:
            log(f"  curl_download attempt {attempt+1}: {e}")
        time.sleep(5)
    return False

def main():
    agente_id = sys.argv[1] if len(sys.argv) > 1 else "agente"
    dataset_id = sys.argv[2] if len(sys.argv) > 2 else None
    config_name = sys.argv[3] if len(sys.argv) > 3 else ""
    colunas = sys.argv[4].split(",") if len(sys.argv) > 4 else ["content"]
    limite_mb = int(sys.argv[5]) if len(sys.argv) > 5 else 500
    nome_saida = sys.argv[6] if len(sys.argv) > 6 else dataset_id.replace("/", "_")
    max_files = int(sys.argv[7]) if len(sys.argv) > 7 else 50

    if not dataset_id:
        log("ERRO: dataset_id nao fornecido")
        return 0

    log(f"\n{'='*60}")
    log(f"AGENTE {agente_id}: {dataset_id}")
    if config_name: log(f"  Config: {config_name}")
    log(f"  Colunas: {colunas}")
    log(f"  Limite: {limite_mb} MB")
    log(f"  Max files: {max_files}")
    log(f"{'='*60}")

    # Pega info do dataset
    info_url = f"https://huggingface.co/api/datasets/{dataset_id}"
    info_json = curl_list(info_url)
    if not info_json:
        log(f"ERRO: Nao foi possivel obter info do dataset {dataset_id}")
        return 0

    info = json.loads(info_json)
    siblings = info.get("siblings", [])

    # Filtra arquivos
    if config_name:
        archs = [s for s in siblings if s["rfilename"].startswith(f"{config_name}/")]
    else:
        archs = [s for s in siblings if not s["rfilename"].startswith(".git") and not s["rfilename"].startswith(".")]

    # Separa parquet e json
    parquets = [s for s in archs if s["rfilename"].endswith(".parquet")]
    jsons = [s for s in archs if s["rfilename"].endswith(".json")]

    if parquets:
        arquivos = parquets
        formato = "parquet"
    elif jsons:
        arquivos = jsons
        formato = "json"
    else:
        log("ERRO: Nenhum arquivo parquet ou json encontrado")
        return 0

    log(f"Arquivos {formato} encontrados: {len(arquivos)}")

    # Limita numero de arquivos
    arquivos = arquivos[:max_files]

    total_downloaded = 0
    total_chars = 0
    total_rows = 0
    limite_bytes = limite_mb * 1024 * 1024

    saída_path = os.path.join(OUTPUT_DIR, f"{nome_saida}.txt")
    texto_acum = []

    for i, arq in enumerate(arquivos):
        if total_downloaded >= limite_bytes:
            log(f"  Limite de {limite_mb}MB de download atingido. Parando.")
            break

        rfilename = arq["rfilename"]
        url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{rfilename}"
        temp_path = os.path.join(OUTPUT_DIR, f"_temp_{agente_id}_{i}.{formato}")

        log(f"  [{i+1}/{len(arquivos)}] {rfilename}...")

        # Download com timeout maior para arquivos grandes
        dl_timeout = 1800 if formato == "parquet" else 600
        ok = curl_download(url, temp_path, dl_timeout)

        if not ok:
            log(f"    FALHA no download")
            continue

        file_size = os.path.getsize(temp_path)
        total_downloaded += file_size
        log(f"    Download: {file_size/1e6:.1f} MB")

        try:
            if formato == "parquet":
                import pyarrow.parquet as pq
                table = pq.read_table(temp_path)
                nrows = len(table)
                log(f"    Linhas: {nrows:,}")

                for col in colunas:
                    if col in table.column_names:
                        col_data = table.column(col)
                        for val in col_data.to_pylist():
                            if val and isinstance(val, str) and len(val) > 30:
                                texto_acum.append(val)
                                total_chars += len(val)
                                total_rows += 1

                                # Salva a cada 500 entradas
                                if len(texto_acum) >= 500:
                                    with open(saída_path, "a", encoding="utf-8") as f:
                                        f.write("\n\n".join(texto_acum) + "\n")
                                    texto_acum = []

            else:
                with open(temp_path, "r", encoding="utf-8") as f:
                    content = f.read()
                try:
                    import json as jm
                    lines = content.strip().split("\n")
                    for line in lines[:20000]:
                        try:
                            obj = jm.loads(line)
                            for col in colunas:
                                val = obj.get(col, "")
                                if val and isinstance(val, str) and len(val) > 30:
                                    texto_acum.append(val)
                                    total_chars += len(val)
                                    total_rows += 1
                                    if len(texto_acum) >= 500:
                                        with open(saída_path, "a", encoding="utf-8") as f:
                                            f.write("\n\n".join(texto_acum) + "\n")
                                        texto_acum = []
                        except:
                            pass
                except:
                    # Tenta como JSON array
                    try:
                        import json as jm
                        arr = jm.loads(content)
                        if isinstance(arr, list):
                            for obj in arr[:20000]:
                                for col in colunas:
                                    val = obj.get(col, "") if isinstance(obj, dict) else ""
                                    if val and isinstance(val, str) and len(val) > 30:
                                        texto_acum.append(val)
                                        total_chars += len(val)
                                        total_rows += 1
                                        if len(texto_acum) >= 500:
                                            with open(saída_path, "a", encoding="utf-8") as f:
                                                f.write("\n\n".join(texto_acum) + "\n")
                                            texto_acum = []
                    except:
                        log(f"    ERRO ao processar JSON")

            log(f"    OK: +{total_chars:,} chars, {total_rows:,} linhas")

        except Exception as e:
            log(f"    ERRO: {e}")

        finally:
            try:
                os.remove(temp_path)
            except:
                pass

    # Salva resto
    if texto_acum:
        with open(saída_path, "a", encoding="utf-8") as f:
            f.write("\n\n".join(texto_acum) + "\n")

    # Copia para dados_brutos
    try:
        shutil.copy2(saída_path, os.path.join(BRUTOS_DIR, f"{nome_saida}.txt"))
    except:
        pass

    log(f"\n  FINAL: {total_chars:,} chars, {total_rows:,} linhas, {total_downloaded/1e6:.1f}MB baixados")
    return total_chars

if __name__ == "__main__":
    main()
