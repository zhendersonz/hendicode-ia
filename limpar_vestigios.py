"""
limpar_vestigios.py
Remove arquivos legados do projeto HendiCode.
"""
import os
import shutil

def limpar():
    raiz = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 50)
    print("  HendiCode - Limpeza de Arquivos Legados")
    print("=" * 50)
    
    remover = [
        "setup_hendicode.py",
        "merge_config.yaml",
        "gen_log.txt",
        "backup_antes_merge",
    ]
    
    substituir = {
        "agent/model.py": "agent/model_novo.py",
        "agent/core.py": "agent/core_atualizado.py",
    }
    
    print("\n[1/4] Removendo arquivos legados...")
    for item in remover:
        caminho = os.path.join(raiz, item)
        if os.path.exists(caminho):
            try:
                if os.path.isdir(caminho):
                    shutil.rmtree(caminho)
                    print(f"  OK Pasta removida: {item}/")
                else:
                    os.remove(caminho)
                    print(f"  OK Arquivo removido: {item}")
            except Exception as e:
                print(f"  Erro ao remover {item}: {e}")
        else:
            print(f"  - Nao encontrado: {item}")
    
    print("\n[2/4] Substituindo arquivos do sistema...")
    for destino, origem in substituir.items():
        caminho_destino = os.path.join(raiz, destino)
        caminho_origem = os.path.join(raiz, origem)
        
        if os.path.exists(caminho_origem):
            backup = caminho_destino + ".bak"
            if os.path.exists(caminho_destino) and not os.path.exists(backup):
                shutil.copy2(caminho_destino, backup)
                print(f"  OK Backup criado: {destino}.bak")
            
            shutil.copy2(caminho_origem, caminho_destino)
            print(f"  OK Substituido: {destino}")
        else:
            print(f"  Origem nao encontrada: {origem}")
    
    print("\n[3/4] Limpando pasta models/hendicode...")
    model_dir = os.path.join(raiz, "models", "hendicode")
    if os.path.exists(model_dir):
        para_remover_model = [
            "README.md",
            "mergekit_config.yml",
            "model.safetensors.index.json",
        ]
        for f in para_remover_model:
            caminho = os.path.join(model_dir, f)
            if os.path.exists(caminho):
                os.remove(caminho)
                print(f"  OK Removido: models/hendicode/{f}")
        
        safetensors = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
        if safetensors:
            print(f"  Encontrados {len(safetensors)} arquivos .safetensors")
            for f in safetensors:
                os.remove(os.path.join(model_dir, f))
                print(f"  OK Removido: {f}")
    
    print("\n[4/4] Limpando requirements.txt...")
    req_path = os.path.join(raiz, "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path) as f:
            linhas = f.readlines()
        
        remover_deps = ["transformers", "accelerate", "trl", "bitsandbytes"]
        novas_linhas = []
        for linha in linhas:
            dep = linha.strip().lower()
            if any(dep.startswith(r) for r in remover_deps):
                print(f"  OK Removido: {linha.strip()}")
            else:
                novas_linhas.append(linha)
        
        novas_deps = [
            "torch\n",
            "tokenizers\n",
            "datasets\n",
        ]
        for dep in novas_deps:
            if dep.strip() not in [l.strip() for l in novas_linhas]:
                novas_linhas.append(dep)
                print(f"  OK Adicionado: {dep.strip()}")
        
        with open(req_path, "w") as f:
            f.writelines(novas_linhas)
    
    print("\n" + "=" * 50)
    print("  Limpeza concluida!")
    print("=" * 50)

if __name__ == "__main__":
    limpar()
