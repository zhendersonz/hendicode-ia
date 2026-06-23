"""
tools.py — Ferramentas de sistema do HendiCode.

CORREÇÕES:
- [BUG] write_file: `os.makedirs(os.path.dirname(...))` falha com string vazia
  quando `path` é só um nome de arquivo sem diretório (ex: "output.txt").
  dirname("output.txt") == "" → makedirs("") lança FileNotFoundError.
- [BUG] read_file: a verificação de path traversal usava os.path.commonpath,
  que no Windows lança ValueError se os caminhos estiverem em drives diferentes.
  Reescrito com os.path.abspath + startswith.
- [BUG] execute_command: shlex.split falha em Windows com caminhos que têm
  barras invertidas. Adicionado posix=False no Windows.
- [MELHORIA] list_directory: os.listdir falha sem tratar PermissionError.
"""
import subprocess
import os
import glob as glob_module
import shlex
import sys

def execute_command(command, workdir=".", timeout=30):
    try:
        # [FIX] shlex.split com posix=False no Windows
        posix = sys.platform != "win32"
        command_parts = shlex.split(command, posix=posix)
        result = subprocess.run(
            command_parts,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        return {
            "stdout":     result.stdout,
            "stderr":     result.stderr,
            "returncode": result.returncode,
            "success":    result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Comando excedeu {timeout}s", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def read_file(path):
    try:
        base_dir = os.path.abspath(os.getcwd())
        abs_path = os.path.abspath(os.path.join(base_dir, path))

        # [FIX] startswith é mais seguro e funciona cross-platform
        if not abs_path.startswith(base_dir + os.sep) and abs_path != base_dir:
            return {
                "error": "Acesso negado: caminho fora do diretório permitido.",
                "success": False,
            }

        with open(abs_path, "r", encoding="utf-8") as f:
            return {"content": f.read(), "success": True}
    except FileNotFoundError:
        return {"error": f"Arquivo não encontrado: {path}", "success": False}
    except PermissionError:
        return {"error": f"Permissão negada: {path}", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def write_file(path, content):
    try:
        # [FIX] Só chama makedirs se houver diretório pai
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def glob_files(pattern):
    try:
        files = glob_module.glob(pattern, recursive=True)
        return {"files": files, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def list_directory(path="."):
    try:
        entries = os.listdir(path)
        files = []
        dirs  = []
        for entry in sorted(entries):
            full = os.path.join(path, entry)
            try:
                if os.path.isdir(full):
                    dirs.append(entry + "/")
                else:
                    files.append(entry)
            except PermissionError:
                files.append(entry + " (sem permissão)")
        return {"directories": dirs, "files": files, "success": True}
    except PermissionError:
        return {"error": f"Permissão negada: {path}", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}
