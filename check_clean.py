import glob, os
files = glob.glob("agent/*.py") + glob.glob("*.py") + glob.glob("*.ipynb")
exclude = {"check_clean.py", "app.py", "test_agent.py", "__init__.py"}
qwen_count = 0
for f in sorted(files):
    if os.path.basename(f) in exclude:
        continue
    c = open(f, encoding="utf-8", errors="ignore").read()
    if "Qwen" in c or "Alibaba" in c:
        print(f"[ERRO] {f} TEM QWEN/ALIBABA")
        qwen_count += 1
    else:
        print(f"[OK] {f} limpo")
print(f"\nTotal com Qwen/Alibaba: {qwen_count}")
