import sys, os
sys.stdout = open("test_output.log", "w", encoding="utf-8")
sys.stderr = sys.stdout

sys.path.insert(0, os.path.dirname(__file__))
from agent.model import HendiCodeLoaderOriginal

m = HendiCodeLoaderOriginal()
print("Carregando modelo...", flush=True)
if m.load():
    print("OK. Gerando...", flush=True)
    for chunk in m.generate_stream("### Instruction:\nOla\n### Response:\n", max_new_tokens=50, temperature=0.3):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print("\nFIM", flush=True)
else:
    print("Falha", flush=True)
