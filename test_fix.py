import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from agent.model import HendiCodeLoaderOriginal

m = HendiCodeLoaderOriginal()
print("Carregando modelo...")
if m.load():
    print("OK. Gerando...")
    for chunk in m.generate_stream("### Instruction:\nOla\n### Response:\n", max_new_tokens=50, temperature=0.3):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print("\nFIM")
else:
    print("Falha")
