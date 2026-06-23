import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
from agent.model import hendi_model

PERGUNTAS = [
    "Crie uma função que verifica se um número é palíndromo em Python",
    "Implemente uma classe Pilha com push, pop e peek em Python",
    "Escreva uma query SQL que encontra os 5 produtos mais vendidos",
    "Crie um componente React de botão com TypeScript",
    "Implemente busca binária em Python",
    "Explique o que é herança em POO com exemplo em Java",
    "Crie um decorator que mede tempo de execução em Python",
    "Faça um SELECT com JOIN entre usuarios e pedidos",
    "Implemente uma função debounce em JavaScript",
    "Crie uma API REST simples com Flask",
]

def testar():
    if not hendi_model.load():
        print("[ERRO] Modelo não carregou")
        return
    
    for i, pergunta in enumerate(PERGUNTAS, 1):
        titulo = pergunta[:60].encode("ascii", errors="replace").decode("ascii")
        print(f"\n{'='*60}")
        print(f"[{i}/{len(PERGUNTAS)}] {titulo}")
        print(f"{'='*60}")
        
        inicio = time.time()
        resposta = "".join(hendi_model.generate_stream(pergunta, max_new_tokens=256, temperature=0.2))
        tempo = time.time() - inicio
        
        tem_codigo = "```" in resposta
        tamanho = len(resposta)
        palavras = len(resposta.split())
        
        def limpar(s):
            return s.encode("ascii", errors="replace").decode("ascii")
        
        marcador = "[OK]" if tem_codigo else "[--]"
        print(f"Tempo: {tempo:.1f}s | Tamanho: {tamanho}car {palavras}pal | Codigo: {marcador}")
        if tem_codigo:
            import re
            blocos = re.findall(r'```(\w*)\n.*?```', resposta, re.DOTALL)
            print(f"Blocos de codigo: {len(blocos)}")
        print(f"Resumo: {limpar(resposta[:150])}")

if __name__ == "__main__":
    testar()
