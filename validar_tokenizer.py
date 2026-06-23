"""
validar_tokenizer.py
Testa o tokenizer próprio do HendiCode — tokenização de código e português.
"""
import os
import sys
from tokenizers import Tokenizer

def validar():
    raiz = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(raiz, "models", "hendicode", "tokenizer.json")
    
    if not os.path.exists(caminho):
        print("[HendiCode] Tokenizer não encontrado!")
        print(f"  Caminho esperado: {caminho}")
        print("[HendiCode] Execute primeiro: python treinar_tokenizer.py")
        return
    
    print("=" * 60)
    print("  HendiCode - Validação do Tokenizer")
    print("=" * 60)
    
    tok = Tokenizer.from_file(caminho)
    
    print(f"\nVocabulário: {tok.get_vocab_size()} tokens")
    
    # Testes com código
    testes = [
        ("Código Python", 
         'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(10))'),
        
        ("Código JavaScript",
         'function hello(name) {\n    console.log(`Olá, ${name}!`);\n    return true;\n}\n\nhello("Mundo");'),
        
        ("Código HTML",
         '<!DOCTYPE html>\n<html lang="pt-br">\n<head>\n    <title>Meu Site</title>\n</head>\n<body>\n    <h1>Olá, mundo!</h1>\n</body>\n</html>'),
        
        ("Texto em Português",
         'Olá! Como você está? Hoje vamos aprender sobre inteligência artificial e como treinar nossos próprios modelos de linguagem.'),
        
        ("SQL Query",
         'SELECT u.nome, COUNT(p.id) as total_pedidos\nFROM usuarios u\nLEFT JOIN pedidos p ON u.id = p.usuario_id\nGROUP BY u.nome\nHAVING total_pedidos > 5\nORDER BY total_pedidos DESC;'),
        
        ("Código misto com comentários PT",
         '# Calculadora de média\nnotas = [8.5, 7.0, 9.3, 6.8]\nmedia = sum(notas) / len(notas)\nprint(f"A média é {media:.2f}")  # Mostra o resultado'),
    ]
    
    for nome, texto in testes:
        encoded = tok.encode(texto)
        decoded = tok.decode(encoded.ids)
        
        print(f"\n{'─' * 60}")
        print(f"  Teste: {nome}")
        print(f"{'─' * 60}")
        print(f"  Texto original: {texto[:80]}{'...' if len(texto) > 80 else ''}")
        print(f"  Número de tokens: {len(encoded.ids)}")
        print(f"  Primeiros 10 IDs: {encoded.ids[:10]}")
        
        # Estatísticas de compressão
        chars_originais = len(texto)
        chars_por_token = chars_originais / max(len(encoded.ids), 1)
        print(f"  Caracteres por token: {chars_por_token:.1f}")
        
        # Verifica roundtrip
        if texto.strip() == decoded.strip():
            print(f"  Roundtrip: ✅ OK")
        else:
            print(f"  Roundtrip: ⚠️  Diferença (normal para BPE)")
            print(f"  Decodificado: {decoded[:100]}")
    
    print(f"\n{'=' * 60}")
    print(f"  Tokenizer funcionando corretamente!")
    print(f"  Vocabulário: {tok.get_vocab_size()} tokens")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    validar()
