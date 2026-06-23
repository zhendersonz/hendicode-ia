"""
retriever.py — Retrieval de contexto com TF-IDF + pontuação melhorada.
"""
import os
import glob
import re
import math
from collections import Counter


class Retriever:
    def __init__(self, data_dir="dados_brutos"):
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", data_dir)
        if not os.path.isdir(self.data_dir):
            print(f"[Retriever] Aviso: diretório '{self.data_dir}' não encontrado. Retrieval desativado.")
        self.arquivos = sorted(glob.glob(os.path.join(self.data_dir, "*.txt")))
        self.indice   = {}
        self.df       = Counter()
        self._build_index()

    def _build_index(self):
        for arq in self.arquivos:
            nome = os.path.basename(arq)
            try:
                with open(arq, "r", encoding="utf-8", errors="ignore") as f:
                    texto = f.read()
                palavras = set(re.findall(r'\w{4,}', texto.lower()))
                for p in palavras:
                    self.indice.setdefault(p, []).append(nome)
                    self.df[p] += 1
            except OSError as e:
                print(f"[Retriever] Aviso: não foi possível indexar '{nome}': {e}")

    def _calcular_tfidf(self, termo: str, arquivo: str, termos_query: list) -> float:
        """
        Calcula TF-IDF aproximado: frequência do termo no arquivo × IDF.
        """
        caminho = os.path.join(self.data_dir, arquivo)
        try:
            with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
                texto = f.read().lower()
            palavras = re.findall(r'\w{4,}', texto)
            total = len(palavras)
            if total == 0:
                return 0
            freq_termo = palavras.count(termo)
            tf = freq_termo / total
            n_docs = max(len(self.arquivos), 1)
            idf = math.log((n_docs + 1) / (self.df.get(termo, 1) + 1)) + 1
            return tf * idf
        except OSError:
            return 0

    def _trecho_relevante(self, texto, termos, window=2000):
        texto_lower = texto.lower()
        melhor_pos = len(texto)
        for termo in termos:
            pos = texto_lower.find(termo)
            if 0 <= pos < melhor_pos:
                melhor_pos = pos

        inicio = max(0, melhor_pos - window // 4)
        fim    = min(len(texto), inicio + window)
        return texto[inicio:fim]

    def buscar(self, pergunta, top_n=3):
        termos = re.findall(r'\w{4,}', pergunta.lower())
        if not termos or not self.indice:
            return []

        relevancia = Counter()
        for termo in termos:
            for arq in self.indice.get(termo, []):
                relevancia[arq] += 1

        mais_relevantes = relevancia.most_common(top_n * 2)
        resultados = []
        for arq, score in mais_relevantes:
            # Re-rankeia com TF-IDF
            soma_tfidf = sum(self._calcular_tfidf(t, arq, termos) for t in termos)
            if soma_tfidf > 0:
                resultados.append({"arquivo": arq, "score": score, "tfidf": soma_tfidf})

        resultados.sort(key=lambda x: (x["tfidf"], x["score"]), reverse=True)
        resultados = resultados[:top_n]

        for r in resultados:
            caminho = os.path.join(self.data_dir, r["arquivo"])
            try:
                with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
                    conteudo = f.read()
                r["conteudo"] = self._trecho_relevante(conteudo, termos)
            except OSError as e:
                print(f"[Retriever] Aviso: erro ao ler '{r['arquivo']}': {e}")
                r["conteudo"] = ""

        return resultados


retriever = Retriever()
