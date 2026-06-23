# HendiCode

IA brasileira especializada em programação, treinada do zero e refinada com merge de modelos open-source.

## Créditos

Este projeto utiliza e combina dois modelos da família **DeepSeek-Coder**, desenvolvidos pela **DeepSeek AI**:

| Modelo | Origem | Licença |
|---|---|---|
| **DeepSeek-Coder-1.3B Base** | [deepseek-ai/deepseek-coder-1.3b-base](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) | MIT |
| **DeepSeek-Coder-1.3B Instruct** | [deepseek-ai/deepseek-coder-1.3b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) | MIT |

### O que foi feito

Os dois modelos foram fundidos via **interpolação linear** (t=0.75 — 75% de peso do modelo instruct, 25% do base) para criar um cérebro único e customizado para o HendiCode:

1. **DeepSeek-Coder-1.3B Base** → modelo bruto, pré-treinado em 1 trilhão de tokens de código
2. **DeepSeek-Coder-1.3B Instruct** → fine-tune do base com 2 bilhões de tokens de instrução

O merge foi realizado com o script `scripts/merge_deepseek.py`, que baixa ambos os modelos do Hugging Face, mescla os pesos camada por camada, e remove todo vestígio dos modelos originais — restando apenas o cérebro final do HendiCode.

Resultado: **1.34 bilhão de parâmetros**, ~5.1 GB, arquitetura LLaMA-like (RoPE, SwiGLU, RMSNorm, GQA).

### Referência acadêmica

```bibtex
@misc{deepseekai2024deepseekcoder,
    title={DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence},
    author={DeepSeek-AI},
    year={2024},
    eprint={2401.14196},
    archivePrefix={arXiv},
    primaryClass={cs.SE}
}
```

## Tecnologias

- **PyTorch** — inferência do modelo
- **Flask** — servidor web com streaming
- **Transformers / Tokenizers** — tokenização
- **Hugging Face Hub** — download dos modelos base

## Requisitos

- Python 3.10+
- 16 GB+ de RAM
- ~5 GB de disco para o cérebro

## Instalação

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

## Executar

```bash
venv\Scripts\python app.py
```

O servidor inicia em `http://localhost:5000`.

## Estrutura

| Pasta | Descrição |
|---|---|
| `agent/` | Núcleo da IA (modelo, memória, ferramentas) |
| `models/hendicode/` | Cérebro do HendiCode (gitignored) |
| `static/` | Front-end (CSS, JS) |
| `templates/` | Interface HTML |
| `memory/` | Memória persistente do usuário |
| `scripts/` | Utilitários de merge e preparação |

---

**HendiCode** — criado por [@zhendersonz](https://github.com/zhendersonz)
