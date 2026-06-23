"""
finetunar.py
Fine-tuning do HendiCode com suporte a LoRA (via PEFT) ou full fine-tuning.
Uso: python finetunar.py --epochs 3 --batch_size 2 --lora
"""
import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from modelo_original import HendiCodeModel, default_config

RAIZ = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(RAIZ, "models", "hendicode")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
PESOS_INICIAIS = os.path.join(MODEL_DIR, "pesos_finais.pt")
DADOS_PATH = os.path.join(RAIZ, "dados_finetuning.jsonl")

# Tentativa de importar PEFT (opcional)
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_DISPONIVEL = True
except ImportError:
    PEFT_DISPONIVEL = False


class FinetuningDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, seq_length=1024):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.exemplos = []

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"Arquivo {jsonl_path} não encontrado. "
                "Crie um arquivo JSONL com {'instruction': ..., 'response': ...}"
            )

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                dado = json.loads(linha)
                texto = (
                    f"### Instruction:\n{dado['instruction']}\n\n### Response:\n{dado['response']}"
                )
                self.exemplos.append(texto)

        print(f"[Finetuning] Carregados {len(self.exemplos)} exemplos de {jsonl_path}")

    def __len__(self):
        return len(self.exemplos)

    def __getitem__(self, idx):
        return self.exemplos[idx]


def collate_fn(batch, tokenizer, seq_length):
    encoded = tokenizer.encode_batch(batch)
    input_ids_list = []
    for enc in encoded:
        ids = enc.ids[:seq_length]
        input_ids_list.append(torch.tensor(ids, dtype=torch.long))

    padded = nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=0
    )
    if padded.size(1) > seq_length:
        padded = padded[:, :seq_length]

    inputs = padded[:, :-1]
    targets = padded[:, 1:]
    return inputs, targets


def contar_parametros(modelo):
    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, treinaveis


def setup_lora(modelo, r=16, lora_alpha=32):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    modelo = get_peft_model(modelo, config)
    return modelo


def train():
    parser = argparse.ArgumentParser(description="Fine-tuning HendiCode com LoRA")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint .pt para retomar")
    parser.add_argument("--lora", action="store_true",
                        help="Usar LoRA (requer PEFT)")
    parser.add_argument("--no_lora", action="store_true",
                        help="Forçar full fine-tuning mesmo com PEFT disponível")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"[Finetuning] Dispositivo: {device}")
    if use_amp:
        print(f"[Finetuning] Mixed precision: ATIVADO")

    usar_lora = args.lora or (PEFT_DISPONIVEL and not args.no_lora)
    if usar_lora and not PEFT_DISPONIVEL:
        print("[AVISO] PEFT não instalado. Usando full fine-tuning.")
        usar_lora = False
    if usar_lora:
        print("[Finetuning] Modo: LoRA (apenas adaptadores serão treinados)")
    else:
        print("[Finetuning] Modo: Full fine-tuning")

    # Carrega tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer não encontrado em {TOKENIZER_PATH}. "
            "Execute treinar_tokenizer.py primeiro."
        )
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"[Finetuning] Tokenizer carregado: {tokenizer.get_vocab_size()} tokens")

    # Dataset
    dataset = FinetuningDataset(DADOS_PATH, tokenizer, args.seq_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.seq_length),
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # Modelo
    config = dict(default_config)
    config["max_position_embeddings"] = args.seq_length
    modelo = HendiCodeModel(config)

    if os.path.exists(PESOS_INICIAIS):
        print(f"[Finetuning] Carregando pesos base: {PESOS_INICIAIS}")
        estado = torch.load(PESOS_INICIAIS, map_location="cpu")
        resultado = modelo.load_state_dict(estado, strict=False)
        if resultado.missing_keys:
            print(f"[Finetuning] Chaves faltando (ignorado): {resultado.missing_keys}")
        if resultado.unexpected_keys:
            print(f"[Finetuning] Chaves inesperadas (ignorado): {resultado.unexpected_keys}")
    else:
        print(f"[AVISO] {PESOS_INICIAIS} não encontrado. Usando pesos aleatórios.")

    if usar_lora:
        modelo = setup_lora(modelo, r=16, lora_alpha=32)
        # Apenas adaptadores LoRA devem ser treináveis
        for name, param in modelo.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

    modelo = modelo.to(device)
    total_params, trainable_params = contar_parametros(modelo)
    print(f"[Finetuning] Parâmetros totais: {total_params:,}")
    print(f"[Finetuning] Parâmetros treináveis: {trainable_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"[Finetuning] Retomando de checkpoint: {args.resume}")
            modelo.load_state_dict(
                torch.load(args.resume, map_location=device), strict=False
            )
            info_path = args.resume.replace(".pt", "_info.json")
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                start_epoch = info.get("epoch", 0)
                global_step = info.get("step", 0)
        else:
            print(f"[AVISO] Checkpoint {args.resume} não encontrado. Ignorando.")

    # Otimizador e scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, modelo.parameters()),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )
    total_steps = len(loader) * args.epochs // args.grad_accum
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\n[Finetuning] Iniciando fine-tuning:")
    print(f"  Épocas: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Grad Accum: {args.grad_accum}")
    print(f"  Seq Length: {args.seq_length}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Total steps: {total_steps}")
    print()

    modelo.train()
    loss_inicial = None
    inicio = time.time()

    for epoch in range(start_epoch, args.epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = modelo(inputs)
                    loss = nn.CrossEntropyLoss()(
                        logits.reshape(-1, config["vocab_size"]), targets.reshape(-1)
                    )
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                logits = modelo(inputs)
                loss = nn.CrossEntropyLoss()(
                    logits.reshape(-1, config["vocab_size"]), targets.reshape(-1)
                )
                loss = loss / args.grad_accum
                loss.backward()

            loss_item = loss.item() * args.grad_accum
            if loss_inicial is None:
                loss_inicial = loss_item

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                epoch_loss += loss_item
                epoch_steps += 1

                if global_step % 10 == 0:
                    elapsed = time.time() - inicio
                    print(
                        f"[Finetuning] Época {epoch+1}/{args.epochs} "
                        f"Step {global_step}/{total_steps} "
                        f"Loss: {loss_item:.4f} "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} "
                        f"Tempo: {elapsed:.0f}s"
                    )

                if global_step % args.save_every == 0:
                    if usar_lora:
                        subdir = os.path.join(MODEL_DIR, "lora")
                        os.makedirs(subdir, exist_ok=True)
                        caminho = os.path.join(subdir, f"adapter_step_{global_step}")
                        modelo.save_pretrained(caminho)
                        print(f"[Finetuning] Adaptadores LoRA salvos: {caminho}")
                    else:
                        caminho = os.path.join(
                            MODEL_DIR, f"checkpoint_finetune_step_{global_step}.pt"
                        )
                        torch.save(modelo.state_dict(), caminho)
                        info = {"epoch": epoch, "step": global_step, "loss": loss_item}
                        with open(caminho.replace(".pt", "_info.json"), "w") as f:
                            json.dump(info, f)
                        print(f"[Finetuning] Checkpoint salvo: {caminho}")

        epoch_loss_avg = epoch_loss / max(epoch_steps, 1)
        print(f"[Finetuning] Fim da época {epoch+1} — Loss média: {epoch_loss_avg:.4f}")

    loss_final = epoch_loss / max(epoch_steps, 1)
    elapsed_total = time.time() - inicio

    # Salvamento final
    if usar_lora:
        lora_dir = os.path.join(MODEL_DIR, "lora")
        os.makedirs(lora_dir, exist_ok=True)
        modelo.save_pretrained(lora_dir)
        print(f"[Finetuning] Adaptadores LoRA finais salvos em: {lora_dir}")
    else:
        caminho_final = os.path.join(MODEL_DIR, "pesos_finetuned.pt")
        torch.save(modelo.state_dict(), caminho_final)
        print(f"[Finetuning] Modelo fine-tuned salvo: {caminho_final}")

    print(f"\n[Finetuning] Fine-tuning concluído!")
    print(f"  Loss inicial: {loss_inicial:.4f}")
    print(f"  Loss final:   {loss_final:.4f}")
    print(f"  Tempo total:  {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")


if __name__ == "__main__":
    train()
