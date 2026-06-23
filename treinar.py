"""
treinar.py
Loop de treinamento para o HendiCode original.
Uso: python treinar.py --epochs 5 --batch_size 4 --lr 3e-4
"""
import os
import sys
import glob
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

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "models", "hendicode")

def load_config(config_name):
    if not config_name:
        return dict(default_config)
    caminho = os.path.join(CONFIG_DIR, config_name)
    if not caminho.endswith(".json"):
        caminho += ".json"
    if os.path.exists(caminho):
        with open(caminho) as f:
            cfg = json.load(f)
        print(f"[HendiCode] Config carregada: {caminho}")
        return cfg
    print(f"[AVISO] Config '{config_name}' não encontrada em {CONFIG_DIR}. Usando default.")
    return dict(default_config)


class TextDataset(Dataset):
    def __init__(self, data_dir="dados_processados", seq_length=2048):
        self.seq_length = seq_length
        self.data = []

        arquivos = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not arquivos:
            print("[AVISO] Nenhum .bin encontrado. Usando dados sintéticos.")
            self.data = torch.randint(0, 1000, (10000,), dtype=torch.long)
            return

        print(f"[HendiCode] Carregando dados de {len(arquivos)} arquivos...")
        for arq in arquivos:
            with open(arq, "rb") as f:
                raw = f.read()
            tokens = []
            for i in range(0, len(raw), 4):
                tid = int.from_bytes(raw[i:i+4], "little")
                tokens.append(tid)
            self.data.extend(tokens)
            print(f"  {os.path.basename(arq)}: {len(tokens)} tokens")

        self.data = torch.tensor(self.data, dtype=torch.long)
        print(f"[HendiCode] Total: {len(self.data)} tokens")

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_length + 1]
        return chunk[:-1], chunk[1:]


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="models/hendicode")
    parser.add_argument("--amp", action="store_true", help="Usar mixed precision")
    parser.add_argument("--config", type=str, default=None, help="Nome da config (ex: 100m_config)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device == "cuda"
    print(f"[HendiCode] Dispositivo: {device}")
    if use_amp:
        print(f"[HendiCode] Mixed precision: ATIVADO")

    dataset = TextDataset(seq_length=args.seq_length)
    if len(dataset) == 0:
        print("[ERRO] Dataset vazio. Coloque arquivos .bin em dados_processados/")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )

    config = load_config(args.config)
    config["max_position_embeddings"] = args.seq_length

    model = HendiCodeModel(config).to(device)

    start_step = 0
    if args.resume:
        print(f"[HendiCode] Retomando de {args.resume}...")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        if os.path.exists(os.path.join(args.output_dir, "checkpoint_info.json")):
            with open(os.path.join(args.output_dir, "checkpoint_info.json")) as f:
                info = json.load(f)
                start_step = info.get("step", 0)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = len(loader) * args.epochs // args.grad_accum
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[HendiCode] Iniciando treino:")
    print(f"  Épocas: {args.epochs}")
    print(f"  Batch: {args.batch_size}")
    print(f"  Grad Accum: {args.grad_accum}")
    print(f"  Steps por época: {len(loader)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    global_step = start_step
    tokens_processados = 0
    inicio = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(inputs)
                    loss = nn.CrossEntropyLoss()(logits.view(-1, config["vocab_size"]), targets.view(-1))
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                logits = model(inputs)
                loss = nn.CrossEntropyLoss()(logits.view(-1, config["vocab_size"]), targets.view(-1))
                loss = loss / args.grad_accum
                loss.backward()

            tokens_processados += inputs.numel()

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

                global_step += 1
                epoch_loss += loss.item() * args.grad_accum

                elapsed = time.time() - inicio
                tokens_por_seg = tokens_processados / max(elapsed, 1)
                print(
                    f"[HendiCode] Época {epoch+1}/{args.epochs} "
                    f"Step {global_step}/{total_steps} "
                    f"Loss: {loss.item() * args.grad_accum:.4f} "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} "
                    f"Tok/s: {tokens_por_seg:.0f}"
                )

                if global_step % args.save_every == 0:
                    caminho = os.path.join(args.output_dir, f"checkpoint_step_{global_step}.pt")
                    torch.save(model.state_dict(), caminho)
                    with open(os.path.join(args.output_dir, "checkpoint_info.json"), "w") as f:
                        json.dump({"step": global_step, "loss": loss.item() * args.grad_accum}, f)
                    print(f"[HendiCode] Checkpoint salvo: {caminho}")

    caminho_final = os.path.join(args.output_dir, "pesos_finais.pt")
    torch.save(model.state_dict(), caminho_final)
    print(f"[HendiCode] Modelo final salvo: {caminho_final}")

    config_path = os.path.join(args.output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[HendiCode] Config do modelo salva: {config_path}")

    elapsed_total = time.time() - inicio
    print(f"[HendiCode] Treino concluído em {elapsed_total/60:.1f} minutos!")
    print(f"[HendiCode] Total de tokens processados: {tokens_processados:,}")


if __name__ == "__main__":
    train()
