"""
runpod_treino.py
Treinamento do HendiCode 1B em GPU paga com suporte a multi-GPU (DDP).
Uso: torchrun --nproc_per_node=N runpod_treino.py
     python runpod_treino.py (single GPU)
"""
import os
import sys
import glob
import json
import math
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from modelo_original import HendiCodeModel, default_config


def setup_ddp():
    if "RANK" in os.environ:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, True
    return 0, 1, False


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class TextDataset(Dataset):
    def __init__(self, data_dir="dados_processados", seq_length=2048):
        self.seq_length = seq_length
        self.data = []
        
        arquivos = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        print(f"Carregando {len(arquivos)} arquivos de dados...")
        
        for arq in arquivos:
            with open(arq, "rb") as f:
                raw = f.read()
            n_tokens = len(raw) // 4
            tokens = [int.from_bytes(raw[i:i+4], "little") for i in range(0, len(raw), 4)]
            self.data.extend(tokens)
            print(f"  {os.path.basename(arq)}: {n_tokens} tokens")
        
        self.data = torch.tensor(self.data, dtype=torch.long)
        print(f"Total: {len(self.data)} tokens")
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_length + 1]
        return chunk[:-1], chunk[1:]


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="models/hendicode")
    parser.add_argument("--model_size", type=str, default="1b", choices=["100m", "300m", "1b"])
    args = parser.parse_args()
    
    local_rank, world_size, is_ddp = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    is_main = local_rank == 0
    
    # Configurações de tamanho do modelo
    tamanhos = {
        "100m": {"hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12, "num_attention_heads": 12, "num_key_value_heads": 4},
        "300m": {"hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 20, "num_attention_heads": 16, "num_key_value_heads": 4},
        "1b":   {"hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 24, "num_attention_heads": 16, "num_key_value_heads": 4},
    }
    
    config = dict(default_config)
    config.update(tamanhos[args.model_size])
    config["max_position_embeddings"] = args.seq_length
    
    if is_main:
        print(f"[HendiCode] Configuração: {args.model_size}")
        print(f"  GPU(s): {world_size}")
        print(f"  Batch: {args.batch_size} × {args.grad_accum} accum × {world_size} GPUs = {args.batch_size * args.grad_accum * world_size} efetivo")
        print(f"  Seq length: {args.seq_length}")
        print(f"  Parâmetros: calculando...")
    
    # Dataset
    dataset = TextDataset(seq_length=args.seq_length)
    sampler = DistributedSampler(dataset) if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    # Modelo
    model = HendiCodeModel(config).to(device)
    
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parâmetros totais: {total_params:,}")
    
    # Otimizador e scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    
    start_step = 0
    if args.resume and is_main:
        if os.path.exists(args.resume):
            model.load_state_dict(torch.load(args.resume, map_location=device))
            print(f"[HendiCode] Retomando de {args.resume}")
    
    total_steps = len(loader) * args.epochs // args.grad_accum
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    
    scaler = torch.cuda.amp.GradScaler() if device != "cpu" else None
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if is_main:
        print(f"[HendiCode] Iniciando treino por {args.epochs} épocas...")
    
    model.train()
    global_step = 0
    tokens_processados = 0
    inicio = time.time()
    
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        
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
            
            tokens_processados += inputs.numel() * world_size
            
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
                
                if is_main and global_step % 20 == 0:
                    elapsed = time.time() - inicio
                    tok_s = tokens_processados / max(elapsed, 1)
                    print(
                        f"[{args.model_size.upper()}] "
                        f"Ep {epoch+1}/{args.epochs} "
                        f"Step {global_step}/{total_steps} "
                        f"Loss: {loss.item() * args.grad_accum:.4f} "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} "
                        f"Tok/s: {tok_s:.0f} "
                        f"GPU{local_rank}"
                    )
                
                if is_main and global_step % args.save_every == 0:
                    caminho = os.path.join(args.output_dir, f"checkpoint_step_{global_step}.pt")
                    if is_ddp:
                        torch.save(model.module.state_dict(), caminho)
                    else:
                        torch.save(model.state_dict(), caminho)
                    print(f"[HendiCode] Checkpoint salvo: {caminho}")
    
    # Salva modelo final
    if is_main:
        caminho_final = os.path.join(args.output_dir, "pesos_finais.pt")
        if is_ddp:
            torch.save(model.module.state_dict(), caminho_final)
        else:
            torch.save(model.state_dict(), caminho_final)
        
        config_path = os.path.join(args.output_dir, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        elapsed_total = time.time() - inicio
        print(f"[HendiCode] Treino concluído em {elapsed_total/3600:.1f} horas!")
        print(f"[HendiCode] Tokens processados: {tokens_processados:,}")
        print(f"[HendiCode] Modelo final: {caminho_final}")
    
    cleanup_ddp()


if __name__ == "__main__":
    train()
