import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from modelo_original import HendiCodeModel

config_path = "models/hendicode/model_config.json"
with open(config_path) as f:
    cfg = json.load(f)

model = HendiCodeModel(cfg)

ft_path = "models/hendicode/pesos_finais.pt"
if os.path.exists(ft_path):
    import torch
    state = torch.load(ft_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    loaded = sum(v.numel() for v in state.values())
    print(f"Fine-tuned weights loaded: {ft_path} ({loaded:,} params)")
else:
    print(f"No fine-tuned weights found at {ft_path}")

lora_dir = "models/hendicode/lora"
if os.path.exists(lora_dir):
    files = os.listdir(lora_dir)
    print(f"LoRA adapters found: {files}")
else:
    print(f"No LoRA directory at {lora_dir}")

model.eval()
input_ids = torch.tensor([[32013, 123, 456]])
logits = model(input_ids)
print(f"Model forward OK. Logits shape: {logits.shape}")
