import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class GenerationConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)
    def __getattr__(self, name):
        return None
    def to_dict(self):
        return {k: v for k, v in object.__getattribute__(self, "__dict__").items()}
    def update(self, **kwargs):
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

default_config = {
    "vocab_size": 32256,
    "hidden_size": 2048,
    "intermediate_size": 5504,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "max_position_embeddings": 16384,
    "rms_norm_eps": 1e-6,
    "rope_theta": 100000.0,
    "initializer_range": 0.02,
    "use_cache": True,
    "tie_word_embeddings": False,
    "pad_token_id": 32014,
    "bos_token_id": 32013,
    "eos_token_id": 32021,
    "model_type": "hendicode",
}

# ──────────────────────────────────────────────
# RoPE — pré-computado uma vez, reutilizado
# ──────────────────────────────────────────────
def precompute_rope_frequencies(
    head_dim: int,
    max_position_embeddings: int,
    theta: float = 100000.0,
    scaling_factor: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_position_embeddings, dtype=torch.float32)
    if scaling_factor and scaling_factor > 1.0:
        t = t / scaling_factor
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    x: (batch, seq_len, num_heads, head_dim)
    cos/sin: (max_pos, head_dim // 2)
    position_ids: (batch, seq_len) — usado no decode (1 token por vez)
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    if position_ids is not None:
        # decode step: busca posição exata de cada token
        cos_sel = cos[position_ids]          # (batch, seq_len, half)
        sin_sel = sin[position_ids]
        cos_sel = cos_sel.unsqueeze(2)       # (batch, seq_len, 1, half)
        sin_sel = sin_sel.unsqueeze(2)
    else:
        # prefill: posições 0..seq_len-1
        seq_len = x.shape[1]
        cos_sel = cos[:seq_len].unsqueeze(0).unsqueeze(2)
        sin_sel = sin[:seq_len].unsqueeze(0).unsqueeze(2)

    return torch.cat([x1 * cos_sel - x2 * sin_sel, x1 * sin_sel + x2 * cos_sel], dim=-1)


# ──────────────────────────────────────────────
# KV Cache
# ──────────────────────────────────────────────
class KVCache:
    """
    Cache de chaves e valores por camada.
    Evita recomputar K/V de tokens anteriores a cada passo de geração.
    """
    def __init__(self):
        self._cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []

    def reset(self):
        self._cache.clear()

    def init_layers(self, num_layers: int):
        self._cache = [None] * num_layers

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatena K/V novos com o cache existente.
        new_k/new_v: (batch, seq_new, num_kv_heads, head_dim)
        Retorna K/V completos: (batch, seq_total, num_kv_heads, head_dim)
        """
        if self._cache[layer_idx] is None:
            self._cache[layer_idx] = (new_k, new_v)
        else:
            cached_k, cached_v = self._cache[layer_idx]
            new_k = torch.cat([cached_k, new_k], dim=1)
            new_v = torch.cat([cached_v, new_v], dim=1)
            self._cache[layer_idx] = (new_k, new_v)
        return self._cache[layer_idx]

    def seq_len(self, layer_idx: int) -> int:
        if self._cache[layer_idx] is None:
            return 0
        return self._cache[layer_idx][0].shape[1]


# ──────────────────────────────────────────────
# Módulos
# ──────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        scaling_factor = config.get("rope_scaling_factor")
        cos, sin = precompute_rope_frequencies(
            self.head_dim,
            config["max_position_embeddings"],
            config["rope_theta"],
            scaling_factor,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        kv_cache: Optional[KVCache] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = apply_rotary_emb(q, self.rope_cos, self.rope_sin, position_ids)
        k = apply_rotary_emb(k, self.rope_cos, self.rope_sin, position_ids)

        # Atualiza e recupera KV Cache
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # GQA: expande K/V para o número de heads de query
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=2)
            v = v.repeat_interleave(self.num_groups, dim=2)

        q = q.transpose(1, 2)   # (batch, num_heads, seq_q, head_dim)
        k = k.transpose(1, 2)   # (batch, num_heads, seq_kv, head_dim)
        v = v.transpose(1, 2)

        # is_causal=True só no prefill (seq_len > 1)
        # No decode (seq_len == 1) não precisa de máscara causal
        is_causal = seq_len > 1

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.o_proj(attn_output)


class SwiGLUFFN(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj   = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = SwiGLUFFN(config)
        self.input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, self.layer_idx, kv_cache, position_ids)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


# ──────────────────────────────────────────────
# Modelo principal
# ──────────────────────────────────────────────
class ConfigObject:
    def __init__(self, config_dict: dict):
        object.__setattr__(self, "_config_dict", config_dict)
        for k, v in config_dict.items():
            object.__setattr__(self, k, v)

    def __getitem__(self, key):
        return object.__getattribute__(self, "_config_dict")[key]

    def __contains__(self, key):
        return key in object.__getattribute__(self, "_config_dict")

    def get(self, key, default=None):
        return object.__getattribute__(self, "_config_dict").get(key, default)


class HendiCodeModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = ConfigObject(config)

        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config["num_hidden_layers"])]
        )
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

        if config.get("tie_word_embeddings", True):
            self.embed_tokens.weight = self.lm_head.weight

        self.generation_config = GenerationConfig(
            pad_token_id=config.get("pad_token_id", 32014),
            eos_token_id=config.get("eos_token_id", 32021),
            bos_token_id=config.get("bos_token_id", 32013),
            max_length=2048,
        )
        self.main_input_name = "input_ids"

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, kv_cache, position_ids)
        x = self.norm(x)
        return self.lm_head(x)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        stop_token_ids=None,
    ):
        """
        Geração com KV Cache:
        - Prefill: processa o prompt inteiro de uma vez (1 forward pass)
        - Decode: 1 token por forward pass, reutilizando K/V do cache
        - repetition_penalty: desencoraja repetição de tokens já gerados (>1.0)
        """
        self.eval()

        if stop_token_ids is None:
            stop_token_ids = {self.config.get("eos_token_id", 32021)}
        elif isinstance(stop_token_ids, int):
            stop_token_ids = {stop_token_ids}
        else:
            stop_token_ids = set(stop_token_ids)

        # Inicializa cache
        kv_cache = KVCache()
        kv_cache.init_layers(len(self.layers))

        # ── Prefill ──────────────────────────────
        seq_len = input_ids.shape[1]
        if seq_len > self.config["max_position_embeddings"]:
            input_ids = input_ids[:, -self.config["max_position_embeddings"]:]
            seq_len = input_ids.shape[1]

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        logits = self.forward(input_ids, kv_cache=kv_cache, position_ids=position_ids)
        next_token_logits = logits[:, -1, :]

        # ── Decode loop ──────────────────────────
        current_pos = seq_len

        # Rastreia tokens gerados para repetition_penalty
        tokens_gerados = [] if repetition_penalty > 1.0 else None

        for _ in range(max_new_tokens):
            if tokens_gerados is not None:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, tokens_gerados, repetition_penalty
                )

            next_token = self._sample(next_token_logits, temperature, top_k, top_p)

            if tokens_gerados is not None:
                tokens_gerados.append(next_token.item())

            yield next_token

            if next_token.item() in stop_token_ids:
                break

            # Próximo passo: só o novo token, posição atual
            position_ids = torch.tensor([[current_pos]], device=input_ids.device)
            logits = self.forward(next_token, kv_cache=kv_cache, position_ids=position_ids)
            next_token_logits = logits[:, -1, :]
            current_pos += 1

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        tokens_gerados: List[int],
        penalty: float,
    ) -> torch.Tensor:
        if not tokens_gerados:
            return logits
        token_ids = torch.tensor(tokens_gerados, device=logits.device, dtype=torch.long)
        logits_penalizados = logits.clone()
        for tid in token_ids:
            if logits_penalizados[0, tid] < 0:
                logits_penalizados[0, tid] *= penalty
            else:
                logits_penalizados[0, tid] /= penalty
        return logits_penalizados

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
            min_top_k = top_k_vals[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_top_k,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            indices_to_remove = sorted_remove.scatter(-1, sorted_indices, sorted_remove)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
