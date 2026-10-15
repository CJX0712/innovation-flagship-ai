"""L4 TinyTrainer — autograd 驱动的可训练微型 Transformer

目标: 在数十秒内于 CPU 上完成一个微型模型的训练 (INV-26 损失下降)，
产出与 ``ifai.models.Transformer`` 完全兼容的参数文件 (同名 + 同结构),
使 ``ifai demo`` 可直接加载推理。

零重写: 训练循环范式参考 Karpathy "nanogpt" (极简 GPT 训练循环)。
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List

import numpy as np

from .. import rng
from ..core import numeric, ops
from ..models.config import ModelConfig
from .autodiff import Tensor
from .optim import AdamW, GlobalStep, clip_global_norm


class TinyTrainer:
    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.params: Dict[str, Tensor] = {}
        self._build()

    def _P(self, name, shape):
        r = rng.get_rng()
        data = (r.standard_normal(shape) * self.cfg.initializer_range).astype(np.float64)
        self.params[name] = Tensor(data, requires_grad=True)

    def _build(self):
        c = self.cfg
        V, d = c.vocab_size, c.d_model
        self._P("embed", (V, d))
        if c.tie_word_embeddings:
            self.params["lm_head"] = self.params["embed"]
        else:
            self._P("lm_head", (V, d))
        self._P("final_norm", (d,))
        n_kv = c.n_kv_heads
        for l in range(c.n_layers):
            p = f"layer{l}."
            self._P(p + "attn_norm", (d,))
            self._P(p + "q_proj", (d, d))
            self._P(p + "k_proj", (d, n_kv * c.d_head))
            self._P(p + "v_proj", (d, n_kv * c.d_head))
            self._P(p + "o_proj", (d, d))
            self._P(p + "ffn_norm", (d,))
            self._P(p + "ffn_gate", (d, c.d_ff))
            self._P(p + "ffn_up", (d, c.d_ff))
            self._P(p + "ffn_down", (c.d_ff, d))

    # ---------------- autograd 前向 ----------------
    def _rope(self, x: Tensor, pos: np.ndarray, dim: int) -> Tensor:
        """RoPE 旋转 (偶数维旋转, 支持反向传播)。

        用可微算子表达: rot1 = x1*cos - x2*sin, rot2 = x1*sin + x2*cos,
        再交织回原顺序 (反向时按位分流)。
        """
        freqs = 1.0 / (self.cfg.rope_base ** (np.arange(0, dim // 2) / (dim // 2)))
        angles = np.outer(pos, freqs)                       # [T, half]
        cos = Tensor(np.cos(angles).reshape(1, len(pos), dim // 2))   # [1,T,half]
        sin = Tensor(np.sin(angles).reshape(1, len(pos), dim // 2))
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        out = Tensor(np.zeros_like(x.data))
        out.data[..., 0::2] = rot1.data
        out.data[..., 1::2] = rot2.data

        def _b():
            if x.requires_grad:
                g = np.zeros_like(x.data)
                g[..., 0::2] = rot1.grad
                g[..., 1::2] = rot2.grad
                x.grad += g
        out._backward = _b
        out._prev = [x]
        return out

    def forward(self, ids: np.ndarray) -> Tensor:
        c = self.cfg
        B, T = ids.shape
        d, dh, n_h, n_kv = c.d_model, c.d_head, c.n_heads, c.n_kv_heads
        pos = np.arange(T)
        x = self.params["embed"][ids]                        # [B,T,d]
        for l in range(c.n_layers):
            p = f"layer{l}."
            # RMSNorm
            ms = (x * x).mean(axis=-1, keepdims=True)
            h = x / (ms + c.norm_eps).pow(0.5) * self.params[p + "attn_norm"]
            q = h.matmul(self.params[p + "q_proj"])
            k = h.matmul(self.params[p + "k_proj"])
            v = h.matmul(self.params[p + "v_proj"])
            q = self._rope(q, pos, d).data.reshape(B, T, n_h, dh)
            k = self._rope(Tensor(k.data), pos, n_kv * dh).data.reshape(B, T, n_kv, dh)
            v = v.data.reshape(B, T, n_kv, dh)
            # 注意力 (单头循环, 微型模型足够)
            ctx = np.zeros((B, T, d))
            scale = 1.0 / dh ** 0.5
            for b in range(B):
                for head in range(n_h):
                    qq = q[b, :, head, :]      # [T,dh]
                    kvh = head // c.n_repeats   # GQA: 多个 q 头共享同一 kv 头
                    kk = k[b, :, kvh, :]
                    vv = v[b, :, kvh, :]
                    scores = (qq @ kk.T) * scale          # [T,T]
                    scores = np.where(np.triu(np.ones((T, T), bool), 1), -1e30, scores)
                    p_att = numeric.stable_softmax(scores, axis=-1)
                    ctx[b, :, head * dh:(head + 1) * dh] = p_att @ vv
            ctx_t = Tensor(ctx)
            o = ctx_t.matmul(self.params[p + "o_proj"])
            x = x + o
            ms2 = (x * x).mean(axis=-1, keepdims=True)
            h2 = x / (ms2 + c.norm_eps).pow(0.5) * self.params[p + "ffn_norm"]
            g = h2.matmul(self.params[p + "ffn_gate"])
            u = h2.matmul(self.params[p + "ffn_up"])
            silu = g * g.sigmoid()                          # SwiGLU 门控
            ff = (silu * u).matmul(self.params[p + "ffn_down"])
            x = x + ff
        ms3 = (x * x).mean(axis=-1, keepdims=True)
        x = x / (ms3 + c.norm_eps).pow(0.5) * self.params["final_norm"]
        logits = x.matmul(self.params["lm_head"].transpose())     # [B,T,V]
        return logits

    def loss(self, ids: np.ndarray, targets: np.ndarray) -> Tensor:
        logits = self.forward(ids)
        # 交叉熵 (autograd 路径)
        logits_flat = logits.reshape(-1, self.cfg.vocab_size)
        tgt = targets.reshape(-1)
        # 数值稳定 log-softmax (max 作为常数 detach, 保证梯度正确)
        m = logits_flat.data.max(axis=-1, keepdims=True)
        z = logits_flat - Tensor(m, requires_grad=False)
        logsum = z.exp().sum(axis=-1, keepdims=True).log()
        lse = logsum + Tensor(m, requires_grad=False)      # 还原真实 logsumexp
        logp = logits_flat - lse                            # = log-softmax (梯度正确)
        nll = -logp[np.arange(len(tgt)), tgt].mean()
        return nll

    def train(self, ids_corpus: np.ndarray, steps: int = 200,
              lr: float = 3e-3, batch_size: int = 8, seq_len: int = 32,
              grad_clip: float = 1.0, warmup: int = 10) -> List[float]:
        params = [p for p in self.params.values() if p.requires_grad]
        opt = AdamW(params, lr=lr, weight_decay=0.01)
        sched = GlobalStep(lr, steps, warmup_steps=warmup)
        losses = []
        N = len(ids_corpus) - seq_len
        for s in range(steps):
            opt.zero_grad()
            opt.lr = sched.step()
            idx = rng.get_rng().integers(0, max(N, 1), size=batch_size)
            xb = np.stack([ids_corpus[i:i + seq_len] for i in idx])
            yb = np.stack([ids_corpus[i + 1:i + seq_len + 1] for i in idx])
            loss_t = self.loss(xb, yb)
            loss_t.backward()
            clip_global_norm(params, grad_clip)
            opt.step()
            losses.append(float(loss_t.data))
        return losses

    def to_numpy_params(self) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in self.params.items():
            out[k] = np.array(v.data, dtype=np.float64)
        # tie 引用一致性: 推理端会 rebind, 但这里保存独立数组
        if self.cfg.tie_word_embeddings:
            out["lm_head"] = out["embed"].copy()
        return out

    def save(self, dir_path: str) -> str:
        os.makedirs(dir_path, exist_ok=True)
        params = self.to_numpy_params()
        np.savez(os.path.join(dir_path, "params.npz"), **params)
        with open(os.path.join(dir_path, "params.npz"), "rb") as f:
            raw = f.read()
        manifest = {
            "config": self.cfg.to_json(),
            "params_sha256": {"_npz": hashlib.sha256(raw).hexdigest()},
            "total_params": int(sum(v.size for v in params.values())),
            "trained_by": "TinyTrainer",
        }
        with open(os.path.join(dir_path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return dir_path
