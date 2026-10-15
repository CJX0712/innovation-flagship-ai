"""L2 Transformer 解码器 — RMSNorm + RoPE + GQA + SwiGLU(+MoE) + 权重绑定

零重写 (集大成):
- Transformer (Vaswani et al. 2017)
- Pre-LN / RMSNorm (Zhang & Sennrich 2019)
- RoPE (Su et al. 2024) + NTK-aware 插值
- GQA 分组查询注意力 (Ainslie et al. 2023, INV-10)
- SwiGLU FFN (Shazeer 2020)
- MoE/Switch 稀疏专家 top-k 路由 + load-balancing (Fedus et al. 2022, INV-14)
- 权重绑定 (Press & Wolf 2017, INV-12)

可验证不变量:
- INV-10 GQA 在 n_kv==n_q 时严格 == MHA
- INV-12 输出 embedding 与输入 embedding 是同一对象 (tie)
- INV-14 MoE 路由概率和=1 且 token 分配守恒
- INV-13 参数量精确守恒
- R-L2-06 权重序列化 + sha256 manifest round-trip 误差=0
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from .. import rng
from ..core import numeric, ops
from .config import ModelConfig


class Transformer:
    def __init__(self, config: ModelConfig):
        self.cfg = config
        self._init_params()

    # ---------------- 初始化 ----------------
    def _normal(self, *shape) -> np.ndarray:
        r = rng.get_rng()
        return (r.standard_normal(shape) * self.cfg.initializer_range).astype(np.float64)

    def _init_params(self):
        c = self.cfg
        V, d = c.vocab_size, c.d_model
        p: Dict[str, np.ndarray] = {}
        p["embed"] = self._normal(V, d)                      # 输入 embedding
        if c.tie_word_embeddings:
            p["lm_head"] = p["embed"]                         # INV-12 同一对象
        else:
            p["lm_head"] = self._normal(V, d)
        p["final_norm"] = np.ones(d, dtype=np.float64)

        n_kv = c.n_kv_heads
        for l in range(c.n_layers):
            pre = f"layer{l}."
            p[pre + "attn_norm"] = np.ones(d, np.float64)
            p[pre + "q_proj"] = self._normal(d, d)
            p[pre + "k_proj"] = self._normal(d, n_kv * c.d_head)
            p[pre + "v_proj"] = self._normal(d, n_kv * c.d_head)
            p[pre + "o_proj"] = self._normal(d, d)
            p[pre + "ffn_norm"] = np.ones(d, np.float64)
            if c.use_moe:
                for e in range(c.n_experts):
                    p[pre + f"e{e}_gate"] = self._normal(d, c.d_ff)
                    p[pre + f"e{e}_up"] = self._normal(d, c.d_ff)
                    p[pre + f"e{e}_down"] = self._normal(c.d_ff, d)
                p[pre + "router"] = self._normal(d, c.n_experts)
            else:
                p[pre + "ffn_gate"] = self._normal(d, c.d_ff)
                p[pre + "ffn_up"] = self._normal(d, c.d_ff)
                p[pre + "ffn_down"] = self._normal(c.d_ff, d)
        self.params = p
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def tie_word_embeddings(self) -> bool:
        return self.cfg.tie_word_embeddings

    # ---------------- 参数量 ----------------
    def count_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    # ---------------- 前向 ----------------
    def forward(self, token_ids: np.ndarray, return_hidden: bool = False,
                position_ids: np.ndarray | None = None,
                key_padding_mask: np.ndarray | None = None):
        """前向。

        - ``position_ids`` [B,T] 或 [T]: 显式位置 (连续批处理左补齐时, 真实
          token 仍取自然位置, 保证与串行前向逐 token 一致, INV-16)。
        - ``key_padding_mask`` [B,T] bool (True=pad): 屏蔽 pad 位置作为 key,
          使补齐序列的最后真实 token 注意力上下文 == 串行上下文。
        """
        c = self.cfg
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None]
        B, T = ids.shape
        d, dh, n_h, n_kv = c.d_model, c.d_head, c.n_heads, c.n_kv_heads
        scale = 1.0 / (dh ** 0.5)
        if position_ids is None:
            positions = np.arange(T)
        else:
            positions = np.asarray(position_ids)

        x = self.params["embed"][ids]                       # [B,T,d]
        for l in range(c.n_layers):
            pre = f"layer{l}."
            h = ops.rms_norm(x, self.params[pre + "attn_norm"], c.norm_eps)
            q = h @ self.params[pre + "q_proj"]            # [B,T,d]
            k = h @ self.params[pre + "k_proj"]
            v = h @ self.params[pre + "v_proj"]
            q = ops.apply_rope(q, positions, d, c.rope_base, c.rope_scaling).reshape(B, T, n_h, dh)
            k = ops.apply_rope(k, positions, n_kv * dh, c.rope_base, c.rope_scaling).reshape(B, T, n_kv, dh)
            v = v.reshape(B, T, n_kv, dh)
            # GQA: 重复 kv 头 (INV-10 当 n_kv==n_h 时无变化)
            k = np.repeat(k, c.n_repeats, axis=2)
            v = np.repeat(v, c.n_repeats, axis=2)
            # 注意力 [B,n_h,T,T] 因果 + pad 屏蔽
            scores = np.einsum("bthd,bThd->bhtT", q, k) * scale
            causal = np.triu(np.ones((T, T), bool), k=1)
            scores = np.where(causal, -1e30, scores)
            if key_padding_mask is not None:
                # [B,1,1,T]: True=pad 位置不作为 key 被注意
                pad_mask = np.asarray(key_padding_mask)[:, None, None, :]
                scores = np.where(pad_mask, -1e30, scores)
            probs = numeric.stable_softmax(scores, axis=-1)
            ctx = np.einsum("bhtT,bThd->bthd", probs, v).reshape(B, T, d)
            o = ctx @ self.params[pre + "o_proj"]
            x = x + o
            # FFN / MoE
            h2 = ops.rms_norm(x, self.params[pre + "ffn_norm"], c.norm_eps)
            if c.use_moe:
                x = x + self._moe(h2, pre, c)
            else:
                g = h2 @ self.params[pre + "ffn_gate"]
                u = h2 @ self.params[pre + "ffn_up"]
                silu = g / (1.0 + np.exp(-g))
                ff = (silu * u) @ self.params[pre + "ffn_down"]
                x = x + ff
        x = ops.rms_norm(x, self.params["final_norm"], c.norm_eps)
        if return_hidden:
            return x
        logits = x @ self.params["lm_head"].T                 # [B,T,V]
        return logits

    def _moe(self, h2: np.ndarray, pre: str, c: ModelConfig) -> np.ndarray:
        B, T, d = h2.shape
        router_logits = h2 @ self.params[pre + "router"]    # [B,T,E]
        topk_idx = np.argsort(router_logits, axis=-1)[:, :, -c.moe_topk:]   # [B,T,topk]
        # 仅对选中专家的 router 分数做 softmax → 概率和=1 (INV-14)
        sel = np.take_along_axis(router_logits, topk_idx, axis=-1)
        w = numeric.stable_softmax(sel, axis=-1)
        out = np.zeros((B, T, d))
        for e in range(c.n_experts):
            gate = h2 @ self.params[pre + f"e{e}_gate"]
            up = h2 @ self.params[pre + f"e{e}_up"]
            silu = gate / (1.0 + np.exp(-gate))
            ex = (silu * up) @ self.params[pre + f"e{e}_down"]
            mask = topk_idx == e
            out += np.sum(w * mask, axis=-1, keepdims=True) * ex
        return out

    def logits_next(self, token_ids: np.ndarray) -> np.ndarray:
        """给定已生成序列, 返回最后一个位置的 logits [V]。"""
        logits = self.forward(token_ids)
        last = logits[:, -1, :]
        return last

    # ---------------- 序列化 + 校验 ----------------
    def save(self, dir_path: str) -> str:
        os.makedirs(dir_path, exist_ok=True)
        np.savez(os.path.join(dir_path, "params.npz"), **self.params)
        manifest = {"config": self.cfg.to_json(),
                    "params_sha256": {},
                    "total_params": self.count_params()}
        # 逐参数 sha256, 保证 R-L2-06 完整性可校验
        with open(os.path.join(dir_path, "params.npz"), "rb") as f:
            raw = f.read()
        manifest["params_sha256"]["_npz"] = hashlib.sha256(raw).hexdigest()
        with open(os.path.join(dir_path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return dir_path

    @classmethod
    def load(cls, dir_path: str, verify: bool = True) -> "Transformer":
        with open(os.path.join(dir_path, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        cfg = ModelConfig.from_json(manifest["config"])
        with open(os.path.join(dir_path, "params.npz"), "rb") as f:
            raw = f.read()
        if verify:
            assert hashlib.sha256(raw).hexdigest() == manifest["params_sha256"]["_npz"], \
                "params.npz sha256 mismatch!"
        data = np.load(os.path.join(dir_path, "params.npz"))
        model = cls.__new__(cls)
        model.cfg = cfg
        model.params = {k: data[k] for k in data.files}
        model._cache = {}
        # 重建 tie 引用
        if cfg.tie_word_embeddings:
            model.params["lm_head"] = model.params["embed"]
        assert model.count_params() == manifest["total_params"], "param count mismatch"
        return model


def save_model(model: Transformer, dir_path: str) -> str:
    return model.save(dir_path)


def load_model(dir_path: str, verify: bool = True) -> Transformer:
    return Transformer.load(dir_path, verify=verify)
