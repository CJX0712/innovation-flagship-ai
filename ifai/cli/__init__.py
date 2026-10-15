"""L9 命令行接口 (CLI) — `ifai`

子命令 (≥10): train, demo, generate, chat, serve, eval, search, agent,
tokenize, version, doctor, selftest。覆盖 R-L9-01。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .. import __version__, rng
from ..models.config import ModelConfig
from ..models.tokenizer import Tokenizer
from ..models.transformer import Transformer, load_model
from ..train.trainer import TinyTrainer
from ..engine.sampler import Sampler
from ..agent.tools import default_registry
from ..agent.react import ReActAgent
from ..agent.sandbox import SafeCodeExecutor
from ..retrieval.bm25 import BM25

DEFAULT_MODEL_DIR = "./ifai_model"
MICRO_CORPUS = (
    "人工智能是计算机科学的一个分支。模型通过学习数据来提升能力。"
    "变压器网络由注意力机制构成。注意力让模型关注重要的信息。"
    "训练需要大量的数据和计算资源。推理是把训练好的模型用于预测。"
    "分词把文本切分成更小的单元。嵌入把符号映射到向量空间。"
    "采样从模型的输出分布中选择下一个词。检索从知识库中找到相关资料。"
    "智能体可以调用工具来完成任务。记忆帮助智能体积累经验。"
)


def _load(model_dir):
    tok = Tokenizer.from_json(open(f"{model_dir}/tokenizer.json", encoding="utf-8").read())
    model = load_model(model_dir)
    return model, tok


def cmd_train(args):
    rng.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    corpus = MICRO_CORPUS
    if args.corpus:
        corpus = open(args.corpus, encoding="utf-8").read()
    cfg = ModelConfig(vocab_size=args.vocab, d_model=args.d_model,
                      n_layers=args.layers, n_heads=args.heads,
                      n_kv_heads=args.kv_heads, ffn_mult=args.ffn_mult,
                      max_seq_len=args.seq_len)
    tok = Tokenizer(vocab_size=args.vocab).train(corpus)
    tok.to_json(f"{args.out}/tokenizer.json")
    ids = tok.encode(corpus)
    trainer = TinyTrainer(cfg)
    t0 = time.perf_counter()
    losses = trainer.train(np.array(ids), steps=args.steps, lr=args.lr)
    dt = time.perf_counter() - t0
    trainer.save(args.out)
    nparams = sum(v.size for v in trainer.to_numpy_params().values())
    print(f"训练完成: {args.steps} 步, 用时 {dt:.1f}s, 末损 {losses[-1]:.4f}")
    print(f"模型已保存至 {args.out} (参数量 {nparams})")


def cmd_demo(args):
    if not _exists(f"{args.model}/manifest.json"):
        print("未找到模型, 先训练一个微型模型...")
        cmd_train(argparse.Namespace(seed=0, corpus=None, vocab=512, d_model=128,
                                     layers=3, heads=4, kv_heads=2, ffn_mult=4,
                                     seq_len=128, steps=120, lr=3e-3, out=args.model))
    model, tok = _load(args.model)
    sampler = Sampler(seed=args.seed)
    ids = tok.encode(args.prompt or "人工智能") or [0]
    out = []
    for _ in range(args.max_new):
        logits = model.forward(np.array([ids]))[0, -1]
        nxt = sampler.greedy(logits) if args.greedy else sampler.sample(logits, temperature=args.temp)
        ids.append(nxt)
        out.append(tok.decode([nxt]))
        if tok.id2token.get(nxt) == "<eos>":
            break
    print(("".join(out)).strip() or "(空)")


def cmd_generate(args):
    model, tok = _load(args.model)
    sampler = Sampler(seed=args.seed)
    ids = tok.encode(args.prompt) or [0]
    out = []
    for _ in range(args.max_new):
        logits = model.forward(np.array([ids]))[0, -1]
        nxt = sampler.sample(logits, temperature=args.temp, top_p=args.top_p, top_k=args.top_k)
        ids.append(nxt)
        out.append(tok.decode([nxt]))
    print("".join(out))


def cmd_chat(args):
    model, tok = _load(args.model)
    sampler = Sampler(seed=args.seed)
    print("进入聊天 (输入 exit 退出):")
    while True:
        q = input("> ") if not args.seed else "你好"
        if q.strip() == "exit":
            break
        ids = tok.encode(q) or [0]
        for _ in range(args.max_new):
            logits = model.forward(np.array([ids]))[0, -1]
            nxt = sampler.sample(logits, temperature=args.temp)
            ids.append(nxt)
            tok_txt = tok.decode([nxt])
            if tok.id2token.get(nxt) == "<eos>":
                break
            print(tok_txt, end="", flush=True)
        print()


def cmd_serve(args):
    from ..serve.api import OpenAIServer, serve as run_server
    model, tok = _load(args.model)
    srv = OpenAIServer(model, tok, seed=args.seed)
    print(f"服务启动于 http://{args.host}:{args.port} (Ctrl+C 退出)")
    run_server(srv, args.host, args.port)


def cmd_eval(args):
    from ..eval.perplexity import perplexity, perplexity_manual
    from ..eval.harness import GSM8K_Mini
    model, tok = _load(args.model)
    seq = tok.encode(MICRO_CORPUS)[:64] or list(range(8))
    p1 = perplexity(model, seq)
    p2 = perplexity_manual(model, seq)
    print(f"困惑度: {p1:.4f} (手工对照 {p2:.4f}, Δ={abs(p1-p2):.2e})")
    gsm = GSM8K_Mini()
    def ans_fn(q):
        model2, tok2 = model, tok
        ids = tok2.encode(q) or [0]
        for _ in range(12):
            lg = model2.forward(np.array([ids]))[0, -1]
            nxt = Sampler(seed=0).greedy(lg)
            ids.append(nxt)
            if tok2.id2token.get(nxt) == "<eos>":
                break
        return "".join(tok2.decode(ids))
    res = gsm.evaluate(ans_fn)
    print(f"GSM8K-mini 管线正确性: {res['correct']}/{res['total']} (无准确率门槛, 仅验证管线)")


def cmd_search(args):
    docs = [d.strip() for d in MICRO_CORPUS.split("。") if d.strip()]
    bm25 = BM25().fit([d.split() for d in docs])
    res = bm25.search(args.query.split(), top_k=args.top_k)
    for i, s in res:
        print(f"[{s:.3f}] {docs[i]}")


def cmd_agent(args):
    def fake_llm(messages):
        # 演示用确定性策略: 遇到"计算"则调用计算器; 看到工具结果则 Finish
        user = messages[-1]["content"]
        if user.startswith("Observation:"):
            return f"Finish: {user.split('Observation:', 1)[1].strip()}"
        if "计算" in user or any(c in user for c in "+-*/"):
            expr = user.split("计算")[-1].strip()
            return f'Thought: 需要计算。\nAction: calculator(expression="{expr}")'
        return "Finish: 已回答 (演示智能体)"
    reg = default_registry()
    agent = ReActAgent(fake_llm, reg, max_steps=4)
    print(agent.run(args.question))


def cmd_tokenize(args):
    tok = Tokenizer(vocab_size=args.vocab).train(MICRO_CORPUS)
    ids = tok.encode(args.text)
    print("ids:", ids)
    print("decode:", tok.decode(ids))


def cmd_version(_):
    print(f"ifai {__version__} · by 晨星 (Morningstar) / CJX0712")


def cmd_doctor(_):
    import numpy as np
    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__)
    print("ifai:", __version__)
    print("GPU: 否 (CPU 推理路径, 100% numpy)")


def cmd_selftest(args):
    """零重写自验证: 运行三项静态审计 + (可选) pytest 黄金测试。

    INV-41 (分层零反向依赖) / INV-42 (≥25 处世界顶级出处) / R-X-03 (文档完备)。
    """
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    audits = [
        ("分层审计 (INV-41)", "scripts/check_layering.py"),
        ("零重写出处审计 (INV-42)", "scripts/audit_no_reinvention.py"),
        ("文档完备审计 (R-X-03)", "scripts/check_docs.py"),
    ]
    all_ok = True
    print("=== ifai selftest (零重写自验证) ===")
    for label, rel in audits:
        path = os.path.join(repo, rel)
        if not os.path.exists(path):
            print(f"[跳过] {label}: 脚本缺失 {rel}")
            continue
        rc = subprocess.run([sys.executable, path], cwd=repo).returncode
        ok = rc == 0
        all_ok &= ok
        print(f"[{'通过' if ok else '失败'}] {label}")
    if getattr(args, "pytest", False):
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=repo).returncode
        ok = rc == 0
        all_ok &= ok
        print(f"[{'通过' if ok else '失败'}] 黄金测试 (pytest)")
    print("=== 结论:", "ALL GREEN ✅" if all_ok else "存在失败 ❌", "===")
    sys.exit(0 if all_ok else 1)


def _exists(p):
    import os
    return os.path.exists(p)


def main():
    ap = argparse.ArgumentParser(prog="ifai", description="创新旗舰 AI 命令行")
    sub = ap.add_subparsers(dest="cmd")
    for name, fn in [("train", cmd_train), ("demo", cmd_demo), ("generate", cmd_generate),
                     ("chat", cmd_chat), ("serve", cmd_serve), ("eval", cmd_eval),
                     ("search", cmd_search), ("agent", cmd_agent),
                     ("tokenize", cmd_tokenize), ("version", cmd_version),
                     ("doctor", cmd_doctor), ("selftest", cmd_selftest)]:
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
    sub.choices["selftest"].add_argument("--pytest", action="store_true",
                                        help="额外运行 pytest 黄金测试")
    # 通用参数
    for name in ["train"]:
        sub.choices[name].add_argument("--seed", type=int, default=0)
        sub.choices[name].add_argument("--corpus", default=None)
        sub.choices[name].add_argument("--vocab", type=int, default=512)
        sub.choices[name].add_argument("--d-model", type=int, default=128, dest="d_model")
        sub.choices[name].add_argument("--layers", type=int, default=3)
        sub.choices[name].add_argument("--heads", type=int, default=4)
        sub.choices[name].add_argument("--kv-heads", type=int, default=2, dest="kv_heads")
        sub.choices[name].add_argument("--ffn-mult", type=int, default=4, dest="ffn_mult")
        sub.choices[name].add_argument("--seq-len", type=int, default=128, dest="seq_len")
        sub.choices[name].add_argument("--steps", type=int, default=120)
        sub.choices[name].add_argument("--lr", type=float, default=3e-3)
        sub.choices[name].add_argument("--out", default=DEFAULT_MODEL_DIR)
    for name in ["demo", "generate", "chat", "serve", "eval"]:
        sub.choices[name].add_argument("--model", default=DEFAULT_MODEL_DIR)
        sub.choices[name].add_argument("--seed", type=int, default=0)
    for name in ["demo", "generate", "chat"]:
        sub.choices[name].add_argument("--max-new", type=int, default=32, dest="max_new")
        sub.choices[name].add_argument("--temp", type=float, default=0.8)
        sub.choices[name].add_argument("--greedy", action="store_true")
    sub.choices["generate"].add_argument("--prompt", default="人工智能")
    sub.choices["generate"].add_argument("--top-p", type=float, default=0.9, dest="top_p")
    sub.choices["generate"].add_argument("--top-k", type=int, default=0, dest="top_k")
    sub.choices["demo"].add_argument("--prompt", default="人工智能")
    sub.choices["serve"].add_argument("--host", default="127.0.0.1")
    sub.choices["serve"].add_argument("--port", type=int, default=8000)
    sub.choices["search"].add_argument("--query", default="注意力")
    sub.choices["search"].add_argument("--top-k", type=int, default=5, dest="top_k")
    sub.choices["agent"].add_argument("--question", default="计算 (3+5)*2")
    sub.choices["tokenize"].add_argument("--text", default="人工智能 模型")
    sub.choices["tokenize"].add_argument("--vocab", type=int, default=512)

    args = ap.parse_args()
    if not getattr(args, "cmd", None):
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
