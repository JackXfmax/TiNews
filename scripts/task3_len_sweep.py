#!/usr/bin/env python
# task3_len_sweep.py
# Theorem 3.2「压缩率—正字法」长度扫：扫摘要生成长度预算 L，测 ISR 随 L 的变化。
#
# 设计要点（诚实性优先）：
#   1) 主口径 = max_new_tokens=T 上限（**允许自然 EOS 提前停止**，不用 min_new_tokens 强撑，
#      避免"超自然长度"伪影把长档位 ISR 人为推高）。
#   2) 每档同时报告 truncation_rate（实际生成 token 数 >= T 的样本比例）——
#      用于识别"截断主导"档位：截断会切出不完整音节，属已知伪影，分析时须单独标注，
#      不得把截断造成的 ISR 上升误读为 Theorem 3.2 的压缩效应。
#   3) 长度口径双报：字符级 compression_char（与 §7.4.2 同口径）+ 音节级 compression_syll。
#   4) 藏文音节切分用 decode_sweep.syll_tokens（按 tsheg），与既有指标同口径。
#
# 运行：
#   python task3_len_sweep.py --model-path <MODEL_ROOT>/models/TiLamb-7B-base \
#       --model-name TiLamb-7B+LoRA --lora-path <MODEL_ROOT>/TiNews/tilamb_lora_bodsum \
#       --test bodsum_test.jsonl --out results_lensweep_tilamb.json --gpu 0 --n 150

import os, sys, json, argparse, time
import numpy as np
import torch

sys.path.insert(0, "<MODEL_ROOT>/TiNews/scripts")
from decode_sweep import isr_scr, syll_tokens
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build_pair(d):
    txt = (d.get("text") or d.get("body") or "").strip()
    smy = (d.get("summary") or d.get("SUM") or "").strip()
    return txt, smy


def gen(tok, model, rows, dev, T, max_src=512, bs=2):
    """以 max_new_tokens=T 上限生成；返回 (hyps, 实际生成 token 数列表)"""
    hyps, ntok = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            chunk = [build_pair(r)[0] + "\nབསྡུས་དོན། " for r in rows[i:i + bs]]
            tok.padding_side = "left"
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_src).to(dev)
            g = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                               max_new_tokens=T, num_beams=4, no_repeat_ngram_size=3,
                               early_stopping=True, pad_token_id=tok.pad_token_id,
                               eos_token_id=tok.eos_token_id)
            new = g[:, enc["input_ids"].shape[1]:]
            for seq in new:
                n = int((seq != tok.pad_token_id).sum())
                ntok.append(n)
            hyps += [s.strip() for s in tok.batch_decode(new, skip_special_tokens=True)]
    return hyps, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--lora-path", default="")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--Ts", default="32,48,64,96,128,192,256")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max-src", type=int, default=512)
    a = ap.parse_args()

    dev = torch.device(f"cuda:{a.gpu}")
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.bfloat16).to(dev)
    if a.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.lora_path)
        print("[lora] loaded ->", a.lora_path, flush=True)

    rows = load_jsonl(a.test)[:a.n]
    Ts = [int(x) for x in a.Ts.split(",") if x.strip()]
    src_syll = [max(1, len(syll_tokens(build_pair(r)[0]))) for r in rows]

    res = {"model": a.model_name, "model_path": a.model_path, "lora_path": a.lora_path,
           "n": len(rows), "Ts": Ts, "bs": a.bs, "max_src": a.max_src,
           "decode": "beam4, no_repeat_ngram=3, early_stopping=True, max_new_tokens=T (natural EOS allowed)",
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "sweep": []}

    for T in Ts:
        t0 = time.time()
        hyps, ntok = gen(tok, model, rows, dev, T, max_src=a.max_src, bs=a.bs)
        _is = [isr_scr(h) for h in hyps]
        isr = float(np.mean([x[0] for x in _is]))
        scr = float(np.mean([x[1] for x in _is]))
        gen_syll = [len(syll_tokens(h)) for h in hyps]
        comp_char = float(np.mean([len(h) / max(1, len(build_pair(r)[0]))
                                   for h, r in zip(hyps, rows)]))
        comp_syll = float(np.mean([g / s for g, s in zip(gen_syll, src_syll)]))
        trunc = float(np.mean([1.0 if n >= T else 0.0 for n in ntok]))
        rec = {"T": T, "isr": isr, "scr": scr,
               "mean_new_tokens": float(np.mean(ntok)),
               "mean_gen_syll": float(np.mean(gen_syll)),
               "compression_char": comp_char, "compression_syll": comp_syll,
               "truncation_rate": trunc, "seconds": round(time.time() - t0, 1),
               "samples": [h[:150] for h in hyps[:3]]}
        res["sweep"].append(rec)
        print(f"[T={T}] isr={isr:.4f} scr={scr:.4f} comp_syll={comp_syll:.4f} "
              f"gen_syll={np.mean(gen_syll):.1f} tok={np.mean(ntok):.1f} "
              f"trunc={trunc:.2f} ({rec['seconds']:.0f}s)", flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("saved ->", a.out, flush=True)


if __name__ == "__main__":
    main()
