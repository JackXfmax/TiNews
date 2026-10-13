#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bootstrap_ppl.py —— CPU 聚合：从逐条 NLL 做 bootstrap，得 PPL 与微调增益的 95% CI（T3）
读取 eval_ppl_peritem.py 输出的 base / lora 两个 json，对每条样本(ce,ntok)做 B 次有放回
重采样，每次重算 PPL=exp(sum ce/sum ntok)，进而 gain=(PPL_base-PPL_lora)/PPL_base。
输出：results_ppl_bootstrap.json
"""
import os, sys, json, math, argparse
import numpy as np

ROOT = os.path.expanduser("~/TiNews")


def load_items(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["items"], d["model"]


def ppl_of(sel_ce, sel_ntok):
    s = sel_ce.sum(); n = sel_ntok.sum()
    return math.exp(s / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "results_ppl_base_peritem.json"))
    ap.add_argument("--lora", default=os.path.join(ROOT, "results_ppl_lora_peritem.json"))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(ROOT, "results_ppl_bootstrap.json"))
    a = ap.parse_args()

    bi, btag = load_items(a.base)
    li, ltag = load_items(a.lora)
    assert len(bi) == len(li), f"sample count mismatch {len(bi)} vs {len(li)}"
    N = len(bi)
    bce = np.array([x["ce"] for x in bi]); bnt = np.array([x["ntok"] for x in bi])
    lce = np.array([x["ce"] for x in li]); lnt = np.array([x["ntok"] for x in li])

    rng = np.random.default_rng(a.seed)
    ppl_b, ppl_l, gain = [], [], []
    for _ in range(a.B):
        idx = rng.integers(0, N, N)
        pb = ppl_of(bce[idx], bnt[idx])
        pl = ppl_of(lce[idx], lnt[idx])
        ppl_b.append(pb); ppl_l.append(pl)
        gain.append((pb - pl) / pb)
    ppl_b = np.array(ppl_b); ppl_l = np.array(ppl_l); gain = np.array(gain)

    def ci(v):
        return {"mean": float(v.mean()), "lo95": float(np.percentile(v, 2.5)),
                "hi95": float(np.percentile(v, 97.5))}

    res = {
        f"ppl_{btag}": ci(ppl_b),
        f"ppl_{ltag}": ci(ppl_l),
        "gain_pct": ci(gain * 100),
        "B": a.B, "n_samples": N,
        "point_ppl_base": float(ppl_of(bce, bnt)),
        "point_ppl_lora": float(ppl_of(lce, lnt)),
        "point_gain_pct": float((ppl_of(bce, bnt) - ppl_of(lce, lnt)) / ppl_of(bce, bnt) * 100),
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: (round(v, 3) if isinstance(v, dict) else v) for k, v in res.items()
                      if not isinstance(v, dict) or k in ()}, ensure_ascii=False, indent=2))
    print(f"[bootstrap] PPL_base={res['point_ppl_base']:.2f} PPL_lora={res['point_ppl_lora']:.2f} "
          f"gain={res['point_gain_pct']:.1f}%  CI gain={res['gain_pct']['lo95']:.1f}~{res['gain_pct']['hi95']:.1f}%")
    print(f"[bootstrap] saved -> {a.out}")


if __name__ == "__main__":
    main()
