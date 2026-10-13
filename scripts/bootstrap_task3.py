#!/usr/bin/env python
# bootstrap_task3.py
# Task 3a 摘要三系统（mT5 / TiLamb+LoRA / Qwen+LoRA）逐条指标的**配对 bootstrap 显著性检验**。
# 输入：hyps_task3a_{mt5,tilamb,qwen}.jsonl（task3_dump_hyps.py 产出，每行 {id,ref,hyp}）。
# 输出：results_bootstrap_task3.json（点估计 + 95% CI + p(>0)）。
#
# ⚠ 口径：isr_scr / chrf_pair 与服务器 decode_sweep 逐字一致（已与 G1 同口径对账）；
#   bleu4 用 corpus_bleu4（音节级、+1 平滑），rougeL 用音节级 LCS。
import json, os, sys, collections, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syllable_slots_v3 import parse, split_syllables, strip_particle

TSHEG = "\u0F0B"
SHAD = "\u0F0D"


def isr_scr(text):
    total = inv = closed = 0
    for syl in split_syllables(text):
        if not syl.strip():
            continue
        core, part = strip_particle(syl)
        total += 1
        if core == "":
            closed += 1
            continue
        slots, status, _ = parse(core)
        if status == "OK":
            if slots.get("vow") or slots.get("suf"):
                closed += 1
        else:
            inv += 1
    if total == 0:
        return 0.0, 0.0
    return inv / total, closed / total


def chrf_pair(ref, hyp, beta=2, nmax=6):
    def cng(s, n):
        s = s.replace(TSHEG, "").replace(SHAD, "")
        return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else [s]
    fs = []
    for n in range(1, nmax + 1):
        rc = collections.Counter(cng(ref, n)); hc = collections.Counter(cng(hyp, n))
        common = sum(min(hc[g], rc[g]) for g in hc)
        tc = sum(hc.values()); tr = sum(rc.values())
        if tc == 0 or tr == 0:
            fs.append(0.0); continue
        P = common / tc; R = common / tr
        fs.append((1 + beta ** 2) * P * R / (beta ** 2 * P + R) if (P + R) > 0 else 0.0)
    return float(np.mean(fs))


def syll_tokens(s):
    return [p for p in s.replace(SHAD, "").split(TSHEG) if p.strip()]


def corpus_bleu4(refs, hyps):
    def ngrams(toks, n):
        if not toks:
            return []
        return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)] or [tuple(toks)]
    R = [syll_tokens(r) for r in refs]; H = [syll_tokens(h) for h in hyps]
    pn = []
    for n in range(1, 5):
        hyp_ng = collections.Counter(); ref_ng = collections.Counter(); ct = 0
        for h, r in zip(H, R):
            for g in ngrams(h, n):
                hyp_ng[g] += 1; ct += 1
            for g in ngrams(r, n):
                ref_ng[g] += 1
        if ct == 0:
            pn.append(0.0); continue
        corr = sum(min(hyp_ng[g], ref_ng.get(g, 0)) for g in hyp_ng)
        pn.append((corr + 1) / (ct + 1))
    hl = sum(len(h) for h in H); rl = sum(len(r) for r in R)
    bp = math.exp(min(0.0, 1 - rl / max(hl, 1)))
    return bp * math.exp(sum(math.log(max(p, 1e-9)) for p in pn) / 4)


def _lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rougeL(ref, hyp):
    a = syll_tokens(ref); b = syll_tokens(hyp)
    if not a or not b:
        return 0.0
    l = _lcs(a, b)
    P = l / len(b); R = l / len(a)
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


def load(name, path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["id"])
    return {
        "name": name,
        "ref": [r["ref"] for r in rows],
        "hyp": [r["hyp"] for r in rows],
        "isr": np.array([isr_scr(r["hyp"])[0] for r in rows]),
        "chrf": np.array([chrf_pair(r["ref"], r["hyp"]) for r in rows]),
        "rouge": np.array([rougeL(r["ref"], r["hyp"]) for r in rows]),
    }


def boot_pair(da, db, metric, B=2000, seed=42):
    """配对 bootstrap：对逐条差值重采样，返回 Δ 与 95% CI 与 p(>0)。"""
    a = da[metric]; b = db[metric]
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_gt0 = float((means > 0).mean())
    return {
        "metric": metric,
        "mean_A": float(a.mean()), "mean_B": float(b.mean()),
        "delta": float(a.mean() - b.mean()),
        "ci95_low": float(lo), "ci95_high": float(hi),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "p_gt0": p_gt0,
    }


def boot_bleu(da, db, B=2000, seed=42):
    """BLEU-4 是 corpus 级指标：重采样 item 集合后各自重算 corpus BLEU 再取差。

    ⚠ 口径修正（2026-10-06）：corpus 级 BLEU 在**有放回重采样**后会系统性偏移——
    重复样本改变 n-gram 总计数与 brevity penalty，使 bootstrap 分布中心 ≠ 全量点估计
    （实测：Δ=0.3297 而 percentile 区间为 [0.285,0.304]，区间甚至不含 Δ）。
    故改用 **basic / reverse-percentile CI**：CI = [2Δ - q97.5, 2Δ - q2.5]，
    使区间围绕点估计 Δ 做偏差修正，而非直接报告有偏的 percentile 区间。
    """
    ra, ha = da["ref"], da["hyp"]; rb, hb = db["ref"], db["hyp"]
    N = len(ha)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(B, N))
    adiff = np.array([corpus_bleu4([ra[i] for i in s], [ha[i] for i in s]) for s in idx])
    bdiff = np.array([corpus_bleu4([rb[i] for i in s], [hb[i] for i in s]) for s in idx])
    d = adiff - bdiff
    q_lo, q_hi = np.percentile(d, [2.5, 97.5])
    A = float(corpus_bleu4(ra, ha)); Bv = float(corpus_bleu4(rb, hb))
    delta = A - Bv
    lo = 2 * delta - float(q_hi)   # basic bootstrap：围绕 Δ 反转分位数
    hi = 2 * delta - float(q_lo)
    return {
        "metric": "bleu4_syll",
        "mean_A": A, "mean_B": Bv,
        "delta": delta,
        "ci95_low": float(lo), "ci95_high": float(hi),
        "ci_type": "basic(reverse-percentile), bias-corrected around delta",
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "p_gt0": float((d > 0).mean()),
    }


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    files = {
        "mT5-base": os.path.join(base, "hyps_task3a_mt5.jsonl"),
        "TiLamb-7B+LoRA": os.path.join(base, "hyps_task3a_tilamb.jsonl"),
        "Qwen2.5-7B+LoRA": os.path.join(base, "hyps_task3a_qwen.jsonl"),
    }
    for name, p in files.items():
        if not os.path.exists(p):
            print(f"MISSING {name}: {p}")
            sys.exit(1)
    M = {k: load(k, v) for k, v in files.items()}

    print("=" * 78)
    print("点估计（与服务器 results_task3a_*.json 对账）")
    for k in M:
        print(f"  {k:<18} ISR={M[k]['isr'].mean():.4f}  chrF={M[k]['chrf'].mean():.4f}  "
              f"ROUGE-L={M[k]['rouge'].mean():.4f}  BLEU-4={corpus_bleu4(M[k]['ref'], M[k]['hyp']):.4f}")

    pairs = [("TiLamb-7B+LoRA", "mT5-base"),
             ("TiLamb-7B+LoRA", "Qwen2.5-7B+LoRA"),
             ("mT5-base", "Qwen2.5-7B+LoRA")]
    out = {"point_estimates": {k: {"isr": float(M[k]["isr"].mean()),
                                   "chrf": float(M[k]["chrf"].mean()),
                                   "rougeL": float(M[k]["rouge"].mean()),
                                   "bleu4": float(corpus_bleu4(M[k]["ref"], M[k]["hyp"]))}
                              for k in M},
           "paired_bootstrap": []}
    for a, b in pairs:
        print("\n" + "=" * 78)
        print(f"{a}  vs  {b}")
        for metric in ["isr", "chrf", "rouge"]:
            r = boot_pair(M[a], M[b], metric)
            sig = "显著" if r["ci_excludes_zero"] else "不显著"
            print(f"  {metric:<6} Δ={r['delta']:+.4f}  CI[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}]  "
                  f"不含0={r['ci_excludes_zero']}  p(>0)={r['p_gt0']:.3f}  → {sig}")
            out["paired_bootstrap"].append({"A": a, "B": b, **r})
        rb_ = boot_bleu(M[a], M[b])
        sig = "显著" if rb_["ci_excludes_zero"] else "不显著"
        print(f"  bleu4  Δ={rb_['delta']:+.4f}  CI[{rb_['ci95_low']:+.4f},{rb_['ci95_high']:+.4f}]  "
              f"不含0={rb_['ci_excludes_zero']}  p(>0)={rb_['p_gt0']:.3f}  → {sig}")
        out["paired_bootstrap"].append({"A": a, "B": b, **rb_})

    with open(os.path.join(base, "results_bootstrap_task3.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nwritten results_bootstrap_task3.json")


if __name__ == "__main__":
    main()
