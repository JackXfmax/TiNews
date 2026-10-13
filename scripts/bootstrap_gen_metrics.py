#!/usr/bin/env python
# bootstrap_gen_metrics.py
# 对续写（G1）的四模型逐条生成做 **配对 bootstrap**（B=2000），给出 ISR / chrF 组间差异的 95% CI。
#
# 为什么做：核心 claim「通用模型 ISR 远高于藏文专门化」此前只有点估计，无显著性检验。
# 按用户红线（没 bootstrap 证实不得写"显著"），必须补。
#
# 口径自检：先用本脚本复算各模型 mean ISR / chrF，与服务器 `results_g1_*.json` 的
# 已知值对账（TiLamb-LoRA: isr 0.0040 / chrf 0.2138；Qwen: 0.0648 / 0.0405；
# Llama: 0.269 / 0.0258；TiLamb-base: 0.0157 / 0.2049）。若吻合 → 口径一致，bootstrap 可信。
#
# 运行：python bootstrap_gen_metrics.py

import json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syllable_slots_v3 import parse, split_syllables, strip_particle

FILES = {
    "TiLamb-LoRA": "results_g1_tilamb_lora_hyps.jsonl",
    "TiLamb-base": "results_g1_tilamb_base_hyps.jsonl",
    "Qwen": "results_g1_qwen_hyps.jsonl",
    "Llama": "results_g1_llama_hyps.jsonl",
}
# 服务器已知值（用于对账）
KNOWN = {
    "TiLamb-LoRA": {"isr": 0.0040, "chrf": 0.2138},
    "TiLamb-base": {"isr": 0.0157, "chrf": 0.2049},
    "Qwen": {"isr": 0.0648, "chrf": 0.0405},
    "Llama": {"isr": 0.2690, "chrf": 0.0258},
}


def isr_scr(text):
    """与服务器 decode_sweep.isr_scr 同实现"""
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


def _ngrams(s, n):
    return [s[i:i + n] for i in range(len(s) - n + 1)]


TSHEG = "\u0F0B"   # ་
SHAD = "\u0F0D"    # །


def chrf_pair(ref, hyp, beta=2, nmax=6):
    """**逐字复制服务器 decode_sweep.chrf_pair**，保证与既有数字同口径。
    ⚠ 注意（如实标注）：该实现与标准 chrF 有两点差异——
      (1) 先剔除 tsheg ་ 与 shad ། 再取字符 n-gram；
      (2) 逐阶 n 先算 F_beta 再对 n 取平均（标准 chrF 是先宏平均 P/R 再算 F）。
    为与历史结果可比，此处沿用服务器口径；若投稿需改标准 chrF，须全表重算。"""
    def cng(s, n):
        s = s.replace(TSHEG, "").replace(SHAD, "")
        return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else [s]
    fs = []
    for n in range(1, nmax + 1):
        rc = collections.Counter(cng(ref, n))
        hc = collections.Counter(cng(hyp, n))
        common = sum(min(hc[g], rc[g]) for g in hc)
        tc = sum(hc.values()); tr = sum(rc.values())
        if tc == 0 or tr == 0:
            fs.append(0.0); continue
        P = common / tc; R = common / tr
        fs.append((1 + beta ** 2) * P * R / (beta ** 2 * P + R) if (P + R) > 0 else 0.0)
    return float(np.mean(fs))


def load(name):
    p = FILES[name]
    rows = json.load(open(p, encoding="utf-8"))   # 整个文件是一行的 list
    return {int(r["id"]): r for r in rows}


def main():
    data = {}
    print("=" * 78)
    print("口径自检：本地复算 vs 服务器已知值")
    print(f"{'模型':<14}{'ISR(本地)':>11}{'ISR(服务器)':>13}{'chrF(本地)':>12}{'chrF(服务器)':>13}")
    for name in FILES:
        d = load(name)
        ids = sorted(d.keys())
        refs = [d[i]["reference"] for i in ids]
        gens = [d[i]["generated"] for i in ids]
        isrs = np.array([isr_scr(g)[0] for g in gens])
        chrfs = np.array([chrf_pair(r, g) for r, g in zip(refs, gens)])
        data[name] = {"ids": ids, "isr": isrs, "chrf": chrfs}
        k = KNOWN[name]
        print(f"{name:<14}{isrs.mean():>11.4f}{k['isr']:>13.4f}"
              f"{chrfs.mean():>12.4f}{k['chrf']:>13.4f}")

    print("\n（若上表两列接近 → 本地口径与服务器一致，下列 bootstrap 可信）")

    # 配对 bootstrap
    B = 2000
    rng = np.random.default_rng(42)
    base = "TiLamb-LoRA"
    print("\n" + "=" * 78)
    print(f"配对 bootstrap（B={B}，seed=42）—— 基线 = {base}")
    print(f"{'对比':<20}{'ΔISR':>10}{'95% CI':>26}{'含0?':>8}")
    print("-" * 78)
    out = {}
    for name in ["Qwen", "Llama", "TiLamb-base"]:
        a = data[base]["isr"]
        b = data[name]["isr"]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        diff = b - a           # 对方 - TiLamb-LoRA（ISR 越低越好，正=对方更差）
        idx = rng.integers(0, n, size=(B, n))
        boots = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ci_contains_zero = (lo <= 0 <= hi)
        print(f"ISR: {name} - {base:<8}{diff.mean():>10.4f}"
              f"   [{lo:>8.4f}, {hi:>8.4f}]{'是' if ci_contains_zero else '否':>8}")
        out[f"isr_{name}_minus_{base}"] = {"mean": float(diff.mean()),
                                           "ci95": [float(lo), float(hi)],
                                           "ci_contains_zero": bool(ci_contains_zero)}

    print(f"\n{'对比':<20}{'ΔchrF':>10}{'95% CI':>26}{'含0?':>8}")
    print("-" * 78)
    for name in ["Qwen", "Llama", "TiLamb-base"]:
        a = data[base]["chrf"]
        b = data[name]["chrf"]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        diff = a - b           # TiLamb-LoRA - 对方（chrF 越高越好，正=LoRA 更好）
        idx = rng.integers(0, n, size=(B, n))
        boots = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ci_contains_zero = (lo <= 0 <= hi)
        print(f"chrF: {base} - {name:<6}{diff.mean():>10.4f}"
              f"   [{lo:>8.4f}, {hi:>8.4f}]{'是' if ci_contains_zero else '否':>8}")
        out[f"chrf_{base}_minus_{name}"] = {"mean": float(diff.mean()),
                                            "ci95": [float(lo), float(hi)],
                                            "ci_contains_zero": bool(ci_contains_zero)}

    json.dump(out, open("results_bootstrap_gen.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nsaved -> results_bootstrap_gen.json")
    print("判读：CI 不含 0 → 差异在 95% 置信水平下显著，可写'显著'；含 0 → 只能描述性报告。")


if __name__ == "__main__":
    main()
