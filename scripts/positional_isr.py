"""位置平稳性检验：固定预算下，ISR 是否随生成位置（第 j 个音节）变化。

目的：为 Theorem 3.2' 的假设 (A2)「字形接地的位置平稳性」提供**独立于跨长度档比较**的证据。
跨长度档比较是循环的（假设平稳 -> 推出恒定 -> 再拿恒定当证据）；位置剖面不是：
它问的是"在同一次生成内部，越往后生成，违反率是否越高"。
"""
import json, math, sys
import numpy as np
from syllable_slots_v3 import parse, split_syllables, strip_particle


def syll_legal_flags(text):
    """返回逐音节的合法性 flag 列表（1=违反/非法, 0=合法）。与 isr_scr 同口径。"""
    flags = []
    for syl in split_syllables(text):
        if not syl.strip():
            continue
        core, part = strip_particle(syl)
        if core == "":
            continue  # 空核（纯虚词/标点）不计入，与 isr_scr 的 total 口径一致需确认
        slots, status, _ = parse(core)
        flags.append(1 if status != "OK" else 0)
    return flags


def load_flags(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [syll_legal_flags(r["hyp"]) for r in rows]


def profile(all_flags, binsize=10, maxbin=80):
    """按绝对音节位置分箱，逐箱统计 ISR 与 Wilson 95% CI。"""
    pooled = {}
    for flags in all_flags:
        for j, f in enumerate(flags):
            b = (j // binsize) * binsize
            if b >= maxbin:
                continue
            pooled.setdefault(b, []).append(f)
    return pooled


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    models = {
        "Qwen2.5-7B+LoRA": "hyps_task3a_qwen.jsonl",
        "TiLamb-7B+LoRA": "hyps_task3a_tilamb.jsonl",
    }
    for name, path in models.items():
        all_flags = load_flags(path)
        syll_counts = [len(f) for f in all_flags]
        print("=" * 62)
        print(f"{name}   n_items={len(all_flags)}")
        print(f"  每 hyp 音节数: 中位 {sorted(syll_counts)[len(syll_counts)//2]}, "
              f"均值 {np.mean(syll_counts):.1f}, max {max(syll_counts)}")
        pooled = profile(all_flags, binsize=10, maxbin=80)
        print(f"\n  {'位置箱':<12}{'n_syll':>8}{'ISR':>9}{'95% CI':>22}")
        rows = []
        for b in sorted(pooled):
            arr = np.array(pooled[b])
            n = len(arr)
            isr = arr.mean()
            lo, hi = wilson(int(arr.sum()), n)
            rows.append((b, n, isr, lo, hi))
            print(f"  [{b:>2},{b+10:>2}){'':<4}{n:>8}{isr:>9.4f}   [{lo:.4f}, {hi:.4f}]")

        # 趋势检验：箱序号 vs ISR，加权 Spearman（按 n 加权忽略，直接用 Pearson on (b, isr)）
        bs = np.array([r[0] for r in rows], dtype=float)
        isrs = np.array([r[2] for r in rows])
        ns = np.array([r[1] for r in rows], dtype=float)
        if len(bs) >= 3:
            # 加权 Pearson 相关（权重 n）
            w = ns / ns.sum()
            mb = (w * bs).sum(); mi = (w * isrs).sum()
            cov = (w * (bs - mb) * (isrs - mi)).sum()
            sb = math.sqrt((w * (bs - mb) ** 2).sum())
            si = math.sqrt((w * (isrs - mi) ** 2).sum())
            r = cov / (sb * si) if sb > 0 and si > 0 else float("nan")
            print(f"\n  加权 Pearson(位置箱, ISR) r = {r:+.3f}")
            # 合并检验：前半 vs 后半
            half = max(bs) / 2
            early = np.concatenate([np.array(pooled[b]) for b in pooled if b < half])
            late = np.concatenate([np.array(pooled[b]) for b in pooled if b >= half])
            if len(early) > 30 and len(late) > 30:
                e, l = early.mean(), late.mean()
                se = math.sqrt(e * (1 - e) / len(early) + l * (1 - l) / len(late))
                z = (l - e) / se if se > 0 else 0.0
                print(f"  前半 ISR={e:.4f} (n={len(early)})  后半 ISR={l:.4f} (n={len(late)})"
                      f"  diff={l-e:+.4f}  z={z:+.2f}")


if __name__ == "__main__":
    main()
