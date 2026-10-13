#!/usr/bin/env python
# human_trunc_control.py
# 【截断伪影对照实验】CPU-only，无需 GPU。
#
# 问题：长度扫里所有档位 truncation_rate=1.00（生成全部撞上 max_new_tokens 上限被截断）。
#       截断会切出不完整音节 → ISR 虚高。所以"越短 ISR 越高"可能只是截断伪影，
#       而非 Theorem 3.2 预测的"压缩加剧 → 解码器进入低概率音节区 → 正字法崩"。
#
# 对照设计：把**人类参考摘要**按同样长度做两种截断，测其 ISR ——
#   (a) 音节边界截断（clean）：只压长度，不切坏音节 → 隔离"纯长度效应"
#   (b) 字符级硬截断（hard）：模拟 token 级截断，末尾音节会被切坏 → 量化"截断伪影"上限
# 判读：
#   - 若 human_hard 的 ISR 已接近模型 ISR → 模型的 ISR 上升**主要是截断伪影**，
#     Theorem 3.2 的速率形式**不能**用这批数据支撑（须另做无截断实验）。
#   - 若 human_hard 的 ISR 仍远低于模型 ISR（≈人类基线 0.004）→ 说明同样的机械截断
#     不足以解释模型的 ISR 抬升 → 支持"真压缩效应"。
#
# 运行：python human_trunc_control.py results_lensweep_tilamb.json [data/task3/bodsum_test.jsonl]

import json, sys, os

TSHEG = "\u0F0B"   # ་
SHAD = "\u0F0D"    # །

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syllable_slots_v3 import parse, split_syllables, strip_particle


def syll_tokens(s):
    """与服务器 decode_sweep.syll_tokens 同实现"""
    parts = s.replace(SHAD, "").split(TSHEG)
    return [p for p in parts if p.strip()]


def isr_scr(text):
    """与服务器 decode_sweep.isr_scr 同实现（口径必须一致）"""
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


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build_pair(d):
    txt = (d.get("text") or d.get("body") or "").strip()
    smy = (d.get("summary") or d.get("SUM") or "").strip()
    return txt, smy


def trunc_clean(ref, n_syll):
    """按音节边界截断：取前 n_syll 个音节块，用 tsheg 连接"""
    parts = syll_tokens(ref)
    if len(parts) <= n_syll:
        return ref
    return TSHEG.join(parts[:n_syll])


def trunc_hard(ref, n_chars):
    """字符级硬截断：直接切前 n_chars 个字符（模拟 token 截断切坏末尾音节）"""
    if len(ref) <= n_chars:
        return ref
    return ref[:n_chars]


def main():
    sweep_path = sys.argv[1] if len(sys.argv) > 1 else "results_lensweep_tilamb.json"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "data/task3/bodsum_test.jsonl"

    d = json.load(open(sweep_path, encoding="utf-8"))
    rows = load_jsonl(data_path)
    refs = [build_pair(r)[1] for r in rows]

    # 参考摘要自身的基线（未截断）
    base = [isr_scr(r) for r in refs]
    base_isr = sum(x[0] for x in base) / len(base)
    print(f"模型: {d['model']}   n(sweep)={d['n']}   参考摘要条数={len(refs)}")
    print(f"人类参考摘要**未截断**基线 ISR = {base_isr:.4f}")
    print("\n对照目标：把人类参考截断到与模型各档**相同生成长度**，看其 ISR")
    print(f"{'T':>5} {'gen_syll':>9} {'模型ISR':>9} | {'人类clean截':>11} {'人类hard截':>11} | 伪影能否解释?")
    print("-" * 78)

    verdicts = []
    for r in d["sweep"]:
        L = r["mean_gen_syll"]
        m_isr = r["isr"]
        # clean：按音节数截
        cl = [isr_scr(trunc_clean(ref, int(round(L))))[0] for ref in refs]
        cl_isr = sum(cl) / len(cl)
        # hard：按字符数截（目标字符数 = L * 平均音节字符数）
        cps = [len(ref) / max(1, len(syll_tokens(ref))) for ref in refs]
        avg_cps = sum(cps) / len(cps)
        target_chars = int(round(L * avg_cps))
        hd = [isr_scr(trunc_hard(ref, target_chars))[0] for ref in refs]
        hd_isr = sum(hd) / len(hd)

        if hd_isr >= m_isr * 0.8:
            v = "伪影可解释（模型ISR≈人类硬截）"
        elif hd_isr < m_isr * 0.5:
            v = "伪影不足以解释 → 支持真压缩效应"
        else:
            v = "部分解释（需谨慎）"
        verdicts.append((r["T"], L, m_isr, cl_isr, hd_isr, v))
        print(f"{r['T']:>5} {L:>9.1f} {m_isr:>9.4f} | {cl_isr:>11.4f} {hd_isr:>11.4f} | {v}")

    print("\n判读规则：hard 截（模拟 token 截断切坏音节）是截断伪影的**上界估计**。")
    print("  若 人类hard ISR 已 ≥ 0.8×模型ISR → 模型 ISR 抬升主要来自截断，不能支撑 Theorem 3.2；")
    print("  若 人类hard ISR < 0.5×模型ISR → 同样机械截断解释不了模型，支持真压缩效应。")


if __name__ == "__main__":
    main()
