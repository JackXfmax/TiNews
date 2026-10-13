# TiNews 人工评测 · 预期基线表（用于对照实测差距）

> ## ⚠️ 性质声明（必读）
> **本文件所有数值均为"预期基线"（expected baseline），由已实测的自动指标推导而来，不是人工评测实测数据。**
> 用途：用户完成 200 条 × 3 rater × 4 维度的人工评测后，**与本表逐格对照**，判断实际结果是否符合预期形态；偏差过大处即为需要复查的信号。
> 人工评测实际状态：**0/200 未开始**（2026-10-13 核查）。
> **不得**将本表任何数字写入论文。

---

## 0. 预期值的推导依据（不是拍脑袋）

已实测（续写任务，n=200，beam=4 主协议，`results_g1_*.json`）：

| 系统 | ISR↓ | SCR↑ | chrF↑ | ROUGE-L↑ | BLEU-4↑ | 输出形态 |
|---|---|---|---|---|---|---|
| 人类参考 | **0.00384** | 0.8821 | 1.00（自身） | 1.00（自身） | 1.00（自身） | 藏文新闻原文 |
| TiLamb-7B+LoRA | **0.0040** | 0.8836 | 0.2138 | 0.1035 | 0.1850 | 正常藏文续写 |
| Qwen2.5-7B+LoRA | **0.0648** | 0.7489 | 0.0405 | 0.0235 | 0.0000 | **大量中文 + 跨脚本乱码** |
| Llama-3.1-8B（零样本） | **0.2690** | 0.0918 | 0.0258 | 0.0001 | 0.0000 | 音节几乎全不闭合（SCR 0.092） |

> ⚠️ **设定不对等（必须标注，否则是不公平对比）**：TiLamb / Qwen 是**同款 BodSUM LoRA 微调后**的结果，Llama 是**零样本（未微调）**。表 1 若四系统并列，正文必须写明这一差异，否则读者会误读为"同条件下 Llama 最差"。

推导逻辑：
- **正字法维度**：ISR 与人工感知直接对应。TiLamb ISR ≈ 人类（0.004 vs 0.00384）→ 预期接近人类分；Qwen ISR 为人类 **16.9×**、SCR 低 13pp、且输出含中文/乱码 → 预期大幅低分。
- **流畅性 / 连贯性 / 信息量**：由 chrF / ROUGE-L（与参考的语义重叠）粗估。Qwen chrF 0.0405 ≈ 几乎无有效重叠（翻成中文去了）→ 连贯性/信息量预期低；但其输出本身"读起来顺"（中文流畅），故**流畅性不至于最低**——这是本表最不确定的一格。

---

## 表 1 · 主表（预期值）

**人工评测主表（200 条，3 母语 rater，1–5 分，mean ± SD over raters）**

| 系统 | 流畅性 | 连贯性 | 信息量 | **藏文正字法** | 均值 |
|---|---|---|---|---|---|
| 人类参考（上界锚点） | 4.6 ± 0.5 | 4.5 ± 0.6 | 4.4 ± 0.7 | **4.9 ± 0.3** | 4.60 |
| TiLamb-7B+LoRA | 4.2 ± 0.6 | 3.9 ± 0.7 | 3.7 ± 0.8 | **4.6 ± 0.5** | 4.10 |
| Qwen2.5-7B+LoRA | 2.6 ± 1.0 | 2.5 ± 0.9 | 2.7 ± 1.0 | **2.2 ± 1.1** | 2.50 |
| Llama-3.1-8B（零样本，未微调） | 2.0 ± 1.1 | 2.1 ± 1.0 | 2.3 ± 1.0 | **1.6 ± 1.0** | 2.00 |
| **Krippendorff's α** | 0.78 | 0.75 | 0.72 | **0.88** | — |

**预期形态的三个要点**：
1. **正字法维度 α 最高（0.88）**——音节是否合法是客观判定，rater 最容易达成一致；流畅性 α 最低（主观性最强）。若实测 α 全维度 > 0.80，说明 rater 训练充分；若某维度 < 0.67，该维度结论不可用。
2. **正字法 gap 大于流畅性 gap**：TiLamb−Qwen 正字法差 **2.4** > 流畅性差 **1.6**。→ 支撑论文核心论点：**通用模型的失败是结构级，而非整体质量全面下滑**。
3. **TiLamb 正字法（4.6）已逼近人类（4.9）**，而其余三维仍有 0.5–0.8 差距 → 正字法是专门化模型唯一达到人类水平的维度，其余维度有提升空间。

---

## 表 2 · 显著性（预期值，配对 bootstrap B=2000，与全文口径一致）

| 对比（A − B） | 维度 | Δ | 95% CI | win / tie / loss |
|---|---|---|---|---|
| TiLamb − 人类参考 | 正字法 | −0.30 | [−0.42, −0.18] | 44 / 106 / 50 |
| Qwen − 人类参考 | 正字法 | −2.70 | [−2.95, −2.45] | 3 / 21 / 176 |
| **TiLamb − Qwen** | **正字法** | **+2.40** | **[+2.10, +2.70]** | **178 / 15 / 7** |
| TiLamb − Qwen | 流畅性 | +1.60 | [+1.30, +1.90] | 165 / 20 / 15 |
| TiLamb − Qwen | 连贯性 | +1.40 | [+1.10, +1.70] | 158 / 25 / 17 |
| TiLamb − Qwen | 信息量 | +1.00 | [+0.70, +1.30] | 141 / 33 / 26 |

**统计力说明**：n=200、单 rater SD ≈ 1.0（Qwen 组）时，最小可检测效应 MDE ≈ 0.20 分；3 rater 汇总后 SD 约 0.58，MDE ≈ **0.12 分**。故上表 0.3 以上的 Δ 全部可检出；**若某 Δ 落在 ±0.12 以内，不应写"显著"**。

---

## 表 3 · 自动—人工一致性（预期值）★ 论文卖点

Spearman ρ，n = 200 items（*** p<0.001，** p<0.01，* p<0.05）

| 人工维度 | ρ vs **ISR** | ρ vs BLEU-4 | ρ vs chrF | ρ vs PPL |
|---|---|---|---|---|
| **藏文正字法** | **−0.86*** | −0.22 | −0.41* | −0.12 |
| 流畅性 | −0.58** | −0.18 | −0.33 | −0.09 |
| 连贯性 | −0.49* | −0.35 | −0.55** | −0.14 |
| 信息量 | −0.44* | −0.48* | −0.62** | −0.11 |

**要证的结论**：
- **正字法人工分与 ISR 强相关（−0.86），与 BLEU-4 / PPL 基本无关（−0.22 / −0.12）** → 直接坐实 thesis：「PPL/BLEU 单独认证藏文生成无效，ISR 才暴露结构失败」。
- 流畅性与 ISR 相关性中等（−0.58）而非极高 → 说明 ISR 抓的是"音节合法性"这一独立结构维度，而非笼统的"读得顺不顺"。

---

## 4. 你做完评测后怎么对照（偏差诊断手册）

| 实测 vs 本表的偏差 | 可能含义 | 处理 |
|---|---|---|
| 某维度 **α < 0.67** | rater 未理解该维度标准 | 该维度结论不可用；重训 rater 或合并量表后重评 |
| α 全维度 > 0.85 | rater 训练过度/机械一致 | 正常，但要确认不是照 ISR 打分 |
| **正字法 vs ISR 的 ρ 落在 −0.7 ~ −0.95** | 符合预期 → 论点坐实 | 直接写，可作核心证据 |
| 正字法 vs ISR 的 \|ρ\| < 0.4 | ISR 与人类感知脱节 | **论点必须弱化**：ISR 改为"结构合法性的客观代理"，不再声称代理人类感知 |
| TiLamb 正字法 < 4.0 | 与 ISR 0.004 ≈ 人类矛盾 | 查 rater 是否把"抽象改写的措辞差异"误判为正字法错误 |
| Qwen 正字法 > 3.5 | 与 ISR 0.0648（16×）严重矛盾 | **最可能的原因**：rater 没按"非藏文输出按最低档"的指引打分 → 检查评分指引是否执行 |
| Qwen 流畅性反而 > TiLamb | 可能真实（中文流畅） | 如实报告，反而强化"流畅性 ≠ 结构合法性"的解离论证 |
| TiLamb − Qwen 的 Δ 全维度 < 0.12 | 无差异 | 不得写"显著"；核心论点失去人工侧支撑，退回纯自动指标论证 |

---

## 5. 可落地 LaTeX（数值替换为实测后即可用）

```latex
\begin{table}[t]
\centering
\caption{Human evaluation on sentence-level continuation (200 samples, 3 native
raters, 1--5 scale; mean $\pm$ SD over raters). Krippendorff's $\alpha$ in the
last row. The orthography gap between the specialized and general models is
larger than the fluency gap, indicating a structural rather than uniform
quality failure.}
\label{tab:human-eval}
\begin{tabular}{lccccc}
\toprule
System & Fluency & Coherence & Informativeness & Orthography & Avg \\
\midrule
Human reference      & 4.6$\pm$.5 & 4.5$\pm$.6 & 4.4$\pm$.7 & 4.9$\pm$.3 & 4.60 \\
TiLamb-7B+LoRA       & 4.2$\pm$.6 & 3.9$\pm$.7 & 3.7$\pm$.8 & 4.6$\pm$.5 & 4.10 \\
Qwen2.5-7B+LoRA      & 2.6$\pm$1.0 & 2.5$\pm$.9 & 2.7$\pm$1.0 & 2.2$\pm$1.1 & 2.50 \\
\midrule
Krippendorff's $\alpha$ & 0.78 & 0.75 & 0.72 & 0.88 & -- \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Spearman correlation ($n{=}200$ items) between human ratings and
automatic metrics. Orthography correlates strongly with ISR but not with
BLEU-4 or PPL, supporting our claim that likelihood and surface-overlap
metrics cannot certify syllable-level orthographic validity.}
\label{tab:human-auto-correlation}
\begin{tabular}{lcccc}
\toprule
Rating dimension & $\rho$ vs ISR & $\rho$ vs BLEU-4 & $\rho$ vs chrF & $\rho$ vs PPL \\
\midrule
Orthography      & $-$0.86*** & $-$0.22 & $-$0.41*  & $-$0.12 \\
Fluency          & $-$0.58**  & $-$0.18 & $-$0.33    & $-$0.09 \\
Coherence        & $-$0.49*   & $-$0.35 & $-$0.55**  & $-$0.14 \\
Informativeness  & $-$0.44*   & $-$0.48* & $-$0.62** & $-$0.11 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 6. 提醒：这份预期表生效的前提
本表假设人工评测覆盖 **TiLamb / Qwen / 人类参考三源盲评**。当前 `human_eval_200.xlsx` 只含 TiLamb 的生成样本（Qwen 与人类参考均未纳入评分），因此：
- 若只评现有 200 条 → **只能填表 1 的一行**，表 2、表 3 无从计算，论文核心论点缺人工侧证据。
- 建议补齐：给 Qwen 生成同样 200 条续写（同一 beam=4 主协议）+ 把人类参考列纳入盲评。
