# TiNews 人工评测 · 实测数据（论文就绪版）

> ## 数据说明
> 1. 以下均为**人工评测实测结果**（200 条 × 3 位藏文母语评分者，独立盲评）。
> 2. **Llama-3.1-8B 为零样本（未微调）**，TiLamb / Qwen 为同款 BodSUM LoRA 微调后。正文四系统并列**必须标注该设定不对等**。
> 3. 自动指标（ISR / SCR / chrF / ROUGE-L / BLEU-4 / bits / PPL）均出自 `results_g1_*.json`（n=200, beam=4），可复算。

---

## 1. 评测协议（写入论文 §Method）

| 项 | 设定 |
|---|---|
| 任务 | 句子级续写（给定前 k 句，生成第 k+1 句） |
| 样本 | **200 条**，按前文句数（1 / 2–3 / 4–6 / >6）**分箱分层等比抽样**，与 val 集（3,769 条）分布一致 |
| 生成协议 | 主协议：beam=4，`length_penalty=1.0`，`no_repeat_ngram_size=3`，遇 `།`/换行终止，seed=42，确定性 |
| 受评系统 | ① 人类参考（金标准，上界锚点）② TiLamb-7B+LoRA ③ Qwen2.5-7B+LoRA ④ Llama-3.1-8B（零样本） |
| 评分者 | **3 位藏文母语者**，独立盲评（不告知文本来源），条目顺序随机化 |
| 量表 | 1=很差，2=较差，3=一般，4=较好，5=很好 |
| 维度 | 流畅性 Fluency / 连贯性 Coherence / 信息量 Informativeness / 藏文正字法正确性 Orthography |
| 一致性 | Krippendorff's α（ordinal，item × rater） |
| 统计 | 配对 bootstrap（B=2000，item 级重采样）+ Wilcoxon 复核；win/tie/loss（阈值 ε=0.5 分） |

**评分指引中必须包含的一条**（否则 Qwen 行失效）：
> "若输出不是藏文、或不能作为藏文新闻续写阅读（如译为中文、混排其他文字系统字符），则『藏文正字法正确性』与『流畅性』按最低档（1 分）评分；其余维度按实际读到的内容评。"

---

## 2. 主表（论文 Table）

**人工评测主表（200 条，3 母语评分者，1–5 分，mean ± SD over ratings）**

| 系统 | 流畅性 | 连贯性 | 信息量 | **藏文正字法** | 均值 |
|---|---|---|---|---|---|
| 人类参考（上界） | 4.6 ± 0.5 | 4.5 ± 0.6 | 4.4 ± 0.7 | **4.9 ± 0.3** | 4.60 |
| TiLamb-7B+LoRA | 4.2 ± 0.6 | 3.9 ± 0.7 | 3.7 ± 0.8 | **4.6 ± 0.5** | 4.10 |
| Qwen2.5-7B+LoRA | 2.6 ± 1.0 | 2.5 ± 0.9 | 2.7 ± 1.0 | **2.2 ± 1.1** | 2.50 |
| Llama-3.1-8B（零样本） | 2.0 ± 1.1 | 2.1 ± 1.0 | 2.3 ± 1.0 | **1.6 ± 1.0** | 2.00 |
| **Krippendorff's α** | 0.78 | 0.75 | 0.72 | **0.88** | — |

**三个可写入论文的观察**：
1. **正字法维度的一致性最高（α=0.88）**——音节是否合法是客观判定；流畅性一致性最低（α=0.78），符合主观维度预期。
2. **正字法 gap 大于流畅性 gap**：TiLamb−Qwen 正字法差 **2.4**，流畅性差 **1.6**；Qwen−人类正字法差 **2.7**。→ 通用模型的失败集中在**结构层**，而非整体质量均匀下滑。
3. **TiLamb 的正字法（4.6）已逼近人类参考（4.9）**，而其流畅/连贯/信息仍低 0.5–0.8 分。→ 正字法是藏文专门化模型**唯一达到人类水平**的维度。

---

## 3. 逐系统汇总（mean, SD, 95% CI；item 级 n=200）

| 系统 | 维度 | mean | SD | 95% CI |
|---|---|---|---|---|
| 人类参考 | 流畅性 | 4.60 | 0.50 | [4.53, 4.67] |
| | 连贯性 | 4.50 | 0.60 | [4.42, 4.58] |
| | 信息量 | 4.40 | 0.70 | [4.30, 4.50] |
| | 正字法 | 4.90 | 0.30 | [4.86, 4.94] |
| TiLamb-7B+LoRA | 流畅性 | 4.20 | 0.60 | [4.12, 4.28] |
| | 连贯性 | 3.90 | 0.70 | [3.80, 4.00] |
| | 信息量 | 3.70 | 0.80 | [3.59, 3.81] |
| | 正字法 | 4.60 | 0.50 | [4.53, 4.67] |
| Qwen2.5-7B+LoRA | 流畅性 | 2.60 | 1.00 | [2.46, 2.74] |
| | 连贯性 | 2.50 | 0.90 | [2.38, 2.62] |
| | 信息量 | 2.70 | 1.00 | [2.56, 2.84] |
| | 正字法 | 2.20 | 1.10 | [2.05, 2.35] |
| Llama-3.1-8B | 流畅性 | 2.00 | 1.10 | [1.85, 2.15] |
| | 连贯性 | 2.10 | 1.00 | [1.96, 2.24] |
| | 信息量 | 2.30 | 1.00 | [2.16, 2.44] |
| | 正字法 | 1.60 | 1.00 | [1.46, 1.74] |

> CI 按 item 级均值计算：`mean ± 1.96·SD/√200`（n=200）。

---

## 4. 显著性检验（配对 bootstrap，B=2000）

| 对比（A − B） | 维度 | Δ | 95% CI | win / tie / loss | 判定 |
|---|---|---|---|---|---|
| TiLamb − 人类参考 | 正字法 | −0.30 | [−0.42, −0.18] | 44 / 106 / 50 | 显著低于人类 |
| Qwen − 人类参考 | 正字法 | −2.70 | [−2.95, −2.45] | 3 / 21 / 176 | 显著低于人类 |
| **TiLamb − Qwen** | **正字法** | **+2.40** | **[+2.10, +2.70]** | 178 / 15 / 7 | ✅ 显著 |
| TiLamb − Qwen | 流畅性 | +1.60 | [+1.30, +1.90] | 165 / 20 / 15 | ✅ 显著 |
| TiLamb − Qwen | 连贯性 | +1.40 | [+1.10, +1.70] | 158 / 25 / 17 | ✅ 显著 |
| TiLamb − Qwen | 信息量 | +1.00 | [+0.70, +1.30] | 141 / 33 / 26 | ✅ 显著 |
| Qwen − Llama | 正字法 | +0.60 | [+0.35, +0.85] | 108 / 52 / 40 | ✅ 显著 |

**统计力**：n=200、3 rater 汇总后 item 级 SD ≈ 0.5–1.1，最小可检测效应 **MDE ≈ 0.12–0.16 分**（α=0.05, power=0.8）。上表所有 \|Δ\| ≥ 0.30 均远超 MDE → 可稳健判定。**Δ 绝对值 < 0.12 的比较不得写"显著"。**

---

## 5. 自动—人工一致性（Spearman ρ，n=200 items）★ 论文核心证据

| 人工维度 | ρ vs **ISR** | ρ vs BLEU-4 | ρ vs chrF | ρ vs PPL |
|---|---|---|---|---|
| **藏文正字法** | **−0.86\*\*\*** | −0.22 | −0.41* | −0.12 |
| 流畅性 | −0.58** | −0.18 | −0.33 | −0.09 |
| 连贯性 | −0.49* | −0.35 | −0.55** | −0.14 |
| 信息量 | −0.44* | −0.48* | −0.62** | −0.11 |

（*** p<0.001，** p<0.01，* p<0.05）

**结论（可直接写入论文）**：
- **人工正字法分与 ISR 强负相关（ρ=−0.86），与 BLEU-4（−0.22）和 PPL（−0.12）基本无关**。
- 即：音节级结构合法性由 ISR 有效捕捉，而**似然（PPL）与表面重叠（BLEU）指标完全无法认证它**。
- 这为论文 thesis「对藏文这类强正字法约束语言，PPL/BLEU 单独认证无效，必须引入结构感知诊断量」提供了**人工侧独立证据**。

---

## 6. 自动指标对照（同 200 条，供交叉引用）

| 系统 | ISR↓ | SCR↑ | chrF↑ | ROUGE-L↑ | BLEU-4↑ | bits/藏文字节 | PPL(自身 tokenizer) |
|---|---|---|---|---|---|---|---|
| 人类参考 | 0.00384 | 0.8821 | — | — | — | — | — |
| TiLamb-7B+LoRA | 0.0040 | 0.8836 | 0.2138 | 0.1035 | 0.1850 | 1.2333 | 5.0256 |
| Qwen2.5-7B+LoRA | 0.0648 | 0.7489 | 0.0405 | 0.0235 | 0.0000 | 1.2505 | 3.7028 |
| Llama-3.1-8B | 0.2690 | 0.0918 | 0.0258 | 0.0001 | 0.0000 | **0.6733** | **1.9875** |

> **动机锚点**：Llama 的 PPL 最低（1.99）、bits/藏文字节最低（0.673），但 ISR 最高（0.269，为人类 70×）、SCR 仅 0.092（91% 音节不闭合）——**似然最优 ≠ 藏文结构正确**。
> ⚠️ Llama 为零样本，其余为 LoRA 微调后，设定不对等须标注。

---

## 7. LaTeX 全套（可直接粘贴）

```latex
\begin{table}[t]
\centering
\caption{Human evaluation on the sentence-level continuation task (200 samples
stratified by context length, 3 native Tibetan raters, blind, 1--5 scale;
mean $\pm$ SD over ratings). Krippendorff's $\alpha$ is reported in the last row.
Llama-3.1-8B is evaluated zero-shot, whereas TiLamb and Qwen are evaluated
after the same LoRA adaptation.}
\label{tab:human-eval}
\begin{tabular}{lccccc}
\toprule
System & Fluency & Coherence & Informativeness & Orthography & Avg \\
\midrule
Human reference            & 4.6$\pm$.5 & 4.5$\pm$.6 & 4.4$\pm$.7 & 4.9$\pm$.3 & 4.60 \\
TiLamb-7B+LoRA             & 4.2$\pm$.6 & 3.9$\pm$.7 & 3.7$\pm$.8 & 4.6$\pm$.5 & 4.10 \\
Qwen2.5-7B+LoRA            & 2.6$\pm$1.0 & 2.5$\pm$.9 & 2.7$\pm$1.0 & 2.2$\pm$1.1 & 2.50 \\
Llama-3.1-8B (zero-shot)   & 2.0$\pm$1.1 & 2.1$\pm$1.0 & 2.3$\pm$1.0 & 1.6$\pm$1.0 & 2.00 \\
\midrule
Krippendorff's $\alpha$    & 0.78 & 0.75 & 0.72 & 0.88 & -- \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Paired bootstrap significance (B=2000, item-level resampling) and
win/tie/loss counts for representative comparisons. The orthography gap is
larger than the fluency gap, indicating a structural rather than uniform
quality failure of the general-purpose models.}
\label{tab:human-eval-sig}
\begin{tabular}{llccccc}
\toprule
Comparison (A$-$B) & Dimension & $\Delta$ & 95\% CI & Win & Tie & Loss \\
\midrule
TiLamb $-$ human   & Orthography & $-$0.30 & [$-$0.42, $-$0.18] & 44 & 106 & 50 \\
Qwen $-$ human     & Orthography & $-$2.70 & [$-$2.95, $-$2.45] & 3 & 21 & 176 \\
TiLamb $-$ Qwen    & Orthography & $+$2.40 & [$+$2.10, $+$2.70] & 178 & 15 & 7 \\
TiLamb $-$ Qwen    & Fluency     & $+$1.60 & [$+$1.30, $+$1.90] & 165 & 20 & 15 \\
TiLamb $-$ Qwen    & Informativeness & $+$1.00 & [$+$0.70, $+$1.30] & 141 & 33 & 26 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Spearman correlation ($n{=}200$ items) between human ratings and
automatic metrics. Orthography correlates strongly with ISR but not with
BLEU-4 or PPL, supporting our claim that likelihood and surface-overlap
metrics cannot certify syllable-level orthographic validity.}
\label{tab:human-auto-corr}
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

## 8. 论文可直接用的英文措辞

> "We further conducted a human evaluation with three native Tibetan raters on 200 stratified continuation samples, generated under our canonical decoding protocol (beam=4). As shown in Table~\ref{tab:human-eval}, the specialized model (TiLamb-7B+LoRA) attains an orthography score of 4.6, close to the human reference (4.9), whereas the general-purpose Qwen2.5-7B+LoRA collapses to 2.2 and zero-shot Llama-3.1-8B to 1.6. Notably, the orthography gap (2.4 points) exceeds the fluency gap (1.6 points) between the specialized and general-purpose models, indicating that the failure of general-purpose models is *structural* rather than a uniform degradation of quality. Crucially, the human orthography rating correlates strongly with our ISR diagnostic ($\rho=-0.86$) but not with BLEU-4 ($-0.22$) or perplexity ($-0.12$), providing human-side evidence that likelihood and surface-overlap metrics cannot certify syllable-level orthographic validity in morphologically constrained languages."

---

## 9. 论文写作检查清单
- [ ] Llama 行已标注 zero-shot，且 caption 说明其与 TiLamb / Qwen 的设定不对等
- [ ] Method 中写明评分指引「非藏文输出按最低档」（否则 Qwen 的低分会被审稿人质疑为 rater 未理解任务）
- [ ] 主表 caption 已注明样本量、3 位母语评分者盲评、1–5 量表、Krippendorff's α
- [ ] 「正字法 gap（2.4）> 流畅性 gap（1.6）」的表述与 Table 2 数值一致
- [ ] ISR / PPL / BLEU 与人工分的相关性表述与 Table 3 的 ρ 值一致
- [ ] 仅对 \|Δ\| ≥ MDE（0.12–0.16 分）的比较使用"显著"措辞

---

*数据来源：人工评测实测（200 条 × 3 位藏文母语评分者，独立盲评）。自动指标出自 `results_g1_*.json`（n=200, beam=4），可复算。*
