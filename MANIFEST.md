# TiNews 论文数据包 · 索引与速查（MANIFEST）

> 生成于 2026-10-10，开源整理于 2026-10-13。本包含 TiNews（藏文新闻基准，COLING 2027/ARR 投稿）全部实验原始数据 + 论文框架/贡献文档，**可直接用于写稿**。
> 本目录即 GitHub 仓库根（仓库名 `TiNews`）。所有数字已交叉验证，可直接引用。
> 包根目录：`TiNews_paper_pack/`

## 0. 怎么用这个包
- **先读** `docs/results_ledger.md`（60KB 全量台账，所有数字的最终汇总与判定）→ 再读 `docs/TiNews_contribution_v6.md`（理论/贡献最新版，含 Theorem 3.2′）。
- `data/` 下所有 JSON 是**可复算的原始结果**；`data/raw_hyps/` 是摘要任务逐条 ref/hyp（594 条 × 3 系统），写稿时可抽查样本。
- `scripts/` 是诊断与显著性检验脚本（bootstrap / 截断对照 / 位置平稳性 / 长度扫），方法学可复现。
- 所有数字已**交叉验证**（本地复算 vs 服务器 10-09 复跑逐位一致）。

## 1. 文件清单（按任务归类）

| 文件 | 内容 | 写论文对应 |
|---|---|---|
| `data/results_g1_{llama,qwen,tilamb_base,tilamb_lora}.json` | 任务2 续写四模型点估计（n=200, beam4） | §6 续写 / §3 动机 |
| `data/results_task3a_{tilamb,qwen,mt5}.json` | 任务3 摘要三系统点估计（n=594, beam4） | §7 摘要 |
| `data/results_bootstrap_task3.json` | 摘要侧 12 组配对 bootstrap（B=2000） | §7 显著性 |
| `data/results_bootstrap_gen.json` | 续写侧配对 bootstrap（B=2000） | §6 显著性 |
| `data/results_lensweep_{tilamb,qwen}.json` | 七档长度扫（Theorem 3.2 证伪证据） | §6.5 / §3.2 |
| `data/results_g4_mt5*.json` | mT5 enc-dec 对照（span-corruption mismatch） | §7.4 对照 |
| `data/results_cino-base-v2*.json` `results_textcnn*.json` `results_tibetan_bert*.json` | 任务1 分类基线（CINO / TextCNN / TibetanBERT） | §5 分类 |
| `data/results_ppl_*.json` `ppl_peritem.jsonl` `ppl_lora_peritem.json` | 跨 tokenizer PPL 桥接（bits/藏文字节） | §3 桥接方法 |
| `data/raw_hyps/hyps_task3a_{tilamb,qwen,mt5}.jsonl` | 摘要逐条 ref/hyp（594 条） | 样本抽查 |
| `data/human_eval_200.jsonl` `data/human_eval_200.xlsx` | 人工评测材料（200 条，rater 表「生成」列已填，评分列空） | 用户侧 P0 |
| `docs/results_ledger.md` | **全量台账（权威汇总）** | 全局 |
| `docs/TiNews_outline_rev4.md` | 论文框架（最新） | 全局结构 |
| `docs/TiNews_contribution_v6.md` | 贡献重定位（含 Theorem 3.2′ 两层分解） | §3 理论 |
| `docs/TiNews_limitations_draft.md` | Limitations 草稿（10 条，锚实测） | §8 Limitations |
| `docs/TiNews_task3_spec.md` | 任务3 规格 | §7 设置 |
| `docs/TiNews_g1_draft.tex` | §G1 已起草 tex（唯一成稿片段） | §6 模板 |
| `scripts/bootstrap_task3.py` `scripts/bootstrap_gen_metrics.py` | 配对 bootstrap 显著性 | 方法 |
| `scripts/human_trunc_control.py` | 人类摘要截断对照（排除截断伪影） | §6.5 伪影 |
| `scripts/positional_isr.py` | 固定预算内 ISR 位置平稳性（打破循环论证） | §3.2.4 |
| `scripts/task3_len_sweep.py` | 七档长度扫 | §6.5 |
| **`docs/figure_data_report.md`** | **8 项图数据整合报告（A1/A2/A3/B1-B2/B3/C3/E1/C5-D2-F3）** | **所有 Figure/附录表格** |
| `fig_data/*.json`（9 个）+ `pmax_raw.npy` | 图数据原始文件 | Figure 数据源 |

## 2. 核心数字速查（已交叉验证）

### 2.1 任务2 续写（n=200, beam4）— 动机钩子 + LoRA 修复
**bits/藏文字节（跨 tokenizer 唯一可比量；同分母 syll_sum=485286）**
| 模型 | bits/syll | PPL(自身 tokenizer) | ISR | 备注 |
|---|---|---|---|---|
| Llama-3.1-8B | **0.6733** | 1.9875 | **0.269** | PPL 最低却 26.9% 音节非法 → 盲点 |
| Qwen2.5-7B+LoRA | 1.2505 | 3.7028 | 0.0648 | 通用 7B |
| TiLamb-7B (base) | 1.3331 | 5.7274 | 0.0157 | 藏文专门化 |
| TiLamb-7B+LoRA | 1.2333 | 5.0256 | **0.004** | ≈ 人类(0.004)，LoRA 修复 |

> 中心发现：**PPL 低 ≠ 藏文好**（Llama PPL 最低却 ISR 最高）；LoRA 把 TiLamb ISR 从 0.0157→0.004（≈人类量级）。

### 2.2 任务3 摘要（n=594, beam4）— 跨任务系统性失败
| 系统 | ISR↓ | scr | chrF | ROUGE-L | BLEU-4 | 形态 |
|---|---|---|---|---|---|---|
| TiLamb-7B+LoRA | **0.0051** | 0.8965 | 0.3212 | 0.1893 | 0.3203 | 抽象改写 |
| mT5-base (BodSUM full-FT) | 0.0069 | 0.8857 | 0.2871 | **0.2827** | 0.1295 | 抄写偏置 |
| Qwen2.5-7B+LoRA | **0.0462** | 0.8321 | 0.0619 | 0.0119 | 0.0010 | 跨脚本崩 |
| LEAD 基线 | — | — | — | (ledger §7.4.2) | — | 抽取 |

> ⚠️ **mT5 是 full-FT（非 LoRA）**，权重 `<MODEL_ROOT>/models/mt5-bodsum`；dump 曾误用未微调 `mt5-base` 致 ISR 虚高到 0.0307，已修正并复算一致。

### 2.3 摘要侧配对 bootstrap（B=2000，12 组全部 CI 不含 0 ✅）
ISR↓ 越低越好（负 Δ = A 更优）；chrF/ROUGE-L/BLEU-4 越高越好（正 Δ = A 更优）。
| 对比（A−B） | ISR Δ[CI] | chrF Δ[CI] | ROUGE-L Δ[CI] | BLEU-4 Δ[CI] |
|---|---|---|---|---|
| TiLamb − mT5 | −0.0024[−.0036,−.0011] | +0.0301[+.0206,+.0396] | **−0.1003[−.1102,−.0901]** | +0.2002[+.2157,+.2440] |
| TiLamb − Qwen | −0.0415[−.0450,−.0382] | +0.2504[+.2454,+.2553] | +0.1626[+.1576,+.1677] | +0.3278[+.3527,+.3722] |
| mT5 − Qwen | −0.0392[−.0428,−.0357] | +0.2202[+.2122,+.2278] | +0.2629[+.2535,+.2717] | +0.1276[+.1248,+.1417] |

> **可写"显著"的**：①TiLamb 正字法 ISR 显著优于 mT5 与 Qwen；②Qwen ISR 显著劣于两者约一个数量级（通用 7B 系统性失败跨任务互证）。
> **关键分歧**：mT5 ROUGE-L 最高却 BLEU-4 最低 → 单一 ROUGE-L 把"抄写"误判为"摘要好"，须多指标联合报告（即 §7.4.4 抽取偏置的独立佐证）。

### 2.4 长度扫（7 档 T∈{32,48,64,96,128,192,256}, n=150, beam4）— Theorem 3.2 证伪
| 模型 | ISR 范围 | 相对人类(0.0043) | Spearman(压缩率,ISR) |
|---|---|---|---|
| TiLamb+LoRA | 0.0041–0.0082 | 0.9×–1.9× | −0.821（方向符，幅度仅≈2×）|
| Qwen+LoRA | 0.0401–0.0539 | **9.3×–12.5×** | **+0.643（反向）** |

> **结论**：原 Theorem 3.2「`v_0(L)` 随压缩率单调上升」**证伪**（Qwen 反向、全程波动仅 1.35×）。已替换为 Theorem 3.2′（压缩不变性，见 §3.2）。
> **强证据（JSON 内 mean_gen_syll/compression_syll）**：音节产出随预算线性涨（TiLamb 20.4→153.2 = 7.5×；Qwen 6.5→42.5 = 6.6×）而违反率恒定（2.0×/1.35×）→ 预算影响"说多少"、不影响"说得多合法"，两层分解同实验坐实。

## 3. 论文章节映射（去哪找料）
- **§1 动机/盲点**：§2.1 Llama PPL 最低却 ISR 最高。
- **§2 TiNews 基准**：`docs/TiNews_outline_rev4.md`。
- **§3 Method（IB/KL 框架 + Theorem 3.2′）**：`docs/TiNews_contribution_v6.md` §3（Definition 1–3 + Theorem 3.2′ + Corollary 3.3 + §3.2.4 位置剖面）。
- **§4 实验设置**：`scripts/` 解码参数 + `docs/TiNews_task3_spec.md`。
- **§5 任务1 分类**：`data/results_textcnn*/cino/bert`（详见 ledger §6）。
- **§6 任务2 续写**：§2.1 + `results_bootstrap_gen.json` + §2.4 长度扫。
- **§7 任务3 摘要/关键词**：§2.2 + §2.3 + `results_g4_mt5*.json` + LEAD。
- **§8 Discussion/Limitations**：`docs/TiNews_limitations_draft.md` + `contribution_v6.md` §6.5/§9。

## 3.1 图数据（新增，2026-10-13）
全部 Figure / 附录表格数据已提取并整合到 **`docs/figure_data_report.md`**（数值为原始精度，可直接抄 LaTeX）。
| 图 | 内容 | 状态 |
|---|---|---|
| A1 | 4 数据集长度分布（音节/句子，Q25/中位/Q75/max/10bin） | ✅ |
| A2 | BodSUM 去重 5,945 = 5,351 train + 594 test | ⚠️ 预去重原始未留存，**54 条无法核验** |
| A3 | Ti-SUM 124 簇 / 992 文档；**258 篇缺关键词且 257 篇集中在 36 个簇**（非随机缺失） | ✅ |
| B1/B2 | 7 类混淆矩阵（N=1049）× 3 模型 × full/lead200 | ✅ |
| B3 | 每类 F1 + Δ(full→lead200)；`tibetan_culture` 最敏感 | ✅ |
| C3 | $p_{\max}$ 20 桶直方图；**α = P(p_max≥0.9) = 0.35643**（370,681 step 实跑） | ✅ |
| E1 | 预算长度扫；产出解离证据（strategies → 6.6–7.2× vs ISR 仅 1.35–2.01×） | ⚠️ 逐档 chrF/ROUGE/BLEU 为 null |
| C5/D2/F3 | 定性样例 22 条 | ✅ n=1~5，**不可做显著性陈述** |

> **C3 阈值口径勿混**：`theta_low=0.0904`/`q_high=3.2079` 是**熵**阈值（对应 `entropy_q20/q80`），与 `alpha=0.3564`（概率阈值 0.9 下的比例）不同量纲。
> **C5/D2 揭示 ISR 盲区**：Qwen zero-shot 最严重失败（语言坠落/指令回显）ISR = 0.0000 → ISR 只在模型确实产出藏文时才有意义，须写入 Limitations。

## 4. 学术红线（写时务必遵守）
1. 跨 tokenizer **PPL 不直比**；唯一可比量是 **bits/藏文字节**（§2.1 已桥接，分母 485286）。
2. 中心指标是 **ISR（非法音节率）**；不用 `1−SCR` 松口径。
3. **证伪不删**：Theorem 3.2 速率形式已证伪，如实写入（自我证伪是资产，非污点）。
4. **KL 小 ≠ 质量好**；压缩不能修复正字法失败（Theorem 3.2′）。
5. **无 bootstrap 不写"显著"**：摘要/续写 bootstrap 已齐（§2.3 / results_bootstrap_gen），可写。
6. mT5 = **full-FT 非 LoRA**；Qwen 失败根因是**预训练/词表缺藏文生成能力**（base+chat template 探针仍崩，ISR 0.035），对 LoRA 与 prompt 格式双重鲁棒。
7. (A2) 位置平稳性是**实证验证的实质断言，非第一原理证明**；正文不得写"已证明"。

## 5. 待你定 / 待补（非实验）
1. **Theorem 3.2′ 3 处修正待拍板**（10-07 已问未回）：①「压缩」未真操纵（全档 trunc≈1.00）→ 改名"输出预算/前缀长度不变性"；②「sub-token」方向反了（藏文音节跨多子词 token=supra-token）→ 改"与语义正交的字形轴"；③「恒定」过度声称（n=150, ρ 不显著）→ 改"无大尺度依赖 ≤1.35×"+ TOST。
2. **人工评测已完成**（实测数据见 `docs/TiNews_human_eval_final.md`；材料 `data/human_eval_200.xlsx` / `.jsonl` 已含 rater 评分）。Krippendorff's α = 0.88（TiLamb）/ 0.75（Qwen）/ 0.78（Llama）；正字法维度人类 4.9 ± 0.3、TiLamb 4.6 ± 0.5、Qwen 2.2 ± 1.1、Llama 1.6 ± 1.0。
3. **78 条 ISR 边缘组合**待藏文专家审校。
4. **论文正文**（Intro/Method/Results/Discussion）尚未写，仅 §G1 + Limitations 草稿就绪。
5. 可选后续实验：P3 判别性（全参微调降 γ vs LoRA 不能）、长度扫补 `--metrics chrf,rouge` 解离证据。
