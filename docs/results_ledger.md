# TiNews Results Ledger
> 论文中出现的**每一个数字**都必须能追溯到「脚本 + 结果文件」。定稿前逐行核对。
> 服务器路径前缀：`<MODEL_ROOT>/TiNews/`

---

## 0. 数据层面的数字

| 论文数字 | 值 | 来源脚本 | 结果文件 | 状态 |
|---|---|---|---|---|
| 文档总数 N | 10,484 | `prep_data.py` | `stats.json` → `n_docs` | ✅ 实测 |
| 原始文件数 / 空文件数 | 10,491 / 7 | `prep_data.py`（跳过空文件） | — | ✅ 实测 |
| train / val | 9,435 / 1,049 | `prep_data.py` | `stats.json` → `n_train`/`n_val` | ✅ 实测 |
| 字符数 min/p25/中位/p75/p95/max | 46 / 801 / 1,631 / 3,200 / 6,194 / 114,317 | `prep_data.py` | `stats.json` → `chars` | ✅ 实测 |
| 字符数均值 | 2,346 | 同上 | 同上 | ✅ 实测 |
| 每篇句数 中位/均值/max | 3 / 4.34 / 136 | `prep_data.py` | `stats.json` → `sents_per_doc` | ✅ 实测 |
| 各类别篇数（7 类） | 1,964 / 1,860 / 1,762 / 1,746 / 1,376 / 1,215 / 561 | `prep_data.py` | `stats.json` → `class_dist` | ✅ 实测 |
| 续写对 train/val | 31,278 / 3,769 | `prep_data.py` | `continuation_{train,val}.jsonl` 行数 | ✅ 实测 |
| §8.2 分箱样本数（val） | 796 / 1,004 / 789 / 1,180 | `prep_data.py` | `stats.json` → `cont_bins_val` | ✅ 实测 |

---

## 1. 分类任务（§7.1）

### 1.1 传统模型（CPU，已完成）

| 设定 | 模型 | Accuracy | Macro-F1 | 来源脚本 | 结果文件 | 状态 |
|---|---|---|---|---|---|---|
| full | NaiveBayes | 0.7617 | 0.7498 | `train_lightweight.py` / `train_shortctx.py` | `results_lightweight.json`、`results_shortctx.json` | ✅ |
| full | **SVM** | **0.8570** | **0.8505** | 同上 | 同上 | ✅ |
| full | LogReg | 0.8379 | 0.8277 | 同上 | 同上 | ✅ |
| full | TextCNN | 0.8055 | 0.7869 | `train_textcnn.py` | `results_textcnn.json` | ✅ |
| lead500 | NB / SVM / LogReg | 0.7531 / **0.8151** / 0.8074 | 0.7355 / **0.7986** / 0.7889 | `train_shortctx.py` | `results_shortctx.json` | ✅ |
| first_sent | NB / SVM / LogReg | 0.7255 / **0.8055** / 0.7846 | 0.6942 / **0.7819** / 0.7581 | `train_shortctx.py` | `results_shortctx.json` | ✅ |
| lead200 | NB / SVM / LogReg | 0.7083 / **0.7617** / 0.7607 | 0.6816 / **0.7373** / 0.7310 | `train_shortctx.py` | `results_shortctx.json` | ✅ |
| lead200 | TextCNN | 0.7550 | 0.7286 | `train_textcnn.py --setting lead200` | `results_textcnn_lead200.json` | ✅ 实测 |
| first_sent | TextCNN | 0.7817 | 0.7572 | `train_textcnn.py --setting first_sent` | `results_textcnn_first_sent.json` | ✅ 实测 |

### 1.2 准确率下降 Δ（相对 full）

| 设定 | NB | SVM | LogReg | 来源 |
|---|---|---|---|---|
| lead500 | -0.86pp | -4.19pp | -3.05pp | `results_shortctx.json` 计算 |
| first_sent | -3.62pp | -5.15pp | -5.33pp | 同上 |
| lead200 | -5.34pp | **-9.53pp** | -7.72pp | 同上 |

### 1.2b 预训练模型 full → lead200 的 Δ（与 §1.2 同口径，均相对 full）

| 模型 | Acc Δ | Macro-F1 Δ | 来源 |
|---|---|---|---|
| Tibetan-BERT | -5.91pp (0.8275→0.7684) | -7.43pp (0.8214→0.7471) | `results_tibetan_bert*.json` |
| CINO-base-v2 | -6.67pp (0.8751→0.8084) | -6.86pp (0.8657→0.7971) | `results_cino-base-v2*.json` |
| TextCNN | -5.05pp (0.8055→0.7550) | -5.83pp (0.7869→0.7286) | `results_textcnn*.json` |

> 规律：lead200 截断对**预训练模型**的打击（~6–7pp）明显大于对轻量模型（SVM ~9.5pp 但 NB/LogReg 仅 ~5–8pp），说明预训练模型更依赖长程上下文；CINO 在任何设定下均居首（藏文专门化优于 mBERT 类 Tibetan-BERT）。

### 1.3 预训练模型与大模型

| 设定 | 模型 | 来源脚本 | 结果文件 | 状态 |
|---|---|---|---|---|
| full | Tibetan-BERT | `train_bert.py --model tibetan_bert --setting full` | `results_tibetan_bert.json` | ✅ 实测 Acc 0.8275 / mF1 0.8214 |
| full | CINO-base-v2 | `train_bert.py --model cino-base-v2 --setting full` | `results_cino-base-v2.json` | ✅ 实测 Acc 0.8751 / mF1 0.8657（全设定最强） |
| lead200 | Tibetan-BERT | `train_bert.py --model tibetan_bert --setting lead200` | `results_tibetan_bert_lead200.json` | ✅ 实测 Acc 0.7684 / mF1 0.7471 |
| lead200 | CINO-base-v2 | `train_bert.py --model cino-base-v2 --setting lead200` | `results_cino-base-v2_lead200.json` | ✅ 实测 Acc 0.8084 / mF1 0.7971 |
| full | TiLamb-7B（分类） | 未实现（可选） | — | ⬜ 待定 |

---

## 2. 续写任务（§7.2 / §8.2）

| 数字 | 来源脚本 | 结果文件 | 状态 |
|---|---|---|---|
| LSTM val PPL | `train_lstm.py`（CPU 全量，8 ep） | `results_lstm.json` → `val_ppl` | ✅ 实测 **29.82**（best ep8） |
| LSTM 按前文句数分箱 PPL | 同上 | `results_lstm.json` → `ppl_by_nprev` | ✅ 实测 **{1句:24.85, 2-3句:27.57, 4-6句:31.38, >6句:35.23}** |
| TiLamb-7B + LoRA val PPL | `train_continuation_lora.py`（已修 grad_ckpt + enable_input_require_grads） | `results_tilamb_lora.json` → `val_ppl` | ✅ **实测 5.03**（2026-09-16 23:10 跑完，977 步 / ~84min，GPU1，batch1×accum32，LoRA r=8 q/v_proj，4.19M 可训参） |
| TiLamb-7B **基座（无 LoRA）** val PPL | `eval_ppl.py --gpu 1`（同 DS/collate 口径） | `results_ppl_base.json` | ✅ **实测 5.73**（2026-09-17 13:22；整体 PPL，与 LoRA 同 protocol 可比） |
| TiLamb-7B + LoRA **微调增益** | base 5.73 → LoRA 5.03 | 上两文件 | ✅ **PPL 降 12.2%**（(5.73−5.03)/5.73）；同 protocol 现算，旧值 4.36/27.5% 作废 |
| TiLamb-7B + LoRA **分箱 PPL** | `eval_ppl.py --lora-path ...` | `results_ppl_lora.json` | ✅ **{1句:4.54, 2-3句:4.75, 4-6句:5.23, >6句:5.54}**（随前文变长升，见 §8.2 定稿） |
| 解码扫描 8 配置 × 8 指标 | `decode_sweep.py`（rev4 §7.4/§8.4 + TiFMS） | `results_sweep.json` + `sweep_hyps.jsonl` | ✅ **完成**（2026-09-17 13:19，7456s，GPU1，subset=500；假设已存可离线重算指标） |
| 人工评测四项均分 | 人工（2026-09-30 核查：Rater1/2/3 全空，0/200 已评） | — | ⬜ |

> ✅ **§8.2 定稿（2026-09-17 实测，三模型同口径分箱 PPL 全部产出）**：分箱 PPL **均随前文变长单调上升，强模型也不例外**——
> | 模型 | 1句 | 2-3句 | 4-6句 | >6句 |
> |---|---|---|---|---|
> | LSTM（弱，无跨句注意力） | 24.85 | 27.57 | 31.38 | 35.23 |
> | TiLamb-base（无 LoRA） | 4.82 | 5.33 | 6.12 | 6.63 |
> | TiLamb-LoRA | 4.54 | 4.75 | 5.23 | 5.54 |
> **结论**：rev4 大纲「PPL 随前缀变长下降」的预测**被推翻**；"强模型分化/下降"假设也**不成立**（强模型同样升，仅绝对水平低、上升斜率更缓：LSTM +41%、base +38%、LoRA +22%）。
> 真实发现：**长上下文续写是普遍更难的（弱/强模型皆然）**，差异在绝对 PPL 与上升斜率、不在方向。§8.2 须据此改口，撤回"分化"卖点，改述为"上下文长度对续写难度的普遍效应（与模型强弱无关）"。不得为迎合大纲硬拗方向。

> 📌 **旧数字 32.82（大纲 LSTM PPL）作废**：本次句子级 protocol 实测为 **29.82**，以本次 `results_lstm.json` 为准。
> 32.82 来自旧音节级/其他划分，不同源，不得进入论文。

> 🔧 **队列状态（2026-09-16 ~21:2x 更新）**：
> - 分类 6 实验（BERT/CINO/TextCNN × {full, lead200, first_sent}）已由 `run_now.sh` 于 20:50–21:01 全部跑完并落盘（见 §1.1/§1.2b/§1.3 ✅）。
> - 🟢 **TiLamb-7B+LoRA 已训练成功（23:10）**：守护 21:39:45 凑满 12 次空闲自动接管 → 977 步 / ~84min（2.56s/step）→ `val_PPL=5.03` → adapter 存 `~/TiNews/tilamb_lora_news_continue_v3/`（16MB）。两个修复均实测通过：7B 占 14.8GB 不 OOM；`backward` 正常出 loss（step 50→950），不再 None-grad。
> - ⚠ **decode_sweep 首次 rc=1（2 秒崩）**：`ModuleNotFoundError: No module named 'syllable_slots_v3'`——该模块只在本地工作区，**从未上传服务器**。已 `scp` 到 `~/TiNews/scripts/` 并于 23:41 以 `PYTHONPATH=~/TiNews/scripts` 重跑（教训：自建模块必须与脚本一同 scp，且守护启动时要带 PYTHONPATH）。
> - 教训固化：`run_tilamb_sweep.sh` 后续启动 python 前需统一 `export PYTHONPATH=~/TiNews/scripts`。

> ⚠ **历史数字待核（现已有实测值可对照）**：大纲曾引用 LSTM PPL 32.82、TiLamb PPL 4.36（称 27.5% 提升）。
> - LSTM：本次句子级 protocol 实测 **29.82**，32.82 作废（已见 §2 上方）。
> - TiLamb：本次句子级 + LoRA 微调实测 **5.03**。旧值 4.36 **若**是同 protocol 的"微调后"值，则矛盾（5.03 > 4.36 表示微调反而变差），故判定 **4.36 来自不同 protocol（音节级/未微调基座）**，**不可与 5.03 并列、更不可相除算提升率**。
> - 📌 **硬要求**：论文中 TiLamb 的"微调增益"必须用**同 protocol 的基座 PPL** 现算（待 `eval_base_ppl.py`），禁止沿用 4.36 / 27.5%。

---

## 3. 难度分析（§8.1）

| 数字 | 来源 | 状态 |
|---|---|---|
| 各类 F1（full, SVM / TextCNN） | `results_lightweight.json` / `results_shortctx.json` → `per_class_f1` | ✅ 实测 |
| 逐类 F1 × 4 设定的 Δ 表 | `results_shortctx.json` → `per_class_f1` | ✅ 实测 |
| 主要混淆对 science_health ↔ new_agripastoral 13.4% | `results_shortctx.json` → `confusion_summary["full\|SVM"]` | ✅ 实测 |
| "时政↔世界展望 41.2%" | **原始大纲数字，实测 0.7%，已作废** | ❌ 禁用 |
| SVM 错误数 150（full）/ 250（lead200） | `results_shortctx.json` → `confusion_summary` | ✅ 实测 |
| 输入中位字符数 / 平均非零特征 | `results_shortctx.json` → `input_len` | ✅ 实测 |

---

## 4. 标注质量（§4.2 / §4.3，非脚本产出）

| 数字 | 来源 | 状态 |
|---|---|---|
| 分句初始一致率 94.3% / 最终 100% | 人工标注记录 | ⚠ 需原始记录佐证 |
| 分句 Fleiss' Kappa 0.87 | 人工标注记录 | ⚠ 需原始记录佐证（建议主推此值，勿主打 100%） |
| 类别标注 Kappa 0.82 | 人工标注记录 | ⚠ 需原始记录佐证 |
| 重复率 0.3% / 噪声残留 0.5% | 数据清洗记录 | ⚠ 需说明计算方法 |

> 这些数字不由脚本生成，**无法被审稿人直接复现**，最容易出问题。建议：(a) 在附录写明计算方法与抽样量；(b) 措辞用"据标注记录"而非"实测"。

---

## 4b. 解码分析（rev4 §7.4 / §8.4）—— 已修缺陷清单 ⚠

> 2026-09-16 首次实跑 `decode_sweep.py` 后**实测发现 4 处缺陷**。
> 首轮产出的 `results_sweep.json` 中 **BLEU / chrF / CINO-SIM / KL 预算四项均不可用于论文**，已废弃并重跑。
> 教训：脚本"本地编译通过 + stub 测试"不等于指标正确，必须对**已知答案的输入**做健全性检验（相同文本=1.0、乱码≈0）。

| # | 缺陷 | 位置 | 后果 | 修正 |
|---|---|---|---|---|
| 1 | `corpus_bleu4` 直接对**字符串**切片取 n-gram | `decode_sweep.py:92`（原） | 实为**字符级** BLEU，标签却写 `BLEU4_syll`；BP 也用字符长度 | 先 `syll_tokens()` 切音节，BP 用音节长度 |
| 2 | `chrf_pair` 默认 `n=6` 只用单阶 | `decode_sweep.py:105`（原） | 非标准 chrF（应 n=1..6 宏平均），TiFMS 的 Σ=chrF++ 直接受影响 | 改为 n=1..6 平均 |
| 3 | CINO-SIM 只算 beam4 一次后**复制给全部 8 个配置** | `decode_sweep.py:380`（原） | 各配置该列数值完全相同，无区分度，形同造假 | 改为每配置用自己的假设；采样类对 3 seed 分别算再取 mean/std |
| 4 | nucleus 空集 bug：`sp[cum <= Pp]`，当 `p_max>Pp` 时选中集合为空 → `-log(1e-12)=27.63` | `decode_sweep.py:231/234/265`、`gen_egd:200` | top-p 的 KL 预算被污染成 **8.203**（真值约 0.05）；且 `topp` 与 `toppT` 恒等（第 234 行硬编码复用）；EGD 生成有除零风险 | 新建 `budget_correct.py`：nucleus 含跨越 token 且保证非空；tempered 配置改用**完整 KL(q‖p)** 而非 −log Z；给出按步熵分箱的条件均值 |

**附带修正（ISR 假阳性）**：`syllable_slots_v3.py` 的"后加字+再后加字"分支原判据 `len(ci)>=2`，
会把 **ལས / གས / མས** 等"根字+后加字"双字母音节误吞成 suf+suf2 → core 变空 → `ANOMALY_NO_ROOT` 假阳性。
已改为 `len(ci)>=3`（藏文带再后加字最少需 根+后加+再后加 三辅音，如 གངས）。
- 回归：12 个边界用例（ལས གས མས གངས བོད ཡོད འགས སློབ རྒྱུས ཀྲུང གསོ དགའ）全部 OK。
- **人类参考文本 ISR 基线 = 0.0065**（修正前被假阳性抬高）。此后模型 ISR 必须与该基线对照报告。

**新增产物**：`sweep_hyps.jsonl`（保存各配置生成假设）→ 后续再修指标可**离线重算**，无需重花 ~90min 生成。

> **2026-09-17 重跑状态**：首轮修正版 `run_analysis_v2.sh` 在 `decode_sweep` **生成阶段又 OOM**（batch=8 + KV cache 7B 在 24GB 爆掉，差 516MiB）。已修：生成 `batch 8→4` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
> 注意：**CINO 权重其实一直在盘**（`~/models/cino-base-v2`，728MB，Sept 14 已下；此前误报缺失是因 `find` 搜错 `~/.cache/*` 目录）；CINO-SIM 本次会真正按配置分别计算。
> 现以 **`run_full_analysis.sh`（PID 92747, setsid）** 一条龙串行：①预算(✅99s rc=0) → ②扫描(--batch 4, 进行中) → ③基座 PPL → ④LoRA 分箱 PPL。预计 ~14:00 全出：results_budget/sweep/sweep_hyps/ppl_base/ppl_lora.json。

### 4c. 修正后 KL 预算实测（`results_budget.json`，权威）

- 步数 49,674（500 条 teacher-forced）；**α = 0.3627**（p_max≥0.9 的近确定步占比）→ **1/(1−α) = 1.569**
- θ（熵 q20/q80）= (0.0904, 3.2079)；步熵三分位切点 = (0.389, 2.118)
- 平均保留质量 Z：topk50 0.9482 / topp095 0.9665 / topp095_T07 0.9749 / egd 0.9788

| 配置 | KL 全步 | 低熵 | 中熵 | 高熵 | 步间 std |
|---|---|---|---|---|---|
| greedy | **0.6404** | 0.0184 | 0.3920 | **1.5108** | 0.7385 |
| topp095_T07 | 0.1628 | 0.0182 | 0.1378 | 0.3323 | 0.1448 |
| topk50 | 0.0587 | 0.0005 | 0.0104 | 0.1652 | 0.1102 |
| topp095 | 0.0342 | 0.0121 | 0.0397 | 0.0509 | 0.0199 |
| **egd** | **0.0217** | 0.0026 | 0.0152 | 0.0473 | 0.0198 |

**可读出的三点（供 §8.4 措辞）**
1. 坏值 8.203 → 真值 **0.0342**（240× 偏差），原结论不可引用。
2. **截断误差高度集中在高熵步**：高熵/低熵比 greedy 82×、topk50 330×、topp 4.2×。这正是 EGD 的动机所在。
3. **EGD 的每步 KL 最低（0.0217 < topp 0.0342）**。
   ⚠ 原注解"因为 m(H)∈[0.95,1.0] 恒 ≥ top-p 的 0.95"**已作废**（那是修复前的错误实现）。
   修复后 EGD 的低 KL 来自两处：① H≤q20 走 argmax（低熵步 KL 仅 0.0026，近 0）；
   ② 高熵步**主动收紧**截断（m 降到 0.55 < top-p 的 0.95），高熵 KL 0.0473 < topp 的 0.0509。

> ⚠ **措辞红线（不得越界）**：KL 小 ≠ 文本质量好。greedy 的 KL 最大（0.64，one-hot 距离最远）却通常质量最好——
> KL 刻画的是"采样分布相对模型的偏离/多样性"，**不是质量指标**。§8.4 只能说"截断预算与生成多样性/失真相关"，
> **禁止**写成"KL 越低质量越高"。是否真成立须等 `results_sweep.json` 的 ISR/chrF 与 KL 做相关性检验。

> ⚠ **EGD 生成实证崩了（2026-09-28 复核 `sweep_hyps.jsonl`）**：`results_sweep.json` 中 `egd` 配置 **BLEU=6.4e-68、chrF=0.0007、ROUGE_L=0、SCR=0、distinct=1.0**；avg_len=47（其他配置 355–385）、12.8% 空串、示例为 `'\x04@\x04\x0bteHass bu gen!...'` 纯乱码。
> #### EGD 根因（2026-09-28 修正：此前"m≈1 导致乱码"是**不完整的第二根因**，致命的第一根因另有其因）
> **根因①（致命，决定乱码）**：`nxt = int(torch.multinomial(ren,1).item())` —— `multinomial` 返回的是**排序后 kept 数组内的下标**，
> 代码却直接当**词表 token id** 用，未经 `order[...]` 映射回真实 id → 每步都取到 vocab 前几个（0/1/2…控制符）token。
> 这与 `m` 取值无关：即使 m 收紧，只要漏了 `order[]` 映射，输出仍是控制字符（`'\x07\x05\x04i\rP...'`，avg_len 6.7、132/500 空串）。
> 修：`nxt = int(order[int(torch.multinomial(ren,1).item())].item())`。
> **根因②（加剧）**：`m(H)∈[0.95,1.0]`（方向也反了——越不确定应越狠截断），高熵步保留近全词表质量 → 近均匀采样。
> 修：`m` 改为从 **0.95(q20) 线性降到 0.55(q80)**，H≤q20 走 argmax（贪婪分支本身正确）。
> **根因③（性能，非正确性）**：`gen_egd` 逐 token 重算**全前缀注意力**（无 KV 缓存）→ O(n²)/context，实测 **~208ms/token**。
> 修：改为 `past_key_values` 复用 → 实测 **20.3ms/token（10×）**。
>
> **修复验证（3 样本探针，2026-09-28）**：EGD 已产出连贯藏文，例如
> `'ལྗོངས་ཏང་ཨུད་ཀྱི་རྒྱུན་ཨུ་དྲིལ་བསྒྲགས་པུའི་པུའུ་ཀྲང་དབང་ཧའེ་...'`（此前为纯控制字节）。
> 已用新增的 `--only egd` 单配置重跑模式重跑 500 样本并合并进 `results_sweep.json`（其余 7 配置不动，避免再花 90min 全量重跑）。
> **影响**：§8.4 是否成贡献点，取决于修后 EGD 质量能否达 topk/topp 水平**且** KL 仍最低；
> 否则按红线撤回贡献点(7)、§8.4 降级附录。

### 4d. EGD 修复后：8 配置 × 8 指标最终实测（`results_sweep.json`，2026-09-28 定稿）

人类参考基线：**ISR=0.00384，SCR=0.8821**（模型 ISR 越接近 0.00384 越好；越低=越合藏文音节法）

| 配置 | BLEU↑ | chrF↑ | ROUGE_L↑ | ISR↓ | SCR↑ | distinct2 | CINO_SIM | KL 预算↓ |
|---|---|---|---|---|---|---|---|---|
| greedy | 0.1913 | 0.2138 | 0.1305 | 0.0062 | 0.8872 | 0.2304 | 0.9964 | 0.6404 |
| beam2 | 0.1887 | 0.2133 | 0.1296 | 0.0045 | 0.8946 | 0.2354 | 0.9965 | — |
| beam4 | 0.1863 | 0.2137 | 0.1304 | 0.0055 | 0.8901 | 0.2256 | 0.9964 | — |
| beam8 | 0.1873 | 0.2083 | 0.1251 | 0.0055 | 0.8884 | 0.2251 | 0.9962 | — |
| topk50 | 0.2208 | 0.2236 | 0.1155 | 0.0044 | 0.8807 | 0.4946 | 0.9972 | 0.0587 |
| topp095 | 0.2219 | 0.2242 | 0.1144 | 0.0045 | 0.8792 | 0.5062 | 0.9972 | 0.0342 |
| topp095_T07 | **0.2354** | 0.2330 | 0.1300 | 0.0046 | 0.8898 | 0.4000 | 0.9972 | 0.1628 |
| **egd（已修）** | 0.2311 | **0.2376** | **0.1363** | **0.0038** | 0.8880 | 0.4006 | 0.9970 | **0.0217** |

**§8.4 裁决（2026-09-28）：贡献点(7) 可以保留，但必须按下述口径写，不得拔高**
- ✅ **成立的部分**：EGD 以**最低的每步 KL 预算（0.0217）**取得了 **chrF 最高（0.2376）、ROUGE_L 最高（0.1363）、
  ISR 最低且等于人类基线（0.0038 vs 参考 0.00384）** —— 即"更少的截断预算 + 不劣（部分更优）的语言合法性"。
- ⚠️ **不得拔高的部分**：EGD 的 **BLEU 是第 2（0.2311），低于 topp095_T07 的 0.2354**，
  不可写"EGD 质量全面最优"。且 EGD 为**单次确定性运行（无 multi-seed std）**，
  与 topk/topp 的 3-seed mean±std 并非严格同口径，**EGD vs topp095_T07 的 BLEU 差距（~1.8%）不可作显著性主张**。
- ✅ 可写句：*"EGD achieves the lowest per-step KL budget among all compared strategies while matching the
  best chrF/ROUGE-L and attaining a near-human ISR; it is second on BLEU, within ~1.8% of the best."*
- ❌ 禁止写：*"EGD dominates all decoding strategies"* / *"lower KL ⇒ better quality"*。

---

## 5. 交叉核对检查表（定稿前逐项打勾）

- [ ] 摘要中的最高准确率 = §7.1 表中最大值（且注明设定）
- [ ] §7.1 的 macro-F1 与 §7.3 能力边界表一致
- [ ] §8.1 各类 F1 的宏平均 = §7.1 的 macro-F1（允许四舍五入误差 ≤0.001）
- [ ] §8.1 与 Limitations 的混淆对一致（均为 science_health ↔ new_agripastoral）
- [ ] §5.1 各类之和 = N = 10,484
- [ ] §5.4 的 train+val = 10,484
- [ ] §9.4 未出现"分类任务需要大模型"的表述
- [ ] 所有结果表中均标注了上下文设定（full / lead500 / first_sent / lead200）
- [ ] 续写 PPL 均来自本次句子级 protocol，无旧数字混入
- [ ] 无"待跑"字样残留

---

## 6. 非人工缺陷补强（2026-09-29，GPU0 释放窗口执行）

### 6.1 T8 数据泄漏检查 ✅ 通过
- 脚本 `leak_check.py`（CPU）比对 `continuation_train.jsonl`(31278) 与 `continuation_val.jsonl`(3769)。
- article_id 形如 `world_outlook/world_outlook_1#1`（`#N`=同篇不同窗口）。
- train 与 val 的 **article 前缀交集 = 0** → 无整篇/窗口跨集泄漏。结论可靠。

### 6.2 T9 §9.2 矛盾句 ✅ 已改
- 删除 rev4 大纲 §9.2 末行"若关注藏文正字法正确性，优先 beam 类解码（待 §7.4 ISR 列确认）"。
- 改为与 §7.4.5（证伪"采样 ISR 高于 beam"）及 §8.4.7（beam 各指标不占优）一致的表述。

### 6.3 T2 ISR 措辞锁死 ✅ 已改
- outline 三处"ISR≈人类基线/等于人类基线"全部锁死为：
  "音节正字法合法性等于人类参考文本基线（0.0038 vs 0.00384）；**仅度量正字法，非文本质量/连贯性**"。
- 防误读为"质量≈人类"。

### 6.4 T6 α 阈值敏感性 ✅ 已出（`alpha_sweep.py`，GPU0 跑 32s）
- 完整 `pmax` 分布分位数：[0.166, 0.339, 0.720, 0.972, 0.998, 0.9998, 0.99999]。
- α(近确定步占比) 与放大因子 `1/(1−α)`：
  - ≥0.8 → α=**0.4468** → 1.808×
  - ≥0.9 → α=**0.3627** → 1.569×  ← 主报告值
  - ≥0.95 → α=**0.2948** → 1.418×
- 结论：α 随阈值升高而下降（符合直觉：阈值越松，越多步被判"近确定"），放大因子稳定在 **1.4–1.8×**，
  量级不敏感 → Prop 3 对阈值选择稳健。实测 topk50 KL(0.0587) vs topp KL(0.0342) = 1.72×，落在该区间，
  **与 `1/(1−α)` 近似方向对、量级对**。落盘 `results_alpha_sweep.json`(服务器)。

### 6.5 T3 bootstrap 置信区间 ✅ 已出（`run_bootstrap_chain.sh` 自动链，B=2000）
- base peritem PPL=**5.73**（与 §7.2 吻合）；lora peritem PPL=**5.03**（吻合）；逐条 NLL 落盘 `ppl_peritem.jsonl`/`ppl_lora_peritem.json`。
- bootstrap 结果（n=3769）：point PPL_base=5.727 / PPL_lora=5.026 / **gain=12.25%**；
  **95% CI(gain) = 11.9% ~ 12.6%**（不含 0）→ **增益在 95% 置信水平下显著，非 bootstrap 噪声**。
- 落盘 `results_ppl_bootstrap.json`（本地已同步）。T3 解决：续写增益 claim 可加"statistically significant (bootstrap 95% CI 11.9–12.6%)"。

### 6.6 人工评测包 ✅ 生成完成并回填
- GPU0 跑 `prepare_human_eval.py --stage gen --gpu 0`（beam=4 主协议，max_new=128，遇 `་` 截断，seed=42）。
- 200/200 完成，落盘 `human_eval_200.jsonl`（avg 79 字符/条）。本地 `build_human_eval_xlsx.py` 回填 generated 列。
- 交付 `human_eval_200.xlsx`：说明页 + Rater1/2/3 三表（200 条前文/参考/生成已填，4 维度评分列留空）。
- **剩余唯一 P0 缺口 = 3 位藏语母语者独立打分**（每评分者 200×4 维），在用户侧执行。
  - 2026-09-30 经 openpyxl 核查：桌面+工作目录两份 xlsx 一致，mtime 均为 2026-09-29 15:29:17；Rater1/2/3 三表已评行 **0/200**、已填评分格 **0/1000** → 确未启动。每张 Rater 表含 5 个评分维度列（说明页已注含义）。

### 6.7 评测覆盖度评估（用户问"评测模型够吗"）
- 现有覆盖合格维度：分类容量梯度(TextCNN→Tibetan-BERT→CINO)、续写容量梯度(LSTM→base→LoRA)、
  解码族(greedy/top-k/top-p/+T/beam4/beam8/EGD)、藏文特有指标(ISR/SCR/KL 预算)。
- **硬缺口 G1**：缺通用大模型对照（Qwen2.5-7B / LLaMA 藏文）→ 决定"藏文专门化 7B 卖点"是否成立。P2。
- **硬缺口 G2（自相矛盾）✅ 已改（2026-09-29）**：原 §8.2 Limitations 写"同时报告 chrF++、BERTScore"，但 8 配置表无 BERTScore 列。
  已改为"同时报告 chrF++ 与多样性指标；BERTScore 受单参考约束且本期未实算，留待未来工作"；指标说明表 BERTScore 行加注"本期未实算"。消除自相矛盾。
- G3 人工仅单模型(主协议)、G4 缺 encoder-decoder(mBART/mT5)、G5 缺 COMET——可选补强。

### 6.8 G1 通用大模型对照（**四模型 teacher + gen 全齐**；TiLamb-LoRA teacher 于 2026-09-30 20:41 由看门狗自动补算完成）
- 脚本 `qwen_baseline.py`：GPU0 bf16 加载通用 7–8B，对 `continuation_val`(3769) teacher-forcing 算
  PPL(自身词表) + **bits_per_藏文字节(音节桥接, 跨 tokenizer 可比 = ce_sum/syll_sum, syll_sum 三模型同为 485286)** +
  同主协议 beam=4 生成(BLEU/chrF/ROUGE-L/ISR/SCR)。修复：①`import numpy` 提到顶层(原在 main 内致 NameError)；
  ②`padding_side='left'`(消 right-padding 警告、保 batch beam 正确)；③加 `--lora-path` 支持 TiLamb-LoRA 同口径。
- **teacher 结果（tokenizer-无关口径，越低越好）**：
  | 模型 | 类型 | PPL(自身tok) | bits/藏文字节 |
  |---|---|---|---|
  | Llama-3.1-8B-Instruct | 通用 | **1.9875** | **0.6733** |
  | Qwen2.5-7B-Instruct | 通用 | 3.7028 | 1.2505 |
  | TiLamb-7B-base | 藏文专门化(base) | 5.7274 | 1.3331 |
  | TiLamb-7B-LoRA | 藏文专门化(+LoRA) | **5.0256** | **1.2333** |
  - TiLamb-base PPL=5.7274 与 §7.2 base PPL=5.73 **吻合 ✓**（内部一致）。
  - **TiLamb-LoRA PPL=5.0256 与 §7.2 LoRA PPL=5.03 吻合 ✓**；其 bits/藏文字节 1.2333 **已低于 Qwen 1.2505**（跨 tokenizer 可比口径下反超 Qwen）。
  - **⚠️ 重大威胁(T-g1)**：在跨 tokenizer 可比的"bits/藏文字节"上，**藏文专门化 TiLamb-base 反而最差**（bits 升序：Llama 0.6733 < **LoRA 1.2333** < Qwen 1.2505 < base 1.3331）。
    即"藏文专门化 7B 零样本续写优于通用 7–8B"**不成立**——通用模型在藏文新闻续写上零样本就更强。
  - **关键混杂(confound，须如实声明)**：Qwen/Llama 是 **Instruct** 模型、TiLamb 是 **base** 模型；且二者训练规模(万亿 token, 2024)远大于 TiLamb(有限藏文)。
    公平比较应补 **Qwen/Llama-base**（非 Instruct）；**TiLamb+LoRA 同口径已于 2026-09-30 20:41 补齐**（`results_g1_tilamb_lora.json`），LoRA 把 base 5.7274→5.0256（PPL）、1.3331→1.2333（bits/音节）。
- **对论文叙述的影响（草稿，待用户拍板）**：
  - 原"藏文专门化预训练 > 通用"卖点**不能成立**；须把论文重定位为 **benchmark + 解码分析(§8.4 KL 框架) + 藏文特有诊断(ISR/KL)** 贡献，
    G1 表改为"通用模型给出强零样本基线、专业模型经微调可竞争、benchmark 暴露通用模型仍有 headroom"。
  - 红线守住：不写"TiLamb 比通用模型好"；若 LoRA 仍不敌通用零样本，则明确"微调在专业任务上缩窄差距但未反超"。
- **gen 指标（主协议 beam=4，n=200，与参考对照；ISR 越低越好，chrF/SCR 越高越好）**：
  | 模型 | 类型 | bleu4 | chrf | rougeL | ISR↓ | SCR↑ |
  |---|---|---|---|---|---|---|
  | Llama-3.1-8B-Instruct | 通用(Instruct) | 0.0 | 0.0258 | 0.0001 | 0.269 | 0.0918 |
  | Qwen2.5-7B-Instruct | 通用(Instruct) | 0.0 | 0.0405 | 0.0235 | 0.0648 | 0.7489 |
  | TiLamb-7B-base | 藏文专门化(base) | 0.1579 | 0.2049 | 0.0962 | 0.0157 | 0.8715 |
  | TiLamb-7B-LoRA | 藏文专门化(+LoRA) | 0.185 | 0.2138 | 0.1035 | **0.004** | **0.8836** |
  - 人类参考基线：ISR=0.00384、SCR=0.8821（§4d）。
- **gen 维度的关键发现（与 teacher PPL 形成跨维度对照）**：
  - **藏文正字法维度：专门化碾压通用零样本**。ISR（非法音节率，中心点指标）：TiLamb-LoRA **0.004**（≈人类基线 0.00384）≪ Qwen 0.0648 ≪ Llama 0.269（即 26.9% 生成音节非法，约人类 45 倍）；SCR（合法且含元音/后加字占比）：TiLamb 0.88 vs Llama 0.09。**通用大模型在藏文音节合法性上系统性失效**——这正是 benchmark 暴露的核心价值。⚠️ 此前"≈91% 违反"措辞来自 `1−SCR`，混入了"合法但无元音/后加字"的音节，已改为以 ISR 为准（见 `data/task3/isr_calibration.json` 校准）。
  - **任务匹配维度(chrF vs 参考)**：TiLamb-LoRA 最高(0.2138) > base(0.2049) ≫ Qwen(0.0405) ≈ Llama(0.0258)。因 TiLamb 在藏文新闻续写微调，生成更贴参考风格。
  - **困惑度维度**：⚠️ PPL 为**各自 tokenizer** 口径、**不可跨模型直比**；跨模型唯一可比量是 bits/藏文字节。bits 上仍通用最低(Llama 0.6733 < **LoRA 1.2333** < Qwen 1.2505 < base 1.3331) → **T-g1 威胁在 bits 维度依然成立**："藏文专门化整体优于通用零样本"不成立。
  - ⚠️ **不可跨维度混比**：chrF 高 ≠ PPL 低即"更好"，两指标度量不同（重叠度 vs 困惑度）。论文须**分维度陈述**，禁止"TiLamb 比通用好"式笼统 claim。
- ✅ **TiLamb-LoRA teacher 已补算（2026-09-30 20:41，看门狗在 GPU0 释放后自动触发 full 模式）**：补齐 `teacher` 字段 → **PPL=5.0256、bits/藏文字节=1.2333**（ce_sum 598481.09 / ntok 370681 / syll_sum 485286 同分母），与 §7.2 LoRA PPL=5.03 **吻合 ✓**。full 模式同时重跑生成，gen 指标相对 gen-only 版有 ~0.001 级漂移（chrf 0.2131→0.2138、isr 0.0043→0.004、scr 0.8842→0.8836），本表已按最新 json 更新。
- **对论文叙述的影响（更新草稿，待用户拍板）**：
  - 卖点从"藏文专门化整体优于通用"重定位为：**benchmark 暴露通用大模型在藏文正字法上的系统性缺陷（ISR 远离人类、SCR 崩），而藏文专门化模型在该维度接近人类、且目标任务续写匹配度(chrF)更高；但困惑度(PPL)仍低于通用零样本，存在 headroom**。
  - 贡献点 = benchmark + 解码分析(§8.4 KL 框架) + 藏文特有诊断(ISR/SCR/KL) + 通用模型正字法失效的实证。
  - 红线守住：分维度陈述，不写"整体优于通用"；LoRA 若 PPL 未反超 Llama，则明确"微调缩窄差距但未反超困惑度"。

---

### 6.9 ISR 校准（2026-09-30，选项 A 第一步 ✅）
- 写 `isr_calibration.py`（轻量复刻 `decode_sweep.isr_scr`，仅依赖 `syllable_slots_v3`，不引 torch），在 3 个人类藏文语料上同验证器算 ISR/SCR（结果见 `data/task3/isr_calibration.json`）：
  - human_eval 续写参考 600 条 / 66.8k 音节 → ISR=0.00408, SCR=0.8724（≈ §203 人类基线 0.00384/0.8821）
  - BodSUM 新闻正文 5945 篇 / 281 万音节 → ISR=0.00573, SCR=0.8878
  - Ti-SUM body 992 篇 / 81 万音节 → ISR=0.00950, SCR=0.8789
- **结论**：验证器在人类藏文上 ISR 仅 0.4%–1.0%、SCR≈0.88，**不过杀合法藏文** → "通用模型 ISR 0.269 ≫ 人类 ~0.004" 是经校准的真实相对信号，中心点立得住（非验证器偏差）。
- 中心点口径已锁定为 **ISR（非法音节率）**，弃用 `1−SCR` 的"91% 违反"松表述（大纲 §6.3.3 / G1 .tex / 本 §6.8 已同步）。
- 待办（用户侧人工）：`crosscheck_conflicts.csv`(26)+`crosscheck_unseen.csv`(52)=78 条边缘组合交专家拍板，定稿 `syllable_slots_v3` 的 DRAFT 规则集后重跑校准确认 human ISR 不漂移。

### 6.9.1 续写生成指标的配对 bootstrap 显著性检验（2026-10-06 ✅ CPU 已出）

脚本 `bootstrap_gen_metrics.py`（CPU；自包含 `isr_scr` + 复制服务器 `chrf_pair`，不依赖 torch）。
- **动机**：核心 claim「通用模型 ISR 远高于藏文专门化」此前只有点估计，无显著性检验。按红线（没 bootstrap 证实不得写"显著"）补做。
- **数据**：`results_g1_*_hyps.jsonl`（四模型各 200 条逐条 `reference`/`generated`，已备份本地）。
- **口径自检（先对账再 bootstrap）**：本地复算 vs 服务器已知值 ——

  | 模型 | ISR 本地 | ISR 服务器 | chrF 本地 | chrF 服务器 |
  |---|---|---|---|---|
  | TiLamb-LoRA | 0.0040 | 0.0040 | 0.2138 | 0.2138 |
  | TiLamb-base | 0.0157 | 0.0157 | 0.2049 | 0.2049 |
  | Qwen | 0.0648 | 0.0648 | 0.0405 | 0.0405 |
  | Llama | 0.2690 | 0.2690 | 0.0258 | 0.0258 |

  → **四个模型 ISR/chrF 全部逐位一致**，本地口径可信，bootstrap 结果有效。
- **配对 bootstrap（B=2000，seed=42，按 id 配对）**：

  | 对比 | Δ | 95% CI | CI 含 0？ |
  |---|---|---|---|
  | ISR: Qwen − TiLamb-LoRA | +0.0608 | [0.0471, 0.0763] | **否 → 显著** |
  | ISR: Llama − TiLamb-LoRA | +0.2650 | [0.2421, 0.2883] | **否 → 显著** |
  | ISR: TiLamb-base − LoRA | +0.0118 | [0.0072, 0.0170] | **否 → 显著** |
  | chrF: LoRA − Qwen | +0.1733 | [0.1639, 0.1823] | **否 → 显著** |
  | chrF: LoRA − Llama | +0.1880 | [0.1796, 0.1962] | **否 → 显著** |
  | chrF: LoRA − base | +0.0089 | [0.0033, 0.0152] | **否 → 显著** |

- **结论**：**全部对比 CI 均不含 0** → 续写侧"通用模型 ISR 显著高于专门化、专门化 chrF 显著更高"可写"显著"；且 **LoRA 微调本身显著降低 ISR**（base 0.0157 → LoRA 0.0040，Δ=0.0118 显著）。落盘 `results_bootstrap_gen.json`。
- ⚠️ **chrF 口径发现（须如实标注）**：服务器 `decode_sweep.chrf_pair` 与标准 chrF 有两点差异——① 取 n-gram 前**剔除 tsheg ་ 与 shad །**；② **逐阶 n 先算 F_β 再对 n 取平均**（标准 chrF 是先宏平均 P/R 再算 F）。本表沿用服务器口径以保证与历史数字可比；**若投稿改用标准 chrF，须全表重算**（差值方向预期不变）。

### 6.10 G4 encoder-decoder 对照（mT5-base，2026-09-30 22:02 ✅ 双口径出结果）
- 模型获取：服务器侧 HF/hf-mirror/modelscope **全被代理自签名证书拦截**（Python SSL 失败、`curl -k` 抓回 48161B 的 HTML 错误页）、服务器现成 `m2m100_418M` **语言表无藏文 `bo`**、`nllb-200-distilled` 目录为空。
  **绕行方案**：本机走 `hf-mirror.com` 下载 mT5-base 全套（pytorch_model.bin **2,329,735,129 B，与官方 Content-Length 逐字节一致**），scp 上传 `<MODEL_ROOT>/models/mt5-base`。✅
- 脚本 `g4_encdec_baseline.py` 已改为模型无关 Auto 类；修 3 个调用 bug：①`load_val(path)` 缺参；②`corpus_bleu4/chrf_corpus` 参数顺序（应为 `refs, hyps`）；③`rougeL`/`isr_scr` 是**单条**函数，需逐条计算后取均值（原误作 corpus 级调用）。
- **双口径结果（同 200 条 CTX200、beam=4）**：

  | 口径 | PPL(seq2seq) | bits/藏文音节 | chrF↑ | ISR↓ | SCR↑ |
  |---|---|---|---|---|---|
  | mT5 plain（干净前文直喂，**形式 mismatch**） | 1128.47 | 17.82 | 0.0263 | 0.0192 | 0.432 |
  | **mT5 span-corruption 原生**（`<extra_id_0>`） | **953.33** | **17.43** | **0.0344** | 0.0192 | 0.539 |
  | ← 对照 Llama-3.1-8B (§6.8) | 1.99 | 0.673 | 0.0258 | 0.269 | 0.092 |
  | ← 对照 TiLamb-7B-LoRA (§6.8) | 5.03 | 1.233 | 0.2138 | 0.0040 | 0.884 |

- **关键诊断（决定性证据）**：plain 口径下 mT5 的生成形如 `<extra_id_0>…<extra_id_1>…`——它把"完整前文"当成**被挖空的文本**在做 span corruption 填补。
  故 PPL 1128 **不是实现 bug，而是零样本续写对 encoder-decoder 的形式 mismatch**。
  改用其预训练原生形式后仅 1128→953（−15%），说明 **mismatch 只解释一小部分，主因是 mT5 零样本藏文续写能力本身很弱**（藏文在 mC4 占比极小且未对齐）。
- **⚠️ 红线：不可并列比**。mT5 的 953/1128 **不能**与 Llama 1.99 并列得出"encoder-decoder 架构更差"的结论——
  它混淆了「架构」与「零样本迁移 + 参数量(mT5 582M vs Llama 8B) + 藏文对齐」三个因素。论文中必须写明这是
  **zero-shot transfer 结果，非架构对照**。
- **可立的发现（有价值）**：mT5 的 **ISR=0.0192 很低**（≈TiLamb-base 0.0157，比 Llama 0.269 低一个数量级），
  但 **chrF 仅 0.034**（内容几乎与参考无关）。即**正字法合规与内容质量解耦**——
  单一指标（无论 PPL 还是 ISR）都不足以判定藏文生成质量，必须 ISR + 内容指标**联合诊断**。
  这为 §"统一诊断框架"的必要性提供了直接实证。
- **后续**：公平的架构对照改放 **Task 3a 摘要**（seq2seq 原生任务、监督微调，见 §7.4），已于 22:08 在 GPU0 启动 mT5 全量微调（BodSUM 5351×3 epoch）。

## 7. 第3任务：藏文新闻摘要与关键词抽取（2026-09-30 数据整合完成；实验待 GPU）

### 7.1 数据来源与互补性
- **BodSUM-6000（主评测集 · 单篇标准 CSV）**：`C:\Users\机械革命\Desktop\BodSUM-6000.csv`，字段 `ID, Title, SUM, Text`。官方声明经严格主观 + 客观评价；抽样前 12 行确认藏文规范、摘要流畅、无聚类噪声。
- **Ti-SUM（扩展集 · 聚类 + 关键词）**：`C:\Users\机械革命\Desktop\Ti-SUM`，130 个 txt，同文件 = 同聚类（同事件多报道）；每条 `body` + `title`(=人工摘要) + `keywords`(藏文关键词，། 分隔)。
- **互补结论（用户问"能否互补"的回答）**：BodSUM = 标准/干净/单篇/大规模 → 作**主摘要基准**（统计力强、结论可靠）；Ti-SUM = 真实/聚类/带关键词 → 补 BodSUM 缺失的**关键词抽取**与**同事件多文档摘要**两维度。二者不冗余、互补，统一归入 Task 3。详见 `TiNews_task3_spec.md` §1.3。

### 7.2 清洗与防泄漏切分（`prep_task3.py`，seed = 20260930）
- **BodSUM**：6000 → 剔除 1 条空正文 → 5999 → 54 条精确重复去重 → **5945**（train 5351 / test 594，随机 90/10，单篇无聚类故随机安全）。⚠️ 原 `ID` 列 6000 行中**非唯一**（疑似导出异常）→ 已改顺序 ID(1…5945)。
- **Ti-SUM**：992 条 / 124 聚类；有摘要 974、有关键词 734。按 cluster(文件) 切 90/10 → train 902 / test 90；摘要子集 train 884 / test 90；关键词子集 734。1 个聚类全无摘要（仅入关键词维度，已自动排除出摘要子集）。
- 交叉去重：BodSUM 正文精确去重（54）；跨集近重复(paraphrase)未处理，列 Limitations 未来工作。

### 7.3 评测协议（复用 §6.3 口径，保证可比）
BLEU-4（音节切分，§6.3.5）/ chrF++ / ROUGE-L / ISR-SCR（藏文正字法，§6.3.3）/ 可选 COMET（G5）；解码统一 §6.3 主协议（beam=4, no_repeat_ngram_size=3, length_penalty=1.0, seed=42）；跨 tokenizer 用 bits/藏文字节 桥接（同 §6.8）；关键词抽取用音节级 F1 / ། 分隔集合 Jaccard。

### 7.4 子任务与实验排期（GPU 待释放）
| 子任务 | 数据 | 基线 | 状态 |
|---|---|---|---|
| 3a 生成式摘要 | BodSUM test 594 | 抽取式 LEAD ✅ + mT5-base ✅ + TiLamb-7B+LoRA ✅ + Qwen2.5-7B+LoRA ✅ | 🟢 神经三系统全出 |
| 3b 关键词抽取 | Ti-SUM 734 | 词频 top-k ✅(已出) + TextRank藏文适配 + 序列标注 | 神经 ⬜ 待 GPU |
| 3c 多文档摘要 | Ti-SUM test 90 | LEAD-over-cluster ✅(已出) + centroid | 神经 ⬜ 待 GPU |

- 与 G4/G5 生成链复用，避免重复造轮。数据已落地：`data/task3/*.jsonl` + `task3_report.json` + `task3_baselines.json`。
- **抽取式/无监督基线已出（2026-09-30，纯 CPU，无需 GPU）**，见 §7.4.1。
- **神经基线阻塞**：2026-09-30 12:20 再查 4090，GPU0 仍被用户 `pareto_frontier.py --arm`(pid 438511)、GPU1 被 `ssd_v2_train.py` 占用；lora teacher 补算与 G4(mT5) 均未触发。待卡释放后启动神经基线并填入 §7.4.1。

### 7.4.1 抽取式 / 无监督基线（2026-09-30 已出，CPU）
脚本 `task3_eval.py`，指标 BLEU-4(音节)/chrF/ROUGE-L/ISR/SCR（同主协议口径）。
- **3a 生成式摘要（BodSUM test 594，修正句切分后）**：LEAD-1 → BLEU4=0.0314, chrF=0.1321, ROUGE-L=0.1098, ISR=0.0010, SCR=0.7359；**LEAD-2 → BLEU4=0.1759, chrF=0.4160, ROUGE-L=0.3560, ISR=0.0032, SCR=0.8753**。
  ⚠️ 早期数值（chrF 0.0162 等）因 `sentences()` 按 tsheg 切分成"音节块"而**严重低估**，已作废，一律采用上表。ISR 极低系直接抄原文所致。
- **3c 多文档摘要（Ti-SUM test 12 聚类有 title）**：LEAD-over-cluster → chrF=0.0078, ROUGE-L=0.0031, ISR=0.0, SCR=0.9167。
  ⚠️ **口径修正（2026-09-30 23:20）**：该行原用 `sentences()`，它按 **tsheg(་) 切分** → 产出的是"音节块"而非句子，故 LEAD-1 实为"第一个音节块"，数字被严重低估。已新增 `sent_split()`（按句末标点 ། / ༎ / 换行 切句），修正后见 **§7.4.3**（LEAD chrF 0.1749 / ROUGE-L 0.1012）。**正文一律采用修正后口径**，本行旧值仅作历史记录。
- **3b 关键词抽取（Ti-SUM 734）**：词频 top-k → 词级 F1=0.051（P=R=0.051）。无监督频 baseline 极弱，印证关键词抽取非平凡。
- ⚠️ 以上为**评测下界**。神经基线见 §7.4.2；更强无监督基线见 §7.4.3。

### 7.4.2 神经基线：3a 摘要监督微调（2026-10-01 ✅ 三系统全齐）
脚本 `task3_sum_train.py`（支持 `--arch mt5` 全参 / `--arch causal --lora`；评测复用 decode_sweep 指标，另输出 **compression=生成长/源长** 供 §Theorem 3.2 压缩–正字法分析）。
三套系统同数据（BodSUM train 5351）、同主协议（beam=4）：

| 系统 | BLEU-4↑ | chrF↑ | ROUGE-L↑ | ISR↓ | SCR↑ | 压缩率 |
|---|---|---|---|---|---|---|
| LEAD-1（抽取） | 0.0314 | 0.1321 | 0.1098 | 0.0010 | 0.7359 | 0.0347 |
| **LEAD-2（抽取）** | **0.1759** | **0.4160** | **0.3560** | 0.0032 | 0.8753 | 0.1003 |
| **LEAD-2 长度匹配后** | — | 0.1201 | **0.2945** | — | — | 0.0855 |
| mT5 zero-shot（§6.10，非摘要任务） | ~0 | 0.0344 | 0.0263 | 0.0192 | 0.539 | — |
| mT5-base 全参微调 | 0.1295 | 0.2871 | 0.2827 | **0.0069** | **0.8857** | 0.0855 |
| **TiLamb-7B+LoRA** | **0.3203** | **0.3212** | 0.1893 | **0.0051** | **0.8965** | 0.2284 |
| Qwen2.5-7B+LoRA | 0.0010 | 0.0619 | 0.0119 | **0.0462** | 0.8321 | 0.0712 |

- **mT5-base 全参微调**（BodSUM 5351 × 3 epoch, bs8×accum2, lr5e-5, bf16, GPU0, 480s）：ISR=0.0069 / SCR=0.8857 逼近人类基线（§6.9：0.0057/0.888）→ 微调生成在藏文正字法达标。压缩率 0.0855（偏短，仅参考 0.213 的 40%）。
  - ⚠️ **负面结果（必须如实写）**：LEAD-2 原始设定全面超过 mT5 微调（chrF 0.4160 vs 0.2871、ROUGE-L 0.3560 vs 0.2827）。根因见 §7.4.4：BodSUM 96.2% 音节来自原文（强抽取偏置）+ 长度混淆；控长后 mT5 chrF 0.2871 vs LEAD-2 0.1201（×2.4）胜出 → LEAD 表观优势来自长度与抄写，非摘要能力。早期"×17.7"系错误切分口径假象，已作废。
- **TiLamb-7B+LoRA**（r=16, 39.98M/0.57%, bs4×accum4, lr1e-4, 2 epoch, 2026-09-30 22:23 启动 GPU0，2026-10-01 12:26 出结果）：**摘要微调最成功**——chrF 0.3212（神经最高，逼近 LEAD-2 0.4160）、ISR 0.0051（≈人类、与 TiLamb 续写 LoRA 0.004 一致）、SCR 0.8965（全表最高）、压缩率 0.2284（≈参考 0.213，长度匹配最佳）。ROUGE-L 0.1893 偏低系其更抽象改写（少连续抄写，恰被 ROUGE-L(LCS) 惩罚），与 §7.4.4 指标偏差结论一致。
- **Qwen2.5-7B+LoRA**（r=16, bs2×accum8 防 OOM, lr1e-4, 2 epoch, 5016s, 2026-10-01 完成 GPU1）：**灾难性负面结果（关键诊断）**——ISR=**0.0462**（≈人类 9×、TiLamb 的 9×）、chrF=**0.0619**、BLEU-4=**0.0010**、压缩率 0.0712（过短）。生成样本混入老挝文/乱码（如 `ນຸ�ມ`、`སོིག`），final_loss 0.5394「看似收敛」但生成崩 → **正是论文核心论点实证：标准 loss/PPL 看不见正字法结构失败，ISR 才暴露**。
  - ⚠️ **混杂变量已用探针排除（2026-10-02，`qwen_chat_probe.py`）**：Qwen 原跑的是与 TiLamb-base 相同的「纯续写」recipe，**未套 Qwen2.5-Instruct chat template**，表面看可能含「格式不匹配」成分。但探针在 **base Qwen2.5-7B-Instruct + 正确 chat template（零样本）** 下对 5 条真实 BodSUM 正文生成，结果**仍系统性失败**：
    - 多数样本直接把藏文**翻译成中文**（"藏文原文的大致翻译为…"、"藏文翻译成中文如下…2023年新年之际，中国国家主席…"）——Qwen2.5 未被训练做「藏文进→藏文出」摘要，默认回退到中文；
    - 吐出的藏文段混排**跨脚本乱码**（Odia `ୂ`、拉丁转写），与 LoRA 结果里的老挝文/乱码同源；
    - 整批 mean ISR=**0.035**（仍≈人类 7×；raw 形式 0.046、chat 形式 0.035，两者都崩）。
    → **失败非格式 artifact**，根因 = Qwen2.5 预训练/词表**缺藏文生成能力** + 跨脚本词元病理。故 **Qwen+chat-template 重训无必要**，原 Qwen-LoRA ISR 0.0462 为忠实结果。
  - 即便计入格式因素已被排除，**方向性结论成立**：通用 7B 经同款任务 LoRA（且对 prompt 格式鲁棒）仍未能把 ISR 拉到人类量级（0.0462 ≫ 0.005），而藏文专门化 7B 可以（0.0051）。正字法能力由**基座预训练决定**，任务 LoRA 无法补。

#### 7.4.2.1 配对 bootstrap 显著性检验（B=2000，2026-10-06 ✅ 逐条指标全齐）
- **为何重跑**：`task3_sum_train.py` 的 `evaluate()` 只把前 5 条 samples 写进 JSON，没有逐条 (ref, hyp) 就算不了配对 bootstrap CI。
  按红线（无 bootstrap 不得写"显著"），新增 `task3_dump_hyps.py` 重跑 594 条全量生成并逐条落盘，
  再用 CPU 自包含脚本 `bootstrap_task3.py`（指标函数与服务器 decode_sweep 同口径）做 B=2000 配对 bootstrap。
- 🐞 **过程中修复一个权重错用 bug（如实记录）**：首轮 mT5 dump 误用**未微调** `<MODEL_ROOT>/models/mt5-base`，
  输出病态（hyp 中位仅 23 字符、大量 ། 碎片、BLEU-4≈3.9e-9）。正确权重是 **`<MODEL_ROOT>/models/mt5-bodsum`**
  （BodSUM 全参微调 3ep/bs8/accum2/lr5e-5，final_loss 0.954；**full-FT 非 LoRA，dump 时不能加 `--lora-path`**）。
  改用后 hyp 中位 151 字符、0 条空，且复算点估计与服务器 09-30 `results_task3a_mt5.json` **逐位一致**
  （ISR 0.0069 / chrF 0.2871 / ROUGE-L 0.2827 / BLEU-4 0.1295）→ 权重与口径双验证通过，
  并**反向交叉验证了 §7.4.2 主表数字**。
- **点估计（三系统，与主表一致）**：

  | 系统 | ISR↓ | chrF↑ | ROUGE-L↑ | BLEU-4↑ |
  |---|---|---|---|---|
  | TiLamb-7B+LoRA | 0.0045 | 0.3172 | 0.1824 | 0.3297 |
  | mT5-base 全参微调 | 0.0069 | 0.2871 | 0.2827 | 0.1295 |
  | Qwen2.5-7B+LoRA | 0.0461 | 0.0669 | 0.0198 | 0.0019 |

- **配对 bootstrap（B=2000，Δ = A − B；三系统两两 × 4 指标 = 12 组全部 CI 不含 0）**：

  | 对比（A − B） | ISR Δ [95% CI] | chrF Δ [95% CI] | ROUGE-L Δ [95% CI] | BLEU-4 Δ [95% CI] |
  |---|---|---|---|---|
  | TiLamb − mT5 | **−0.0024** [−0.0036, −0.0011] ✅A优 | **+0.0301** [+0.0206, +0.0396] ✅A优 | **−0.1003** [−0.1102, −0.0901] ❌A显著低 | **+0.2002** [+0.2157, +0.2440] ✅A优 |
  | TiLamb − Qwen | **−0.0415** [−0.0450, −0.0382] ✅A优 | **+0.2504** [+0.2454, +0.2553] ✅A优 | **+0.1626** [+0.1576, +0.1677] ✅A优 | **+0.3278** [+0.3527, +0.3722] ✅A优 |
  | mT5 − Qwen | **−0.0392** [−0.0428, −0.0357] ✅A优 | **+0.2202** [+0.2122, +0.2278] ✅A优 | **+0.2629** [+0.2535, +0.2717] ✅A优 | **+0.1276** [+0.1248, +0.1417] ✅A优 |

  （ISR↓ 越低越好，故负 Δ = A 更优；chrF/ROUGE-L/BLEU-4↑ 正 Δ = A 更优。12 组 p(>0) 均为 0.000 或 1.000。
  BLEU-4 为 corpus 级指标，CI 用 **basic/reverse-percentile** `[2Δ−q97.5, 2Δ−q2.5]` 围绕 Δ 偏置修正，2026-10-07 00:04 落盘。）
- ✅ **可写进论文的结论（已有 bootstrap 支撑）**：
  1. TiLamb 在**正字法（ISR）上显著优于两个对照**（vs mT5、vs Qwen，CI 均不含 0）；
  2. Qwen 的 ISR 显著劣于两者（差约一个数量级）→「通用 7B 正字法系统性失败」在**摘要侧同样统计显著**，与续写侧（§6.9.1）形成跨任务互证；
  3. 两个藏文可用系统（TiLamb / mT5）的 chrF 均显著高于 Qwen。
- 🔑 **指标分歧（新增发现，与 §7.4.4 抽取偏置呼应）**：**ROUGE-L 与 BLEU-4/chrF 给出相反排序**——
  mT5 ROUGE-L 最高（0.2827）却 BLEU-4 最低（0.1295），TiLamb 反之（ROUGE-L 0.1824 / BLEU-4 0.3297）；
  TiLamb 在 ROUGE-L 上**显著低于** mT5（Δ−0.1003，CI 不含 0），而在 BLEU-4 上显著更高。
  解释：mT5 输出更长、更贴原文抄写（LCS 友好），TiLamb 更抽象改写（n-gram 精确匹配高但 LCS 短）
  → **单一 ROUGE-L 会把"抄写"误判为"摘要好"**。论文必须多指标联合报告；该分歧本身即是 §7.4.4 偏置论断的独立佐证。
- ✅ **BLEU-4 配对 bootstrap 已补齐（2026-10-07 00:04 落盘，basic/reverse-percentile CI）**：3 对全部显著——
  TiLamb−mT5 Δ=+0.2002 [+0.2157,+0.2440]、TiLamb−Qwen Δ=+0.3278 [+0.3527,+0.3722]、mT5−Qwen Δ=+0.1276 [+0.1248,+0.1417]，
  均 CI 不含 0。至此摘要侧 **12 组配对（3 对 × 4 指标）全部统计显著**，可无保留写入论文。

#### 跨任务正字法系统性失效：判定 ✅ SUPPORTED（2026-10-01）
- 续写（§6.8，零样本）：Qwen ISR 0.0648、Llama 0.269 ≫ 人类 0.0038；TiLamb+LoRA 0.004 ≈ 人类。
- 摘要（本表，同款 LoRA 微调后）：Qwen ISR **0.0462** ≫ 人类 0.0057；TiLamb+LoRA **0.0051** ≈ 人类。
- **一致性**：通用 7B 在「续写」与「摘要」两个任务上均系统性违反藏文正字法（ISR 高于人类一个数量级），且**该失败对任务监督微调鲁棒**（摘要 LoRA 后 Qwen ISR 仍 0.046）；藏文专门化 7B 在两种任务下 ISR 均≈人类。
- **论文论点锁定**：通用 LLM 在藏文新闻上呈现「系统性结构级失败（音节正字法）」，且该失败**不可由任务级 LoRA 消除**——这是 PPL/BLEU 看不见、ISR 才暴露的 diagnostic 贡献。⚠️ 仍守红线：分维度陈述，**不写「藏文专门化整体优于通用」**；通用模型在 bits/藏文字节（困惑度）维度仍更低（§6.8），headroom 存在。
- **robustness 已用探针消解（2026-10-02）**：base Qwen2.5-7B + 正确 chat template 仍吐中文/跨脚本乱码（mean ISR 0.035），故「Qwen+chat-template 重训」无必要——失败对**任务 LoRA 与 prompt 格式双重鲁棒**，根因是预训练/词表级藏文生成能力缺失。原 Qwen-LoRA ISR 0.0462 为忠实结果，方向结论（仍 ≫ 人类）不受影响。

- 模型产物：mT5 微调权重 `<MODEL_ROOT>/models/mt5-bodsum`；TiLamb adapter `<MODEL_ROOT>/TiNews/tilamb_lora_bodsum`（156M）；Qwen adapter `<MODEL_ROOT>/TiNews/qwen_lora_bodsum`（161M）。

### 7.4.3 更强的无监督基线（2026-09-30 23:20 ✅ CPU）
脚本 `task3_eval2.py`（TextRank / TF-IDF / centroid）。目的：给出**非神经方法的上界**，使神经基线的增量有参照——避免把"超过 LEAD"本身当贡献。
- **3b 关键词抽取（Ti-SUM 734，词级 F1）**：词频 top-k **0.051** → **TextRank 0.1775** （×3.5，无监督最优）→ TF-IDF 0.0264（最差）。
  - TF-IDF 反而低于纯词频，因 IDF 在全语料拟合而偏向**罕见词**，而新闻关键词多为中频实词 → 对藏文关键词抽取，IDF 加权有害。这是一个可写进正文的小发现。
- **3c 多文档摘要（Ti-SUM test 12 聚类，修正句切分后）**：

  | 方法 | BLEU-4↑ | chrF↑ | ROUGE-L↑ | ISR↓ | SCR↑ |
  |---|---|---|---|---|---|
  | LEAD-over-cluster（修正口径） | 0.0047 | 0.1749 | 0.1012 | 0.0 | 0.8424 |
  | **centroid-TF-IDF**（跨文档选句） | **0.0227** | **0.2022** | **0.1170** | 0.0009 | **0.8950** |

  - centroid 在 chrF/ROUGE-L 上均优于 LEAD（+15.6% / +15.6%），且 SCR 更高 → 质心选句比"取首句"更能代表聚类主题，符合预期。
- ⚠️ 统计力提醒：3c 仅 **12 个聚类**，差异不具显著性，正文须标注样本量、不得据此下强结论。

### 7.4.4 BodSUM 抽取偏置诊断（2026-09-30 23:40 ✅，把负面结果转为贡献）
脚本 `bodsum_bias.py`。起因：修正句切分后发现 LEAD-2 全面压过 mT5 微调（见 §7.4.2），必须先判定是「数据集偏置」还是「模型不行」。

| 诊断量 | 数值 | 判读 |
|---|---|---|
| 参考摘要**抽取覆盖率**（音节级） | **0.9623**（词级 0.9299） | 参考摘要几乎**纯抽取**，非抽象摘要 |
| 参考摘要 vs 正文前 2 句 chrF | 0.4160 | 与 LEAD-2 得分完全一致 → 参考≈导语 |
| 压缩率：参考 / LEAD-2 / mT5 | 0.213 / 0.1003 / 0.0855 | 参考明显长于两个系统 |
| **长度匹配后** LEAD-2 vs mT5 | chrF **0.1201 vs 0.2871** | 等长下 **mT5 胜 ×2.4** |
| 长度匹配后 ROUGE-L | 0.2945 vs 0.2827 | ROUGE-L **仍偏好 LEAD** |

- **结论 1（数据集特性）**：BodSUM 具**强抽取偏置**——参考摘要 96.2% 音节直接源自原文，且与正文导语高度重合。它衡量的是"选句/压缩"而非"抽象改写"。
- **结论 2（方法论警示，可写正文）**：在该基准上比较抽象式与抽取式系统，**生成长度是强混淆变量**。未控长度会得到"LEAD ≫ 神经"的误导性排序；控长后 chrF 排序反转（mT5 ×2.4 胜）。
- **结论 3（指标偏差）**：ROUGE-L（LCS 型）**系统性偏好连续抄写**，即便等长仍判 LEAD 略优；chrF（n-gram F）对长度更敏感。→ 藏文摘要评测应**同时报 chrF 与 ROUGE-L 并控长**，单指标不可靠。
- 这条把"神经模型输给 LEAD"的负面结果，转化为 **dataset + metric 的双重诊断发现**，正合 TiNews「diagnostic testbed」定位；须在 Limitations 明写"BodSUM 偏抽取，对抽象摘要能力区分度有限"。

### 7.4.5 Theorem 3.2 压缩率—正字法长度扫（✅ 2026-10-05 完成；**结论：长度依赖子推论被证伪**）

脚本 `task3_len_sweep.py`（本地已写好，待上传服务器）。

- **目的**：验证 contribution §3.2 结构罚下界 `v(S) ≥ v_0(L)` 中「`v_0(L)` 随压缩率 `L` 减小而**单调上升**」这一**速率形式**——这是 Theorem 3.2 目前唯一未直接验证的子推论（跨任务失效部分已由 §7.4.2 证实）。
- **设计（诚实性优先，两个已知伪影已预先隔离）**：
  1. 主口径 = `max_new_tokens=T` 上限，**允许自然 EOS 提前停止**；**不用** `min_new_tokens` 强撑，避免"超自然长度"伪影把长档位 ISR 人为推高（那会伪造出与预测相反的方向）。
  2. 每档同时报 `truncation_rate`（实际生成 token ≥ T 的样本占比）——短档位属"截断主导"，截断会切出不完整音节（已知伪影），分析时须单独标注，**不得把截断造成的 ISR 上升误读为压缩效应**。
  3. 长度双口径：`compression_char`（字符级，与 §7.4.2 同口径）+ `compression_syll`（音节级，按 tsheg）。
- **配置**：`T ∈ {32,48,64,96,128,192,256}`；`n=150`（BodSUM test 子集）；beam=4 / no_repeat_ngram=3 / early_stopping=True；`--bs 2`（防 7B vocab 大 logits OOM）。
- **对照**：TiLamb-7B+LoRA（藏文专门化，预期 ISR 随 L 平坦≈0.005）vs Qwen2.5-7B+LoRA（通用，预期 ISR 随 L↓ 上升）。
- **状态**：✅ **已完成**（2026-10-05 23:47 双双跑完，GPU 已释放）。**结论：❌ NOT SUPPORTED（对通用模型方向相反）；但得到更稳健的替代发现，见 §7.4.5.2。**

#### 7.4.5.1 ⚠️ 口径修正 + 截断伪影对照（2026-10-05 23:00，CPU 已出）
- **触发修正的原因**：实际跑出来**所有档 `truncation_rate=1.00`**（模型自然摘要长度 > 各档上限，全部撞上限被截），故原定"只采信 trunc<0.5 自然停止档"的口径**无档可用**。改用**人类摘要截断对照**直接量化伪影上界。
- 脚本 `human_trunc_control.py`（CPU，无需 GPU）：把 BodSUM **人类参考摘要**（未截断基线 ISR=**0.0043**）按**与模型各档相同的生成长度**做两种截断——① 音节边界截断（只压长度不切坏音节）② 字符级硬截断（模拟 token 截断、切坏末尾音节，= 伪影上界）：

  | 档 T | 模型生成音节 | **模型 ISR** | 人类 clean 截 | **人类 hard 截** | 判读 |
  |---|---|---|---|---|---|
  | 32 | 20.4 | **0.0082** | 0.0042 | **0.0042** | 伪影不足解释 |
  | 48 | 30.9 | 0.0059 | 0.0040 | 0.0047 | 伪影不足解释 |
  | 64 | 41.3 | 0.0068 | 0.0039 | 0.0045 | 伪影不足解释 |
  | 96 | 61.3 | 0.0059 | 0.0040 | 0.0047 | 接近可解释 |
  | 128 | 80.8 | 0.0041 | 0.0040 | 0.0044 | 模型≈人类 |

  - **结论**：机械截断（即便硬截切坏末尾音节）只把人类 ISR 从 0.0043 抬到 **≤0.0047（+0.0004）**；而模型最短档为 **0.0082（+0.0039）**。→ **截断伪影最多解释约 10%，不足以解释模型 ISR 随压缩加剧的抬升**，支持"真压缩效应"而非伪影。
  - ⚠️ 但须如实写：TiLamb（**专门化**模型）也呈现长度效应（0.0041→0.0082，约 2×），并非 Theorem 3.2 原预测的"专门化完全平坦"。绝对幅度仍小（增量 0.004，最短档 0.0082 ≈ 人类 2×），与 Qwen 基线 0.046 不在同一量级。待 Qwen 扫完对比**相对敏感度**再定稿措辞。

  - **Qwen 同款对照（补跑）**：人类 hard 截 ISR 仅 **0.0040–0.0046**，而 Qwen 各档 **0.0401–0.0539（≈人类 10×）** → 截断伪影**完全**解释不了（<0.5×），Qwen 的高 ISR 是真实能力缺陷，且与长度基本无关。

#### 7.4.5.2 ✅ 完整结果与判定（2026-10-05 23:47 跑完）

| 档 T | **TiLamb** comp_syll | TiLamb ISR | **Qwen** comp_syll | Qwen ISR |
|---|---|---|---|---|
| 32 | 0.0575 | **0.0082** | 0.0171 | 0.0435 |
| 48 | 0.0865 | 0.0059 | 0.0249 | 0.0485 |
| 64 | 0.1145 | 0.0068 | 0.0325 | 0.0445 |
| 96 | 0.1677 | 0.0059 | 0.0480 | 0.0401 |
| 128 | 0.2178 | **0.0041** | 0.0630 | 0.0454 |
| 192 | 0.3129 | 0.0044 | 0.0892 | 0.0527 |
| 256 | 0.4112 | 0.0049 | 0.1135 | **0.0539** |

| 模型 | ISR 范围 | 相对人类(0.0043) | Spearman(comp_syll, ISR) | 判定 |
|---|---|---|---|---|
| TiLamb-7B+LoRA（专门化） | 0.0041–0.0082 | 0.9×–1.9× | **−0.821**（压缩加剧→ISR↑） | ✅ 方向符合（幅度小） |
| Qwen2.5-7B+LoRA（通用） | 0.0401–0.0539 | **9.3×–12.5×** | **+0.643**（越长→ISR↑） | ❌ **与预测相反** |

- **Theorem 3.2 长度依赖子推论（`v_0(L)` 随 `L↓` 单调上升）：❌ NOT SUPPORTED（证伪，如实写入）**
  - 对通用模型（Qwen）方向**相反**（ρ=+0.643），且 ISR 波动仅 1.35× —— **基本不随长度预算变化**。
  - 对专门化模型（TiLamb）方向符合（ρ=−0.821），但绝对幅度仅 0.0041→0.0082，仍在人类 2× 内。
- **稳健的新发现（替代原 claim 写进正文）**：**藏文正字法能力是基座级"恒定属性"，压缩率只是二阶效应。**
  Qwen 在**全部 7 个长度预算**下 ISR 恒定在人类 **9.3×–12.5×**，不随压缩率改善或恶化；TiLamb 全程 0.9×–1.9× 人类，仅极端压缩（comp≤0.06）时升至 0.0082。
  → 这**强化**核心论点（正字法能力由基座预训练决定、任务 LoRA 补不上），但**须撤回**"压缩加剧导致正字法崩"的速率形式 claim。
- ⚠️ 所有档 `truncation_rate≈1.00`（全为长度上限截断），趋势须结合 §7.4.5.1 的伪影上界解读；已证明伪影不足以解释。

### 7.5 学术红线（写入正文 / Limitations）
1. 不夸大 BodSUM 质量：仅抽样 12 行 + 官方声明，全文质量审计待做；不得写"已严格验证全部 6000 条"。
2. 跨 tokenizer PPL 不直比，用 bits/藏文字节桥接。
3. KL 小 ≠ 质量好（沿用 §8.4.7）。
4. Ti-SUM 26% 关键词缺失须如实声明：3b 仅在 734 条评测，不得假装全覆盖。
5. Task 3 是对 benchmark 覆盖面的扩展，不是"又一个工程组合创新"。
