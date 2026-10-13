# `data/` — 实验结果与系统输出

本目录全部为**本研究计算出的指标 / 模型输出**，可直接用于复现论文表格。
统一约定：所有 JSON 为点估计或聚合结果；`raw_hyps/` 为逐条 ref/hyp；
模型权重路径中的 `<MODEL_ROOT>` 为服务器模型目录占位（模型身份以
`YoLo2000/TiLamb-7B`、LoRA 名等字段记录，绝对路径已脱敏）。

## 1. 任务2 · 续写（G1, n=200, beam4）
| 文件 | 含义 |
|---|---|
| `results_g1_llama.json` `results_g1_qwen.json` `results_g1_tilamb_base.json` `results_g1_tilamb_lora.json` | 四模型点估计：bits/藏文字节、PPL(自身 tokenizer)、ISR |
| `results_g1_*_hyps.jsonl`（在仓库外的原始 dump 中） | 逐条续写 hyp |
| `results_bootstrap_gen.json` | 续写侧配对 bootstrap（B=2000, basic/reverse-percentile CI） |
| `results_budget.json` `results_alpha_sweep.json` | 预算 / α 扫（teacher-forcing p_max 收集协议） |

## 2. 任务3 · 摘要 + 关键词（n=594, beam4）
| 文件 | 含义 |
|---|---|
| `results_task3a_{tilamb,qwen,mt5}.json` | 三系统点估计：ISR / scr / chrF / ROUGE-L / BLEU-4 |
| `results_bootstrap_task3.json` | 摘要侧 12 组配对 bootstrap |
| `results_g4_mt5.json` `results_g4_mt5_sc.json` | mT5 enc-dec 对照（full-FT，权重 `<MODEL_ROOT>/models/mt5-bodsum`，final_loss 0.954） |
| `results_lensweep_{tilamb,qwen}.json` | 七档长度扫（Theorem 3.2 证伪证据，含 `mean_gen_syll`/`compression_syll`/`truncation_rate`） |
| `results_sweep.json` | 解码超参扫 |

## 3. 任务1 · 7 类分类（N=1049）
| 文件 | 含义 |
|---|---|
| `results_cino-base-v2*.json` `results_tibetan_bert*.json` `results_textcnn*.json` `results_textcnn_first_sent.json` `results_textcnn_lead200.json` | CINO / TibetanBERT / TextCNN 在 full 与 lead200 设置下的混淆矩阵与每类 F1 |
| `results_lstm.json` `results_lightweight.json` | 轻量基线 |

## 4. 跨 tokenizer PPL 桥接（§3）
| 文件 | 含义 |
|---|---|
| `results_ppl_base.json` `results_ppl_lora.json` `results_ppl_bootstrap.json` | bits/藏文字节 桥接点估计与 bootstrap |
| `ppl_peritem.jsonl` `ppl_lora_peritem.json` | 逐条 PPL（用于桥接复算） |

## 5. 人工评测集（本研究自建，200 条）
| 文件 | 含义 |
|---|---|
| `human_eval_200.jsonl` `human_eval_200.xlsx` | 200 条评测材料 + rater 评分（实测数据）；维度：正字法 / 连贯 / 忠实 / 信息量 |

## 6. 系统输出（摘要逐条 ref/hyp，写稿抽查用）
`raw_hyps/hyps_task3a_{tilamb,qwen,mt5}.jsonl`（各 594 条）+ `hyps_task3a_tilamb_head.jsonl`（前段样本）。

## 不在本目录（第三方原始语料，另行获取）
- `bodsum_test.jsonl`（BodSUM 测试集）、`tisum_parsed.jsonl`（Ti-SUM 扩展集）、
  `human_eval_contexts.jsonl`（评测上下文）属第三方 / 新闻原文，按各自许可获取，
  **未随本仓库分发**；详见仓库根 `README.md` 的「数据来源」一节。
