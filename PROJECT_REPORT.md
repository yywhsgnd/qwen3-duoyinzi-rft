# Qwen3-8B古诗多音字平仄消歧
## 数据、训练、评测中的具体贡献与可复现结果

---

## 1. 项目概述

本项目解决**古诗语境中多音字平仄判断**任务：给定完整古诗上下文和词典义项，判断目标多音字在该语境下应读**平声**还是**仄声**。

- **基础模型**：Qwen3-8B
- **训练框架**：[veRL](https://github.com/volcengine/verl)（字节跳动开源的 LLM 强化学习框架）
- **训练算法**：GRPO（Group Relative Policy Optimization）
- **工作范围**：数据构造、全诗上下文 Prompt 设计、自定义 Reward、训练配置、自动化评测、bad case 分析

> 基于 veRL 和 Qwen3-8B 构建古诗多音字平仄消歧 RFT 流程，覆盖数据构造、全诗上下文 Prompt、自定义 Reward、强化训练、自动化评测和 bad case 分析。

---

## 2. 数据构建中的具体贡献

### 数据来源
- 诗词数据：搜韵诗词库（含平仄标注的古诗文本）
- 词典数据：汉语大词典（多音字读音、释义、平仄分类）

### 实际完成的工作

1. **多源数据整合**：关联诗词文本、目标多音字、目标位置、词典义项和平仄标签
2. **全诗上下文构造**：将目标字放入完整诗歌上下文中，高亮目标诗句（使用 `【】` 标记），避免模型仅依据单字做判断
3. **Prompt 设计**：
   - **System Prompt**：角色定义 + 输出格式约束（`#### 平声/仄声`）
   - **User Prompt**：全诗上下文 + 高亮目标诗句 + 目标字位置 + 平/仄两边词典释义
4. **数据格式转换**：适配 Qwen3（chat template）和 veRL（parquet + reward_model 字段）
5. **数据过滤**：保留平声和仄声释义均存在的多音字样本
6. **8:1:1 分层划分**，固定随机种子 `random_state=42`，保证数据划分可重复

### 数据规模

| 数据集 | 样本数 | 平声比例 | 备注 |
|--------|--------|----------|------|
| 训练集（全诗上下文） | 304,151 行 | 63.1% | train.parquet 行数，数据源构成待进一步核验 |
| 验证集 | 37,792 行 | 62.6% | valid.parquet 行数 |
| hanyuda 测试集（全诗） | 6,000 | 65.6% | hanyuda 测试集覆盖 263 个唯一目标字 |
| AA 测试集（全诗，人工标注） | 30 | 50.0% | — |
| unzidian 测试集（全诗） | 6,000 | 65.6% | — |

### 数据示例

**诗句**：河津传五漏 / 雾雨入千家 / 白日鸣饥鼠 / **【深更歇怒蛙】** / 境空愁稍进 / 诗短意能加 / 客路多风雨 / 深忧斗柄斜

**目标字**：`更`（目标句第2字）

| 平声义项 | 仄声义项 |
|----------|----------|
| 改正、改变、调换、替代… | 副词：再、又、愈加… |

**原始标签**：`平声`

**模型输入**：
```
【System】你是一个古诗平仄分析专家，请根据诗句上下文语义和词典释义…
【User】全诗上下文…目标诗句（已用【】标出）里第2个字 '更' 的平仄。
参考资料：更字释义：【平声】1.改正 2.改变…【仄声】1.再、又…
```

**期望输出**：`#### 平声`

> 完整样例见 `examples/processed_examples.jsonl`。

---

## 3. 模型训练中的具体贡献

项目没有重新实现强化学习框架，而是基于 veRL 完成任务适配和训练实验。

### veRL 框架提供的基础能力

- GRPO 训练主流程、Ray 分布式调度
- FSDP 模型分片、vLLM rollout
- Reward Manager、checkpoint 与日志管理

### 我完成的训练工作

1. 将古诗多音字任务接入 veRL
2. 设计训练 Prompt（System/User）和结构化答案格式（`#### 平声/仄声`）
3. 编写任务专用 Reward Function（3级评分）
4. 配置 Qwen3-8B 的 GRPO 训练超参数
5. 执行训练和管理 checkpoint
6. 使用 WandB 跟踪训练过程，分析 reward、格式遵循率和模型输出变化

### Reward Function 设计

| 输出情况 | Reward |
|----------|--------|
| 格式正确且答案正确（`#### 平声/仄声` 且与 GT 一致） | +1 |
| 格式正确但答案错误 | 0 |
| 输出无法解析（无有效 `####` 标记） | -1 |

三级 Reward 同时约束**任务正确性**和**结构化输出能力**：避免模型虽然给出相关解释，但无法稳定输出可解析答案。

### 训练配置

| 参数 | 值 |
|------|-----|
| 基础模型 | Qwen3-8B |
| 算法 | GRPO |
| 训练框架 | veRL |
| GPU | 3 × NVIDIA GPU |
| Batch size | 504 |
| Learning rate | 1e-6 |
| GRPO group size（rollout.n） | 4 |
| PPO epochs | 4 |
| Max prompt length | 1,500 tokens |
| Max response length | 2,048 tokens |
| KL 系数 | 0.0005（low_var_kl） |
| Rollout 引擎 | vLLM（temperature=0.8） |
| Reward Manager | prime（异步并行） |
| 训练精度 | bfloat16 |

> 公开运行入口见 `configs/train_config_public.sh`。

---

## 4. 模型评测中的具体贡献

### 评测方法

1. **统一评测条件**：Base 和 RFT 使用相同测试集、相同 Prompt 模板、相同答案解析规则和相同生成参数（temperature=0，贪心解码）
2. **多维指标**：Accuracy、Precision、Recall、Macro-F1、Parse Fail
3. **两种准确率**：
   - **有效样本 Accuracy**：仅在可成功解析为“平声”或“仄声”的样本上计算
   - **严格端到端 Accuracy**：正确样本数 / 测试集总样本数，Parse Fail 按错误计入
4. **错误分类**：将 Base 和 RFT 的预测差异拆分为 both / base_only / RFT_only
5. **可视化报告**：生成可交互 HTML bad case 报告

### 评测结果

| 测试集 | 模型 | 样本数 | 有效样本 Acc | 严格端到端 Acc | Macro-F1 | Parse Fail |
|--------|------|--------|-------------|---------------|----------|------------|
| AA 全诗 | Base | 30 | 89.66% | 86.67% | 89.45% | 1 |
| AA 全诗 | RFT | 30 | 96.67% | 96.67% | 96.66% | 0 |
| hanyuda 全诗 | Base | 6,000 | 86.40% | 85.88% | 84.95% | 36 |
| hanyuda 全诗 | RFT | 6,000 | 87.21% | 87.08% | 85.95% | 9 |
| unzidian 全诗 | Base | 6,000 | 82.09% | 73.43% | 78.73% | 633 |
| unzidian 全诗 | RFT | 6,000 | 82.94% | 82.92% | 80.92% | 2 |

> **口径说明**：有效样本 Accuracy 和 Macro-F1 仅在成功解析为“平声”或“仄声”的样本上计算；严格端到端 Accuracy 使用全部测试样本作为分母，Parse Fail 按错误计入。数字来源为 Base 和 RFT 的原始评测日志（`verl/eval_log/quanshi/`）及 `bad_case_analysis.md` 中的 sklearn 分类报告。**所有主结果均使用原始测试标签，未使用 DeepSeek 结果修改 Ground Truth。**

### 结果概要

RFT 在三个测试集上的主要收益同时体现在任务判断和格式稳定性上。hanyuda 全诗严格端到端准确率由 85.88% 提升至 87.08%（+1.20 个百分点），并净减少 72 条错误；unzidian 全诗的 Parse Fail 从 633 降至 2，使严格端到端准确率由 73.43% 提升至 82.92%（+9.49 个百分点）。AA 仅含 30 条样本，其结果作为小规模案例验证，不作普遍性结论。

### Bad Case 与标签质量分析

hanyuda 全诗错误拆分：

| 类别 | 数量 | 说明 |
|------|------|------|
| Base 总错误 | 847 | 含 36 个 Parse Fail |
| RFT 总错误 | 775 | 含 9 个 Parse Fail |
| both（双方均错） | 644 | — |
| base_only（RFT 修正） | 203 | — |
| RFT_only（RFT 退化） | 131 | — |
| RFT 净减少错误 | 72 | 203 − 131 = 72 |

RFT 修正了 203 条 Base 错误，同时新增了 131 条错误，最终净减少 72 条错误。

人工查看 both 和 base_only 后，发现原始标签中存在一些明显或高度疑似的语境标注问题，尤其集中在“当、相、行、看”等高频多音字。

为进一步了解标签质量，对 hanyuda 中 644 条 Base 和 RFT 共享错误样本进行了 DeepSeek 三次独立判断（temperature=0）。其中 582 条三次结果一致，62 条不一致；在三次结果一致的样本中，417 条与原始标签冲突，占 71.65%。该结果提示原始测试标签中可能存在较明显的语境标注噪声。DeepSeek 结果仅作为辅助分析信号，未直接替代 Ground Truth，也未用于重新计算主结果。

#### 代表性 Bad Case

**案例 1**：诗句 `躬耕与行役`，目标字 `行`
- 原始标签：`平声`
- Base 预测：`仄声` | RFT 预测：`仄声`
- DeepSeek 三次：`仄声` / `仄声` / `仄声`
- 分析：“行役”表示出行或服役，“行”在此处为“出行”义，符合仄声义项

**案例 2**：诗句 `回看飞瀑下`，目标字 `看`
- 原始标签：`平声`
- Base 预测：`平声` | RFT 预测：`仄声`
- DeepSeek 三次：`仄声` / `仄声` / `仄声`
- 分析：“回看”为“回头看”，“看”在此处为“看见、观察”义，符合仄声的主要用法

**案例 3**：诗句 `清心凉除暑`，目标字 `除`
- 原始标签：`仄声`
- Base 预测：`平声` | RFT 预测：`平声`
- DeepSeek 三次：`平声` / `平声` / `平声`
- 分析：“除暑”意为“去除暑热”，“除”为“去除”义，属于平声义项

> 以上案例均来自真实数据。完整 bad case 见 `reports/` 目录下的 HTML 报告和 `examples/representative_bad_cases.md`。

项目结果不仅包括模型性能对比，还包括错误转移分析、格式失败分析和标签质量诊断。对于存在争议的样本，保留原始结果并单独说明标签风险，不使用模型自动结果替代 Ground Truth。

---

## 5. 可复现范围

公开仓库提供可复现的数据处理、Prompt 构造、Reward 计算、指标计算和 Bad Case 分析流程。完整训练数据与模型 checkpoint 因授权和存储限制未公开，因此原始 8B 训练结果属于**可核验结果**（可通过公开的指标文件和日志摘要进行核验），而非仅依赖公开仓库即可完整重训的结果。

### 可核验内容

- 原始评测指标（见 `results/raw_metrics.csv` 和 `results/summary.json`）
- 错误拆分统计（见 `results/bad_case_statistics.csv`）
- 以上数字可通过 `tests/` 中的单元测试和 `evaluation/` 中的脚本交叉验证

### 可复现流程

| 步骤 | 说明 | 依赖 |
|------|------|------|
| 查看数据样例 | `examples/processed_examples.jsonl` | 无需额外依赖 |
| Reward 单元测试 | `python tests/test_reward.py` | 无需 GPU |
| 指标计算验证 | `python tests/test_metrics.py` | 无需 GPU |
| 数据处理流程 | `python src/data/build_dataset.py` | 需自有数据 |
| Bad case 分类 | `python evaluation/classify_bad_cases.py` | 需预测结果文件 |
| 完整训练 | `bash configs/train_config_public.sh` | 需完整训练数据 + Qwen3-8B + 3 GPU |

### 公开材料清单

| 材料 | 路径 |
|------|------|
| 数据处理脚本 | `src/data/` |
| 脱敏数据样例 | `examples/` |
| 自定义 Reward Function | `src/reward/duoyinzi_reward.py` |
| veRL 训练配置（公开版） | `configs/train_config_public.sh` |
| Base/RFT 评测脚本 | `evaluation/evaluate_model_vllm.py` |
| 错误分类脚本 | `evaluation/classify_bad_cases.py` |
| 指标计算脚本 | `evaluation/compute_metrics.py` |
| Bad case HTML 报告 | `reports/` |
| 环境与运行说明 | `REPRODUCE.md` |

---

## 6. 局限性

- 训练数据原始标签基于字音和词典自动生成，非逐条人工标注，存在标签噪声
- 古诗创作年代跨度大，部分多音字的古音与现代标准存在差异
- AA 测试集仅 30 条，统计效力有限，结果作为小规模验证参考
- 部分样本的读音判断存在语义两可情况，需要进一步专家标注
- 完整训练数据和模型 checkpoint 未公开，训练结果属于可核验但不可完整重训的结果
