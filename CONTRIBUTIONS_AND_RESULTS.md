# Qwen3-8B古诗多音字平仄消歧
## 我的具体贡献与可核验结果

---

## 项目一句话介绍

本项目使用 Qwen3-8B 和 veRL/GRPO，根据完整古诗上下文和词典义项，判断多音字在具体语境中应读平声还是仄声。veRL 提供强化学习训练框架（GRPO、Ray、FSDP、vLLM），我负责将具体任务接入 veRL，完成数据构造、Prompt 设计、自定义 Reward、训练配置、Base/RFT 统一评测和 Bad Case 分析。

---

## 我的具体贡献

| 环节 | 我具体做了什么 | 代表性产出 |
|------|---------------|-----------|
| 数据 | 将诗词、目标字、目标位置、词典义项和平仄标签整理为训练样本，构造完整诗歌上下文并高亮目标句 | 数据处理脚本、Prompt 模板、脱敏数据样例 |
| 训练 | 编写三级规则 Reward，将任务接入 veRL 并配置 Qwen3-8B 的 GRPO 训练参数 | Reward 代码、训练配置、WandB 记录 |
| 评测 | 编写 vLLM 批量评测脚本，统一比较 Base 与 RFT，拆分 Both / Base only / RFT only 三类错误 | 结果表、Bad Case HTML、错误分析 |

---

## 数据构造例子

**原始诗句**：郭外纵横亩 / 秋田远近山 / 三农方待穫 / 一雁下庾关 / 似诉风霜苦 / 焉知稼穑艰 / **【躬耕与行役】** / 相视未能閒

**目标字**：`行`（目标句第 4 字）

**模型输入**包含：
- 完整 8 句诗歌上下文，目标句用 `【】` 高亮
- 目标字位置（第 4 字）
- “行”的平声义项（行走、前往、道路…）和仄声义项（刚强貌、品行…）
- 明确要求仅根据语义判断，不使用整句格律规则

**期望输出**：`#### 平声` 或 `#### 仄声`

> 我将原始诗词和词典信息转换为带完整上下文、目标位置和候选义项的结构化训练样本，避免模型只根据单字或固定格律猜测答案。

[查看脱敏数据样例](./examples/processed_examples.jsonl)

---

## Reward 设计例子

| 模型输出 | Reward |
|----------|--------|
| `#### 平声`，且答案正确 | +1 |
| 格式正确（`#### 平声` 或 `#### 仄声`），但判断错误 | 0 |
| 没有输出可解析答案 | −1 |

该 Reward 同时约束答案正确性和结构化输出稳定性：模型不仅要判断平仄，还必须输出程序可以稳定解析的答案。

> veRL 提供 GRPO、Ray、FSDP 和 vLLM 等基础设施；我完成任务 Prompt、自定义 Reward、训练配置和实验运行。

[Reward 代码](./src/reward/duoyinzi_reward.py) · [训练配置](./configs/train_config_public.sh)

---

## 评测与结果

| 测试集 | Base 严格准确率 | RFT 严格准确率 | Base Parse Fail | RFT Parse Fail |
|--------|---------------|---------------|-----------------|-----------------|
| AA 全诗 | 86.67% | 96.67% | 1 | 0 |
| hanyuda 全诗 | 85.88% | 87.08% | 36 | 9 |
| unzidian 全诗 | 73.43% | 82.92% | 633 | 2 |

> 严格准确率使用全部测试样本作为分母，Parse Fail 按错误计入。所有结果均使用原始测试标签，未使用 DeepSeek 结果修改 Ground Truth。

RFT 在 hanyuda 中修正了 203 条 Base 错误，同时新增 131 条错误，最终净减少 72 条错误。最明显的变化是结构化输出稳定性：unzidian 的 Parse Fail 从 633 条降至 2 条。

[查看完整结果](./results/raw_metrics.csv)

---

## Bad Case 分析例子

**目标句**：【躬耕与**行**役】　**目标字**：行　**原始标签**：仄声

**Base 预测**：平声　**RFT 预测**：平声

**语义分析**：“行役”表示出行、服役或因公务奔波，与平声义项“行走、前往”一致；仄声义项“刚强貌”与该语境不符。

这个案例表明，模型与原标签不一致不一定全部属于模型错误。因此我进一步将错误拆分为 Both、Base only 和 RFT only，并结合完整上下文检查标签质量。

| 类型 | 数量 |
|------|------|
| Both（双方均错） | 644 |
| Base only（RFT 修正） | 203 |
| RFT only（RFT 退化） | 131 |
| RFT 净减少 | 72 |

对 644 条共享错误进行 DeepSeek 三次辅助判断后，582 条结果一致，其中 417 条与原标签冲突。该结果仅用于提示标签噪声，不用于修改主结果。

[代表性 Bad Case](./examples/representative_bad_cases.md) · [交互式报告](./reports/hanyuda_bad_cases_sample.html)

---

## 可复现范围

**公开仓库可以直接复现**：数据样例格式转换、Prompt 构造、Reward 计算、答案解析、Accuracy 与 Parse Fail 计算、Bad Case 分类和 HTML 生成。

**公开仓库不能单独完整重训**：完整训练数据未公开、Qwen3-8B 训练 checkpoint 未公开、原始服务器训练环境未公开。

公开仓库提供完整的处理、Reward 和评测流程；原始 8B 训练结果可通过结果文件和配置核验，但不能仅依赖公开仓库完成全部重训。

---

## 核心入口

- [完整项目报告](./PROJECT_REPORT.md)
- [复现说明](./REPRODUCE.md)
- [数据样例](./examples/processed_examples.jsonl)
- [Reward 代码](./src/reward/duoyinzi_reward.py)
- [评测代码](./evaluation/)
- [Bad Case 报告](./reports/hanyuda_bad_cases_sample.html)
