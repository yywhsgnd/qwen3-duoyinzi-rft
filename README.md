# Qwen3-8B古诗多音字平仄消歧

本项目基于 [veRL](https://github.com/volcengine/verl) 和 Qwen3-8B，针对古诗语境中的多音字平仄判断任务，构建了从数据处理、全诗上下文 Prompt、自定义三级 Reward、GRPO 训练，到 Base/RFT 统一评测和 bad case 分析的完整流程。

## 建议首先阅读

👉 [我的具体贡献与可核验结果](./CONTRIBUTIONS_AND_RESULTS.md)

该文档用真实数据、Reward 和 Bad Case 例子，集中说明我在数据、训练和评测中的具体工作及项目结果。

---

## 我的主要工作

- **数据构造与 Prompt 设计**：整合诗词、词典、平仄标签，构造全诗上下文 Prompt，高亮目标句
- **任务专用 Reward Function**：3 级评分（+1/0/-1），同时约束正确性和结构化输出
- **veRL/GRPO 训练配置**：基于 Qwen3-8B 完成训练、checkpoint 管理和 WandB 追踪
- **vLLM 批量评测**：Base/RFT 统一评测，多维指标，Parse Fail 独立统计
- **错误拆分与标签质量分析**：both / base_only / RFT_only 分类，识别标签噪声
- **可视化 bad case 报告**：交互式 HTML，含完整 Prompt 和模型推理链

> veRL 框架提供了 GRPO 训练流程、Ray 调度、FSDP 分布式训练和 vLLM rollout 等基础能力。本项目未重新实现上述组件。

---

## 核心结果

| 测试集 | 模型 | 严格端到端 Acc | Parse Fail | Macro-F1 |
|--------|------|:---:|:---:|:---:|
| AA 全诗（30条） | Base | 86.67% | 1 | 89.45% |
| AA 全诗（30条） | RFT | **96.67%** | **0** | **96.66%** |
| hanyuda 全诗（6000条） | Base | 85.88% | 36 | 84.95% |
| hanyuda 全诗（6000条） | RFT | **87.08%** | **9** | **85.95%** |
| unzidian 全诗（6000条） | Base | 73.43% | 633 | 78.73% |
| unzidian 全诗（6000条） | RFT | **82.92%** | **2** | 80.92% |

> **严格端到端准确率** = 正确数 / 总样本数（Parse Fail 按错误计入）。所有结果使用原始测试标签，未用 DeepSeek 校正。RFT 净减少 72 条错误（修正 203 条 + 新增 131 条）。

**可复现性说明**：公开仓库提供可复现的数据处理、Prompt 构造、Reward 计算、指标计算和 Bad Case 分析流程。完整训练数据与模型 checkpoint 因授权和存储限制未公开，原始 8B 训练结果属于可核验结果（可通过公开的指标文件和日志摘要核验），而非仅依赖公开仓库即可完整重训的结果。

---

## 快速导航

| 你想了解 | 看这里 |
|----------|--------|
| 完整项目说明与个人贡献 | [PROJECT_REPORT.md](PROJECT_REPORT.md) |
| 数据样例 | [examples/](examples/) |
| Reward 代码 | [src/reward/duoyinzi_reward.py](src/reward/duoyinzi_reward.py) |
| 训练配置 | [configs/train_config_public.sh](configs/train_config_public.sh) |
| 评测代码 | [evaluation/](evaluation/) |
| Bad case 报告 | [reports/](reports/) |
| 复现说明 | [REPRODUCE.md](REPRODUCE.md) |
| 贡献归属 | [ATTRIBUTION.md](ATTRIBUTION.md) |

---

## 技术栈

Qwen3-8B · veRL · GRPO · vLLM · FSDP · Ray · WandB

---

## 快速验证

```bash
# 无需 GPU，1 分钟验证核心逻辑
python tests/test_reward.py        # Reward 三级评分逻辑
python tests/test_answer_parser.py # 答案解析与 Parse Fail 检测
python tests/test_metrics.py       # 严格准确率计算
```
