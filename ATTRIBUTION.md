# 贡献归属（Attribution）

## veRL 框架提供的基础能力

本项目基于 [veRL](https://github.com/volcengine/verl) 完成，以下核心基础设施由 veRL 提供，**不是本项目的个人贡献**：

| 能力 | 说明 |
|------|------|
| 强化学习训练主流程 | PPO/GRPO 的 Ray 分布式训练循环 |
| GRPO 算法实现 | Group Relative Policy Optimization 的 advantage 估计与策略更新 |
| Ray 分布式调度 | Ray worker 管理、资源分配 |
| FSDP | PyTorch FSDP 模型分片、梯度同步 |
| vLLM rollout | 高吞吐量推理引擎 |
| Reward Manager | PrimeRewardManager 异步并行评分基础框架 |
| Checkpoint 与日志 | 模型保存/恢复、WandB 指标上报 |

**本项目没有独立实现 GRPO、FSDP、Ray 或 vLLM。**

## 本项目的自定义工作

| 模块 | 工作内容 | 贡献程度 |
|------|---------|----------|
| 任务定义 | 古诗多音字平仄消歧的问题建模与数据构造 | 独立完成 |
| Prompt 设计 | System/User 分层 Prompt，全诗上下文高亮 | 独立完成 |
| 输出协议 | `#### 平声/仄声` 结构化输出格式 | 独立完成 |
| Reward Function | 3 级 rule-based 评分（+1/0/-1） | 独立完成 |
| 训练配置 | Qwen3-8B 的 GRPO 超参配置 | 基于 veRL 配置模板修改 |
| 评测脚本 | vLLM 批量推理、多维指标、Parse Fail 检测 | 独立完成 |
| Bad Case 分类 | both/base_only/RFT_only 三维错误拆分 | 独立完成 |
| 标签质量分析 | DeepSeek 三次复核辅助判断标签噪声 | 独立完成 |
| 可视化报告 | 交互式 HTML bad case 报告 | 独立完成 |

## 模型与数据来源

- **基础模型**：Qwen3-8B（阿里通义千问团队开源）
- **训练框架**：veRL（字节跳动 Seed 团队开源）
- **词典数据**：汉语大词典（研究用途）
- **诗词数据**：搜韵诗词库（研究用途）

本项目的自定义代码基于 veRL 的 Apache 2.0 许可证。
