# 复现说明（Reproduce）

## 可复现范围

公开仓库提供可复现的数据处理、Prompt 构造、Reward 计算、指标计算和 Bad Case 分析流程。

**可核验但不可完整重训的内容**：原始 8B 训练结果（见 `results/raw_metrics.csv`）可通过公开的指标文件和日志摘要进行核验，但由于完整训练数据与模型 checkpoint 因授权和存储限制未公开，不能仅依赖公开仓库完成完整重训。

## 环境要求

- Python 3.10+
- CUDA 12.4+（训练需要 3+ GPU，评测需要 1+ GPU）
- veRL 训练框架：[安装文档](https://github.com/volcengine/verl)
- Qwen3-8B 模型：[HuggingFace](https://huggingface.co/Qwen/Qwen3-8B)

## 安装

```bash
cd qwen3-duoyinzi-rft
pip install -r requirements.txt

# 配置环境（根据需要填写）
export MODEL_PATH="/path/to/Qwen3-8B"
export DATASET_ROOT_DIR="/path/to/dataset"
export WANDB_API_KEY="${WANDB_API_KEY:-}"  # 可选
```

## 最小验证（无需 GPU，1 分钟）

```bash
python tests/test_reward.py        # Reward 三级评分（+1/0/-1）
python tests/test_answer_parser.py # 答案解析与 Parse Fail 检测
python tests/test_metrics.py       # 严格准确率计算（Parse Fail 按错误计入）
```

## 1. 数据处理

```bash
# 查看数据样例
cat examples/processed_examples.jsonl | head -3

# 处理自有数据（需提供词典和诗词文件）
python src/data/build_dataset.py \
    --dictionary_file your_dictionary.json \
    --poem_file your_poems.jsonl \
    --output_file output.jsonl

# 8:1:1 分层划分
python src/data/split_dataset.py
```

## 2. 训练

```bash
# 公开训练入口
bash configs/train_config_public.sh
```

关键配置：GRPO 算法、Qwen3-8B、batch_size=504、lr=1e-6、3 GPU、rollout.n=4。
完整参数见 `configs/config_summary.yaml`。

## 3. 评测

```bash
# 修改 MODEL_PATH 和 TEST_FILE_PATH 后运行
export MODEL_PATH="/path/to/model"
export TEST_FILE_PATH="/path/to/test.parquet"
python evaluation/evaluate_model_vllm.py
```

**评测要求**：Base 和 RFT 必须使用相同的测试集、Prompt 模板、解析规则和生成参数（temperature=0）。

## 4. 结果计算与 Bad Case 分析

```bash
# 计算指标
python evaluation/compute_metrics.py \
    --predictions predictions.parquet \
    --output results/my_metrics.csv

# 拆分 bad case（both / base_only / RFT_only）
python evaluation/classify_bad_cases.py \
    --base_preds base.parquet \
    --rft_preds rft.parquet \
    --output_dir results/

# 生成 HTML 报告
python evaluation/generate_bad_case_report.py \
    --audit_file audit.parquet \
    --output reports/bad_cases.html
```

## 完整流程总览

| 步骤 | 命令 | 依赖 |
|------|------|------|
| 单元测试 | `python tests/test_reward.py` | 无 |
| 数据样例 | `cat examples/processed_examples.jsonl` | 无 |
| 数据处理 | `python src/data/build_dataset.py` | 需自有数据 |
| 划分数据 | `python src/data/split_dataset.py` | 需处理后数据 |
| 训练 | `bash configs/train_config_public.sh` | 需完整数据 + Qwen3-8B + 3 GPU |
| 评测 Base | `python evaluation/evaluate_model_vllm.py` | 需 Base 模型 |
| 评测 RFT | 同上（指向 RFT checkpoint） | 需 RFT checkpoint |
| 计算指标 | `python evaluation/compute_metrics.py` | 需预测文件 |
| 拆分 bad case | `python evaluation/classify_bad_cases.py` | 需 Base + RFT 预测 |
| 生成报告 | `python evaluation/generate_bad_case_report.py` | 需审计数据 |

## 无法公开的内容

| 内容 | 原因 | 替代方案 |
|------|------|----------|
| 完整诗词数据库 | 版权限制 | 脱敏样例 + 处理代码 |
| 汉语大词典数据 | 版权限制 | 样例 + 处理代码 |
| 训练好的 RFT checkpoint | ~16GB 存储限制 | 完整训练脚本可参考 |
| 原始 8B 训练结果 | 完整数据未公开 | 指标文件和日志摘要可供核验 |
