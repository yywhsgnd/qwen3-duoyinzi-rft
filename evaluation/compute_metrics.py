#!/usr/bin/env python3
"""compute_metrics.py — 计算严格端到端准确率及其它指标

从模型预测结果文件或原始测试集+预测列表计算指标。
支持原始标签和审计标签的切换。

用法:
    python compute_metrics.py --predictions preds.parquet --output results/raw_metrics.csv
    python compute_metrics.py --predictions preds.parquet --audited_labels corrected_labels.jsonl --output results/audited_metrics.csv
"""
import argparse
import pandas as pd
import json
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.test_metrics import compute_strict_accuracy


LABEL_MAP = {'平声': 'P', '仄声': 'Z', 'P': 'P', 'Z': 'Z'}


def parse_prediction(raw_output: str) -> str:
    """解析模型原始输出为 P/Z/PARSE_FAIL"""
    import re
    if raw_output is None or str(raw_output).strip() == "":
        return "PARSE_FAIL"
    text = str(raw_output)
    match = re.search(r"####\s*(平声|仄声)", text)
    if match:
        return LABEL_MAP.get(match.group(1), "PARSE_FAIL")
    # 兜底：找最后一个平声/仄声
    ping_pos = text.rfind("平声")
    ze_pos = text.rfind("仄声")
    if ping_pos > ze_pos:
        return "P"
    elif ze_pos > ping_pos:
        return "Z"
    return "PARSE_FAIL"


def load_audited_labels(audited_file: str) -> dict:
    """加载审计标签映射: sample_id -> audited_label"""
    mapping = {}
    with open(audited_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            mapping[item['sample_id']] = item['audited_label']
    return mapping


def compute_from_predictions_df(df: pd.DataFrame, audited_labels: dict = None,
                                  test_set_name: str = "unknown") -> dict:
    """
    从包含预测结果的DataFrame计算指标。

    df必须包含: 'ground_truth' 和 'prediction' 列
    """
    gts = []
    preds = []

    for _, row in df.iterrows():
        gt = row['ground_truth']
        pred = row['prediction']

        # 如果提供了审计标签，替换 ground truth
        if audited_labels:
            sample_id = row.get('sample_id', '')
            if sample_id in audited_labels:
                gt = audited_labels[sample_id]

        gts.append(gt)
        preds.append(pred)

    metrics = compute_strict_accuracy(preds, gts)
    metrics['test_set'] = test_set_name
    metrics['label_type'] = 'audited' if audited_labels else 'original'
    return metrics


def compute_from_test_set(test_parquet: str, predictions: list,
                           audited_labels: dict = None,
                           test_set_name: str = "unknown") -> dict:
    """
    从测试集Parquet和预测列表计算指标。
    """
    test = pd.read_parquet(test_parquet)
    gts = []
    for _, row in test.iterrows():
        gt = row['reward_model'].get('ground_truth', '')
        gts.append(gt)

    if len(predictions) != len(gts):
        print(f"WARNING: predictions ({len(predictions)}) != gts ({len(gts)})")

    n = min(len(predictions), len(gts))
    preds = [LABEL_MAP.get(p, p) for p in predictions[:n]]
    gts_short = [LABEL_MAP.get(g, g) for g in gts[:n]]

    # Apply audited labels
    if audited_labels:
        for i in range(n):
            sample_id = f"{test_set_name}_{i}"
            if sample_id in audited_labels:
                gts_short[i] = LABEL_MAP.get(audited_labels[sample_id], audited_labels[sample_id])

    metrics = compute_strict_accuracy(preds, gts_short)
    metrics['test_set'] = test_set_name
    metrics['label_type'] = 'audited' if audited_labels else 'original'
    return metrics


def main():
    parser = argparse.ArgumentParser(description='计算严格端到端准确率')
    parser.add_argument('--predictions', help='包含预测结果的 parquet 文件')
    parser.add_argument('--test_parquet', help='测试集 parquet 文件')
    parser.add_argument('--pred_list', help='预测列表文件（每行一个预测）')
    parser.add_argument('--audited_labels', help='审计标签 JSONL 文件')
    parser.add_argument('--output', required=True, help='输出 CSV 文件路径')
    parser.add_argument('--test_set_name', default='unknown', help='测试集名称')
    parser.add_argument('--append', action='store_true', help='追加模式（不覆盖已有文件）')

    args = parser.parse_args()

    # 加载审计标签
    audited = None
    if args.audited_labels:
        audited = load_audited_labels(args.audited_labels)
        print(f"Loaded {len(audited)} audited labels")

    # 计算指标
    if args.predictions:
        df = pd.read_parquet(args.predictions)
        metrics = compute_from_predictions_df(df, audited, args.test_set_name)
    elif args.test_parquet and args.pred_list:
        with open(args.pred_list) as f:
            preds = [line.strip() for line in f if line.strip()]
        metrics = compute_from_test_set(args.test_parquet, preds, audited, args.test_set_name)
    else:
        print("ERROR: need --predictions or (--test_parquet + --pred_list)")
        sys.exit(1)

    # 输出
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    df_out = pd.DataFrame([metrics])
    if args.append and os.path.exists(args.output):
        existing = pd.read_csv(args.output)
        df_out = pd.concat([existing, df_out], ignore_index=True)

    df_out.to_csv(args.output, index=False)
    print(f"Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
