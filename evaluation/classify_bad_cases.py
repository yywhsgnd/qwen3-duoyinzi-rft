#!/usr/bin/env python3
"""classify_bad_cases.py — 将错误样本按类型分类

产出:
    - both: Base和RFT均错误
    - base_only: 仅Base错误（RFT修正）
    - rft_only: 仅RFT错误（RFT退化）
    - parse_fail_base: Base的Parse Fail
    - parse_fail_rft: RFT的Parse Fail
"""
import argparse
import pandas as pd
import json
import sys
import os


def classify_errors(base_preds: list, rft_preds: list, ground_truths: list) -> dict:
    """
    分类错误样本。

    Returns:
        dict with keys: both, base_only, rft_only, base_parse_fail, rft_parse_fail,
                        base_total_errors, rft_total_errors, all_correct
    """
    both = []
    base_only = []
    rft_only = []
    base_parse_fail = []
    rft_parse_fail = []
    all_correct = []

    for i, (bp, rp, gt) in enumerate(zip(base_preds, rft_preds, ground_truths)):
        # 标准化标签
        gt_s = gt[0] if gt in ('平声', '仄声') else gt
        bp_s = bp[0] if bp in ('平声', '仄声', 'P', 'Z') else bp
        rp_s = rp[0] if rp in ('平声', '仄声', 'P', 'Z') else rp

        # 统一 P/Z
        bp_s = {'P': 'P', 'Z': 'Z', '平': 'P', '仄': 'Z'}.get(bp_s, bp_s)
        rp_s = {'P': 'P', 'Z': 'Z', '平': 'P', '仄': 'Z'}.get(rp_s, rp_s)
        gt_s = {'P': 'P', 'Z': 'Z', '平': 'P', '仄': 'Z'}.get(gt_s, gt_s)

        base_correct = (bp_s == gt_s) and bp != "PARSE_FAIL"
        rft_correct = (rp_s == gt_s) and rp != "PARSE_FAIL"
        base_pf = (bp == "PARSE_FAIL")
        rft_pf = (rp == "PARSE_FAIL")

        entry = {
            'idx': i,
            'ground_truth': gt,
            'base_pred': bp,
            'rft_pred': rp,
        }

        if base_pf:
            base_parse_fail.append(entry)
        if rft_pf:
            rft_parse_fail.append(entry)

        if not base_correct and not rft_correct:
            both.append(entry)
        elif not base_correct and rft_correct:
            base_only.append(entry)
        elif base_correct and not rft_correct:
            rft_only.append(entry)
        else:
            all_correct.append(entry)

    total_base_errors = len(both) + len(base_only) + len(base_parse_fail)
    total_rft_errors = len(both) + len(rft_only) + len(rft_parse_fail)

    return {
        'both': both,
        'base_only': base_only,
        'rft_only': rft_only,
        'base_parse_fail': base_parse_fail,
        'rft_parse_fail': rft_parse_fail,
        'all_correct': all_correct,
        'base_total_errors': total_base_errors,
        'rft_total_errors': total_rft_errors,
        'total': len(base_preds),
    }


def main():
    parser = argparse.ArgumentParser(description='错误样本分类')
    parser.add_argument('--base_preds', required=True, help='Base模型预测文件')
    parser.add_argument('--rft_preds', required=True, help='RFT模型预测文件')
    parser.add_argument('--output_dir', required=True, help='输出目录')

    args = parser.parse_args()

    # 加载预测
    base = pd.read_parquet(args.base_preds)
    rft = pd.read_parquet(args.rft_preds)

    # 提取 ground truth 和预测
    # 假定 parquet 包含 'ground_truth' 和 'prediction' 列
    gts = base['ground_truth'].tolist()
    bp = base['prediction'].tolist()
    rp = rft['prediction'].tolist()

    result = classify_errors(bp, rp, gts)

    # 输出统计
    print(f"Total: {result['total']}")
    print(f"Both wrong: {len(result['both'])}")
    print(f"Base only (RFT fixed): {len(result['base_only'])}")
    print(f"RFT only (RFT regression): {len(result['rft_only'])}")
    print(f"Base Parse Fail: {len(result['base_parse_fail'])}")
    print(f"RFT Parse Fail: {len(result['rft_parse_fail'])}")
    print(f"All correct: {len(result['all_correct'])}")

    # 保存分类结果
    os.makedirs(args.output_dir, exist_ok=True)
    for key in ['both', 'base_only', 'rft_only', 'base_parse_fail', 'rft_parse_fail']:
        with open(f"{args.output_dir}/{key}.jsonl", 'w') as f:
            for item in result[key]:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved: {args.output_dir}/{key}.jsonl ({len(result[key])} entries)")

    # 保存统计
    summary = {
        'total': result['total'],
        'both_wrong': len(result['both']),
        'base_only': len(result['base_only']),
        'rft_only': len(result['rft_only']),
        'base_parse_fail': len(result['base_parse_fail']),
        'rft_parse_fail': len(result['rft_parse_fail']),
        'all_correct': len(result['all_correct']),
        'base_total_errors': result['base_total_errors'],
        'rft_total_errors': result['rft_total_errors'],
    }
    with open(f"{args.output_dir}/classification_summary.json", 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
