"""测试指标计算逻辑"""
import json
import sys
import os


def compute_strict_accuracy(predictions: list[str], ground_truths: list[str]) -> dict:
    """
    计算严格端到端准确率。
    Parse Fail 按错误计入。

    Args:
        predictions: 预测标签列表，'P', 'Z', 或 'PARSE_FAIL'
        ground_truths: 真实标签列表，'P' 或 'Z' (或 '平声'/'仄声')

    Returns:
        包含各种指标的字典
    """
    total = len(predictions)
    correct = 0
    parse_fail = 0
    p_correct = 0
    p_total = 0
    z_correct = 0
    z_total = 0

    # 统一标签格式：平声/P -> 'P', 仄声/Z -> 'Z'
    LABEL_MAP = {'平声': 'P', '仄声': 'Z', 'P': 'P', 'Z': 'Z'}

    for pred, gt in zip(predictions, ground_truths):
        gt_short = LABEL_MAP.get(gt, gt)
        pred_short = LABEL_MAP.get(pred, pred)

        if pred == "PARSE_FAIL":
            parse_fail += 1
        elif pred_short == gt_short:
            correct += 1

        if gt_short == 'P':
            p_total += 1
            if pred_short == 'P':
                p_correct += 1
        elif gt_short == 'Z':
            z_total += 1
            if pred_short == 'Z':
                z_correct += 1

    strict_acc = correct / total if total > 0 else 0.0
    p_acc = p_correct / p_total if p_total > 0 else 0.0
    z_acc = z_correct / z_total if z_total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "strict_accuracy": round(strict_acc, 4),
        "parse_fail": parse_fail,
        "parse_fail_rate": round(parse_fail / total, 4),
        "P_accuracy": round(p_acc, 4),
        "Z_accuracy": round(z_acc, 4),
        "P_count": p_total,
        "Z_count": z_total,
    }


def test_all_correct():
    preds = ['P', 'Z', 'P', 'Z']
    gts = ['平声', '仄声', '平声', '仄声']
    metrics = compute_strict_accuracy(preds, gts)
    assert metrics['strict_accuracy'] == 1.0
    assert metrics['parse_fail'] == 0


def test_parse_fail_penalized():
    """Parse Fail 按错误计入严格准确率"""
    preds = ['P', 'PARSE_FAIL', 'P', 'P']
    gts = ['平声', '平声', '平声', '平声']
    metrics = compute_strict_accuracy(preds, gts)
    assert metrics['correct'] == 3  # 3个正确
    assert metrics['strict_accuracy'] == 0.75  # 3/4 = 0.75
    assert metrics['parse_fail'] == 1


def test_all_parse_fail():
    preds = ['PARSE_FAIL'] * 10
    gts = ['平声'] * 10
    metrics = compute_strict_accuracy(preds, gts)
    assert metrics['strict_accuracy'] == 0.0
    assert metrics['parse_fail'] == 10


def test_label_format_unification():
    """测试平声/仄声 与 P/Z 的统一"""
    # 预测用 P/Z
    preds = ['P', 'Z']
    # 真实标签用 平声/仄声
    gts = ['平声', '仄声']
    metrics = compute_strict_accuracy(preds, gts)
    assert metrics['strict_accuracy'] == 1.0


def test_switching_labels():
    """验证原始标签和审计标签可以正确切换"""
    original_preds = ['P', 'P', 'Z', 'Z', 'PARSE_FAIL']
    original_gts =   ['P', 'Z', 'Z', 'P', 'P']

    # 原始标签结果
    orig = compute_strict_accuracy(original_preds, original_gts)
    assert orig['correct'] == 2  # P,P vs P,Z (mismatch), Z,Z vs Z,P (mismatch)
    assert orig['strict_accuracy'] == 0.4

    # 审计后标签：假设第2个和第4个样本的标签被修正
    audited_gts =    ['P', 'P', 'Z', 'Z', 'P']
    audited = compute_strict_accuracy(original_preds, audited_gts)
    assert audited['correct'] == 4  # 现在只有PARSE_FAIL错了
    assert audited['strict_accuracy'] == 0.8


if __name__ == "__main__":
    test_all_correct()
    test_parse_fail_penalized()
    test_all_parse_fail()
    test_label_format_unification()
    test_switching_labels()
    print("所有指标计算测试通过！")
