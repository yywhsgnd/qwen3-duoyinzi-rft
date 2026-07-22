#!/usr/bin/env python3
"""generate_bad_case_report.py — 生成代表性 Bad Case 轻量 HTML 报告

从审计结果中挑选每种类型的代表性样本，生成精简 HTML。
每种类型选取 3-5 条有代表性的真实样本。

类型:
1. 原始GT高度疑似错误（DS三次一致但与原GT冲突）
2. Base错误、RFT修正
3. Base正确、RFT退化
4. Base和RFT均真正判断错误
5. Base Parse Fail、RFT成功解析
6. DeepSeek三次不一致（需人工判断）
7. 高频问题字（当、相、行、看等）

输出：单个轻量 HTML 文件，约 100-500KB。
"""
import argparse
import pandas as pd
import json
import os
import sys
from collections import Counter


def pick_representative_samples(audit_df, category, n=5):
    """从审计结果中选取代表性样本"""
    if category == 'high_confidence_label_conflict':
        subset = audit_df[audit_df['audit_status'] == 'high_confidence_label_conflict']
        return subset.head(n)

    elif category == 'needs_manual_review':
        subset = audit_df[audit_df['audit_status'] == 'needs_manual_review']
        return subset.head(n)

    elif category == 'high_freq_chars':
        # Extract target character from prompt
        chars = []
        for _, row in audit_df.iterrows():
            prompt = row.get('prompt', [])
            if hasattr(prompt, 'tolist'):
                prompt = prompt.tolist()
            for msg in prompt:
                if msg.get('role') == 'user':
                    content = msg['content']
                    import re
                    char_match = re.search(r"'\s*(\S)\s*'", content)
                    if char_match:
                        chars.append(char_match.group(1))
                    break

        char_counts = Counter(chars)
        top_chars = [c for c, _ in char_counts.most_common(10)]

        # Pick samples for top characters
        picks = []
        for char in top_chars:
            for _, row in audit_df.iterrows():
                prompt = row.get('prompt', [])
                if hasattr(prompt, 'tolist'):
                    prompt = prompt.tolist()
                for msg in prompt:
                    if msg.get('role') == 'user':
                        if f"'{char}'" in msg['content']:
                            picks.append(row)
                            break
                if len(picks) >= n:
                    break
            if len(picks) >= n:
                break
        return pd.DataFrame(picks[:n])

    return pd.DataFrame()


def format_sample_html(row, idx, label=""):
    """格式化单个样本为 HTML 片段"""
    prompt = row.get('prompt', [])
    if hasattr(prompt, 'tolist'):
        prompt = list(prompt)

    system_content = ""
    user_content = ""
    for msg in prompt:
        if msg.get('role') == 'system':
            system_content = msg['content']
        elif msg.get('role') == 'user':
            user_content = msg['content']

    # 提取关键信息
    import re
    char_match = re.search(r"'\s*(\S)\s*'", user_content)
    target_char = char_match.group(1) if char_match else "?"
    poem_match = re.search(r'【(.+?)】', user_content)
    target_sentence = poem_match.group(1) if poem_match else "?"

    original_label = row.get('original_label', '?')

    ds_preds = []
    for run in [1, 2, 3]:
        col = f'ds_run{run}_pred'
        if col in row:
            ds_preds.append(str(row[col]))
    ds_consistent = row.get('ds_consistent', False)
    ds_majority = row.get('ds_majority_pred', '?')

    audit_status = row.get('audit_status', '?')

    status_badge = {
        'high_confidence_label_conflict':
            '<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px">高置信标签冲突</span>',
        'needs_manual_review':
            '<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:4px">需人工复核</span>',
        'consistent_with_original':
            '<span style="background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:4px">与原标签一致</span>',
    }.get(audit_status, audit_status)

    html = f'''
    <div style="border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin:12px 0;background:#fff">
        <div style="margin-bottom:8px">
            #{idx} {status_badge} {label}
        </div>
        <table style="font-size:13px;margin-bottom:8px">
            <tr><td style="color:#64748b;width:80px">目标字</td><td><b>{target_char}</b></td></tr>
            <tr><td style="color:#64748b">目标句</td><td>{target_sentence}</td></tr>
            <tr><td style="color:#64748b">原始标签</td><td>{original_label}</td></tr>
            <tr><td style="color:#64748b">DS Run1/2/3</td>
                <td>{' / '.join(ds_preds)}</td></tr>
            <tr><td style="color:#64748b">DS一致</td><td>{'是' if ds_consistent else '否（需人工）'}</td></tr>
            <tr><td style="color:#64748b">DS多数</td><td>{ds_majority}</td></tr>
        </table>
        <details>
            <summary style="cursor:pointer;color:#3b82f6;font-size:13px">
                📋 完整 Prompt
            </summary>
            <pre style="white-space:pre-wrap;font-size:12px;background:#f8fafc;padding:8px;
                       border-radius:4px;max-height:300px;overflow-y:auto;margin-top:8px">
【System】
{system_content}

【User】
{user_content[:1000]}{'...' if len(user_content) > 1000 else ''}
            </pre>
        </details>
    </div>'''
    return html


def main():
    parser = argparse.ArgumentParser(description='生成轻量 Bad Case HTML 报告')
    parser.add_argument('--audit_file', required=True,
                        help='DeepSeek 审计结果 parquet 文件')
    parser.add_argument('--output', required=True,
                        help='输出 HTML 文件路径')
    parser.add_argument('--title', default='Bad Case 分析报告',
                        help='报告标题')
    parser.add_argument('--max_per_category', type=int, default=3,
                        help='每种类型最多展示样本数')
    args = parser.parse_args()

    df = pd.read_parquet(args.audit_file)
    n = args.max_per_category

    sections = []

    # 类型 1: 高置信标签冲突
    conflicts = pick_representative_samples(df, 'high_confidence_label_conflict', n)
    if len(conflicts) > 0:
        items = ''
        for i, (_, row) in enumerate(conflicts.iterrows()):
            items += format_sample_html(row, i+1, '高置信标签冲突')
        sections.append(f'''
        <h2>1. 原始标签高度疑似错误（DS三次一致但与原标签冲突）</h2>
        <p>共 {len(df[df["audit_status"]=="high_confidence_label_conflict"])} 条，展示 {len(conflicts)} 条：</p>
        {items}
        ''')

    # 类型 2: 需人工复核（DS三次不一致）
    manual = pick_representative_samples(df, 'needs_manual_review', n)
    if len(manual) > 0:
        items = ''
        for i, (_, row) in enumerate(manual.iterrows()):
            items += format_sample_html(row, i+1, '需人工复核')
        total_manual = len(df[df["audit_status"]=="needs_manual_review"])
        sections.append(f'''
        <h2>2. DeepSeek 三次不一致（需人工判断）</h2>
        <p>共 {total_manual} 条，展示 {len(manual)} 条：</p>
        {items}
        ''')

    # 类型 3: 高频问题字
    high_freq = pick_representative_samples(df, 'high_freq_chars', n)
    if len(high_freq) > 0:
        items = ''
        for i, (_, row) in enumerate(high_freq.iterrows()):
            items += format_sample_html(row, i+1, '高频字')

        # 统计高频字
        all_status = df['audit_status'].value_counts().to_dict()
        sections.append(f'''
        <h2>3. 审计统计</h2>
        <table style="border-collapse:collapse;font-size:13px">
            <tr><th style="text-align:left;padding:4px 12px;border-bottom:1px solid #e2e8f0">状态</th>
                <th style="text-align:right;padding:4px 12px;border-bottom:1px solid #e2e8f0">数量</th></tr>
        ''' + ''.join(
            f'<tr><td style="padding:2px 12px">{k}</td>'
            f'<td style="text-align:right;padding:2px 12px">{v}</td></tr>'
            for k, v in all_status.items()
        ) + '</table>')

    # 组装 HTML
    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>{args.title}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI",sans-serif;background:#f1f5f9;color:#1e293b;
      padding:24px;max-width:1000px;margin:0 auto}}
h1{{font-size:22px;margin-bottom:4px}}
h2{{font-size:17px;margin:20px 0 12px;padding-bottom:6px;border-bottom:2px solid #3b82f6}}
.subtitle{{color:#64748b;font-size:14px;margin-bottom:20px}}
p{{margin:8px 0;font-size:14px}}
details{{margin:4px 0}}
summary{{user-select:none}}
summary:hover{{color:#2563eb}}
pre{{line-height:1.5}}
</style>
</head>
<body>
<h1>{args.title}</h1>
<div class="subtitle">共 {len(df)} 条审计样本</div>
{''.join(sections)}
</body>
</html>'''

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(html)
    print(f"HTML 报告已保存: {args.output} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
