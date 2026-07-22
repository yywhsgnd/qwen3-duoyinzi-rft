# duoyinzi.py
import re

# 定义截取长度
_SOLUTION_CLIP_CHARS = 300

def extract_solution(solution_str: str, method: str = "strict") -> str | None:
    """
    从模型的输出字符串中提取“平声”或“仄声”的答案。
    (严格要求 "#### " (带空格) 并且只取第一个匹配项 )
    """
    assert method in ["strict", "flexible"], "method 参数必须是 'strict' 或 'flexible'"

    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        match = re.search(r"#### (平声|仄声)", solution_str)
        
        if match:
            # 3. 如果找到了，返回它
            return match.group(1)
        
        # 4. 如果没找到 (比如模型输出了 ####\n仄声 或 ####仄声)，返回 None
        return None
       
    elif method == "flexible":
        matches = re.findall(r"(平声|仄声)", solution_str)
        if matches:
            return matches[-1]
        return None

def compute_score(solution_str: str, ground_truth: str) -> int:
    """
    评分规则:
      1: 格式正确且答案正确
      0: 格式正确但答案错误
     -1: 格式错误
    参数:
        solution_str: 模型生成的解决方案文本。
        ground_truth: 正确答案，应为 "平声" 或 "仄声"。

    返回:
        一个整数分数：1, 0, 或 -1。
    """
    # 1. 首先，用严格模式检查格式是否正确
    strict_answer = extract_solution(solution_str, method="strict")

    # 2. 如果格式不正确 (strict_answer 为 None)，直接判定为格式错误，得 -1 分
    if strict_answer is None:
        return -1  # 格式错误

    # 3. 程序能走到这里，说明格式是正确的。
    #    接下来只需要根据答案是否正确来给出 1 分或 0 分。
    if strict_answer == ground_truth:
        return 1  #  格式正确，答案也正确
    else:
        return 0  #  格式正确，但答案错误