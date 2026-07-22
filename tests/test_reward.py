"""测试 duoyinzi_reward.py 的 Reward Function"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'reward'))

from duoyinzi_reward import extract_solution, compute_score


def test_extract_solution_strict_correct_format():
    """正确格式：#### 平声"""
    assert extract_solution("...思考...\n#### 平声", method="strict") == "平声"
    assert extract_solution("#### 仄声\n解析：...", method="strict") == "仄声"


def test_extract_solution_strict_malformed():
    """错误格式：无####或格式不对"""
    assert extract_solution("最终答案是平声", method="strict") is None
    assert extract_solution("####平声", method="strict") is None  # 要求空格
    assert extract_solution("", method="strict") is None


def test_extract_solution_flexible():
    """宽松模式：找最后一个平声/仄声"""
    assert extract_solution("平声...仄声", method="flexible") == "仄声"
    assert extract_solution("答案是平声", method="flexible") == "平声"


def test_compute_score_correct():
    """格式正确 + 答案正确 = +1"""
    assert compute_score("#### 平声", "平声") == 1
    assert compute_score("...\n#### 仄声\n解析", "仄声") == 1


def test_compute_score_wrong_answer():
    """格式正确 + 答案错误 = 0"""
    assert compute_score("#### 平声", "仄声") == 0
    assert compute_score("#### 仄声", "平声") == 0


def test_compute_score_format_error():
    """格式不可解析 = -1"""
    assert compute_score("答案是平声", "平声") == -1
    assert compute_score("", "仄声") == -1
    assert compute_score("####平声", "平声") == -1  # 缺少空格


def test_compute_score_long_output():
    """长输出：只检查尾部300字符"""
    long_prefix = "x" * 500
    assert compute_score(f"{long_prefix}\n#### 平声", "平声") == 1


def test_strict_vs_flexible():
    """strict要求'#### '后有空格，flexible只找最后一个关键词"""
    # strict 找不到（缺少空格）
    assert extract_solution("####仄声", method="strict") is None
    # flexible 能找到
    assert extract_solution("####仄声", method="flexible") == "仄声"


if __name__ == "__main__":
    test_extract_solution_strict_correct_format()
    test_extract_solution_strict_malformed()
    test_extract_solution_flexible()
    test_compute_score_correct()
    test_compute_score_wrong_answer()
    test_compute_score_format_error()
    test_compute_score_long_output()
    test_strict_vs_flexible()
    print("所有 Reward Function 测试通过！")
