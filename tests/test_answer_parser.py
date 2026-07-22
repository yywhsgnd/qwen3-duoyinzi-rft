"""测试答案解析器（与 evaluate_model_vllm.py 逻辑一致）"""
import re


def parse_model_output(generated_text: str) -> str:
    """与 evaluate_model_vllm.py 完全一致的解析逻辑"""
    match = re.search(r"####\s*(平声|仄声)", generated_text)
    if not match:
        return "PARSE_FAIL"
    answer = match.group(1)
    if answer == "平声":
        return "P"
    elif answer == "仄声":
        return "Z"
    else:
        return "PARSE_FAIL"


def test_parse_standard():
    assert parse_model_output("...\n#### 平声") == "P"
    assert parse_model_output("#### 仄声\n...") == "Z"


def test_parse_no_space():
    """正则 ####\\s* 允许无空格"""
    assert parse_model_output("####平声") == "P"
    assert parse_model_output("####仄声") == "Z"


def test_parse_multiline():
    assert parse_model_output("<think>...</think>\n\n#### 平声\n\n解析：") == "P"


def test_parse_fail():
    assert parse_model_output("答案是平声") == "PARSE_FAIL"
    assert parse_model_output("") == "PARSE_FAIL"
    assert parse_model_output("#### 平") == "PARSE_FAIL"
    assert parse_model_output("#### 平声平声") == "P"  # 只取第一个匹配


def test_parse_edge_cases():
    """边缘情况"""
    # 只有###没有####
    assert parse_model_output("### 平声") == "PARSE_FAIL"
    # 多个####只取第一个
    assert parse_model_output("#### 平声...#### 仄声") == "P"
    # 中文标点
    assert parse_model_output("#### 平声。") == "P"


if __name__ == "__main__":
    test_parse_standard()
    test_parse_no_space()
    test_parse_multiline()
    test_parse_fail()
    test_parse_edge_cases()
    print("所有答案解析器测试通过！")
