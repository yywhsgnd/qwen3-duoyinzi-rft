import json
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict  # ✅ 导入 defaultdict

# --- 配置日志记录 ---
# 配置日志记录器，用于替代print函数，提供更灵活的输出控制
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ==================== 函数1：基本不变 ====================
def load_dictionary_data(file_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    加载 pz_complied.json 字典数据。
    该函数会读取原始的JSON文件（预计为一个对象列表），
    并将其转换为一个以汉字为键、读音列表为值的字典，以便快速查询。

    :param file_path: pz_complied.json 文件的路径对象
    :return: 以汉字为键的字典
    """
    logging.info(f"正在加载新字典数据从: {file_path}")
    try:
        with file_path.open('r', encoding='utf-8') as f:
            data_list = json.load(f)
            # 将列表转换为字典以便 O(1) 复杂度的查询
            return {item['char']: item.get('readings', []) for item in data_list}
    except FileNotFoundError:
        logging.error(f"错误: 字典文件未找到于 '{file_path}'")
        raise
    except json.JSONDecodeError:
        logging.error(f"错误: 无法解析字典文件 '{file_path}' 的JSON格式")
        raise
    except (TypeError, KeyError) as e:
        logging.error(f"错误: 字典文件 '{file_path}' 格式不符合预期: {e}")
        raise

# ==================== 函数2：按 poem_id 组织数据 ====================
def load_poem_data(file_path: Path) -> Dict[Any, List[Dict[str, Any]]]:
    """
    加载所有诗歌数据 (JSON Lines格式)。
    ✅ 修改：按 poem_id 组织成字典。

    :param file_path: 诗歌JSONL文件的路径对象
    :return: ✅ 以 poem_id 为键，诗句列表为值的字典
    """
    logging.info(f"正在加载诗歌数据从: {file_path}")
    poems_by_id = defaultdict(list)
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    poem_line = json.loads(line.strip())
                    poem_id = poem_line.get('poem_id')
                    
                    if poem_id is None:
                        logging.warning(f"警告: 跳过一个没有 'poem_id' 的诗歌行: {line.strip()}")
                        continue
                        
                    poems_by_id[poem_id].append(poem_line)
                    
                except json.JSONDecodeError:
                    logging.warning(f"警告: 跳过一个无法解析的诗歌行: {line.strip()}")
        return poems_by_id
    except FileNotFoundError:
        logging.error(f"错误: 诗歌文件未找到于 '{file_path}'")
        raise

# ==================== 函数3：基本不变 ====================
def find_disambiguation_chars(original_pz: str, disamb_pz: str, content: str) -> List[Dict[str, Any]]:
    """
    从诗句中找出需要消歧的多音字及其正确读音。(此函数未改变)
    :param original_pz: 原始平仄标记字符串
    :param disamb_pz: 消歧后的平仄标记字符串
    :param content: 诗句内容
    :return: 一个包含多音字信息的字典列表
    """
    disamb_chars = []
    for i, (orig, disamb, char) in enumerate(zip(original_pz, disamb_pz, content)):
        if orig == 'A':
            if disamb.lower() == 'p':
                correct_pingze = "平声"
            elif disamb.lower() == 'z':
                correct_pingze = "仄声"
            else:
                continue
            disamb_chars.append({
                'position': i, 'character': char, 'correct_pingze': correct_pingze
            })
    return disamb_chars

# ==================== 函数4：基本不变 ====================
def organize_readings_by_pingze(readings_list: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    根据新的数据源(pz_complied.json)结构，整理字典释义，按平仄分组。
    它会直接使用 'pz' 字段 ('P' 或 'Z') 来判断平仄。

    :param readings_list: 从新字典中获取的某字的 "readings" 列表
    :return: 按平仄分组的释义字典
    """
    pingze_meanings: Dict[str, List[str]] = {"平声": [], "仄声": []}

    if not isinstance(readings_list, list):
        return pingze_meanings

    for reading in readings_list:
        pz_tag = reading.get("pz")
        definition = reading.get("definition")

        if not definition or not pz_tag:
            continue

        if pz_tag == "P":
            pingze_meanings["平声"].append(definition)
        elif pz_tag == "Z":
            pingze_meanings["仄声"].append(definition)

    return pingze_meanings

# ==================== 函数5：修改为使用全诗上下文 ====================
def create_rft_training_example(
    full_poem_context: str,  # ✅ 修改：接收全诗上下文
    disamb_char: Dict, 
    pingze_meanings: Dict
) -> Optional[Dict[str, Any]]:
    """
    创建精简版的、适用于RFT训练的样本。
    ✅ 修改：使用全诗上下文构建 Prompt。
    
    :param full_poem_context: ✅ 包含高亮目标句的全诗字符串
    :param disamb_char: 单个多音字的信息
    :param pingze_meanings: 该字按平仄分组的释义
    :return: 一个格式化的训练样本字典，或在无效情况下返回None
    """
    character = disamb_char['character']
    correct_pingze = disamb_char['correct_pingze']
    position = disamb_char['position'] + 1 # 目标字在 *目标句* 中的位置

    explanation_text = f"{character}字释义：\n"
    has_meanings = False
    for pingze, meanings in pingze_meanings.items():
        if meanings:
            has_meanings = True
            explanation_text += f"\n【{pingze}】\n"
            for i, definition in enumerate(meanings, 1):
                explanation_text += f"{i}. {definition}\n"

    if not has_meanings:
        return None

    system_content = "你是一个古诗平仄分析专家，请根据诗句上下文语义和词典释义，不要考虑整句诗的格律规则，只考虑语义，判断此处目标字的平仄。请使用####格式输出答案。"
    
    # ✅ 修改：使用全诗上下文构建 user_content
    user_content = f"""请分析以下全诗中，目标诗句（已用【】标出）里第 {position} 个字 '{character}' 的平仄。

全诗上下文：
{full_poem_context}

参考资料：
{explanation_text}
请判断此处"{character}"字应该读平声还是仄声？仅根据诗句上下文语义及参考资料，不要考虑整句诗的格律规则"""

    prompt_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    return {
        "prompt": prompt_messages,
        "reward_model": {"ground_truth": correct_pingze},
        "data_source": "duoyinzi"
    }

# ==================== 函数6：修改主循环 ====================
def process_data_for_rft(dictionary_file: Path, poem_file: Path, output_file: Path):
    """
    处理数据的主函数，生成精简版的RFT训练集。
    ✅ 修改：遍历“首”而不是“行”。
    """
    # 使用新的加载函数
    dictionary_data = load_dictionary_data(dictionary_file)
    # ✅ poem_data 现在是一个字典 {poem_id: [line1, line2, ...]}
    poem_data = load_poem_data(poem_file)

    logging.info(f"字典数据: {len(dictionary_data)} 字")
    logging.info(f"诗歌数据: {len(poem_data)} 首") # ✅ 修改

    training_examples = []
    processed_poems = 0 # ✅ 计数器改为“首”
    
    # ✅ 外层循环：遍历每一首诗
    for poem_id, full_poem_lines in poem_data.items():
        
        if processed_poems > 0 and processed_poems % 1000 == 0: # ✅ 修改进度报告
            logging.info(f"已处理 {processed_poems} 首诗，生成 {len(training_examples)} 个样本")

        # ✅ 内层循环：遍历该首诗的每一句
        for line_index, poem_line in enumerate(full_poem_lines):
            
            # ✅ 新增：构建带有高亮目标句的全诗上下文
            full_poem_context_parts = []
            for i, line in enumerate(full_poem_lines):
                content = line.get('content', '') # 安全获取
                if i == line_index:
                    # 高亮当前正在处理的诗句
                    full_poem_context_parts.append(f"【{content}】 <-- 目标诗句")
                else:
                    full_poem_context_parts.append(content)
            full_poem_context = "\n".join(full_poem_context_parts)

            # --- 以下逻辑与原来基本一致 ---
            
            disamb_chars = find_disambiguation_chars(
                poem_line.get('original_pz', ''),
                poem_line.get('disamb_pz', ''),
                poem_line.get('content', '')
            )

            for disamb_char in disamb_chars:
                character = disamb_char['character']
                
                if character in dictionary_data:
                    # 获取该字的读音和释义列表
                    readings = dictionary_data[character]
                    # 使用新的释义整理函数
                    pingze_meanings = organize_readings_by_pingze(readings)
                    
                    # 核心过滤逻辑：确保平声和仄声都有对应的释义
                    if pingze_meanings["平声"] and pingze_meanings["仄声"]:
                        
                        # ✅ 修改调用：传入全诗上下文
                        example = create_rft_training_example(
                            full_poem_context, # 替换原来的 poem_line
                            disamb_char, 
                            pingze_meanings
                        )
                        if example:
                            training_examples.append(example)
                            
        processed_poems += 1 # ✅ 增加已处理的诗歌计数

    logging.info(f"生成完成！共 {len(training_examples)} 个训练样本")

    if not training_examples:
        logging.warning("未生成任何训练样本。")
        return

    # --- 以下保存和统计逻辑保持不变 ---
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as f:
        for item in training_examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logging.info(f"数据已保存为: {output_file}")

    try:
        parquet_file = output_file.with_suffix('.parquet')
        df = pd.DataFrame(training_examples)
        df.to_parquet(parquet_file, index=False)
        logging.info(f"数据已保存为: {parquet_file}")
    except ImportError:
        logging.warning("警告: 未安装 'pyarrow' 库，无法保存为 Parquet 文件。请运行 'pip install pyarrow'。")

    pingze_stats: Dict[str, int] = {}
    for example in training_examples:
        pingze = example['reward_model']['ground_truth']
        pingze_stats[pingze] = pingze_stats.get(pingze, 0) + 1
    
    logging.info("=== 统计信息 ===")
    logging.info("平仄分布:")
    total = len(training_examples)
    for pingze, count in pingze_stats.items():
        percentage = (count / total) * 100
        logging.info(f"  {pingze}: {count} 个 ({percentage:.1f}%)")

    logging.info("=== 第一个样本示例 ===")
    print(json.dumps(training_examples[0], indent=2, ensure_ascii=False))


def main():
    """主函数，用于解析命令行参数并启动处理流程"""
    parser = argparse.ArgumentParser(
        description="从古诗数据中为多音字平仄消歧任务创建RFT训练集。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 更新了命令行参数以反映新的数据源
    parser.add_argument(
        "--dictionary_file",
        type=Path,
        default=Path("./data/pz_souyun_deepseek.json"),
        help="包含平仄和释义的字典JSON文件的路径"
    )
    parser.add_argument(
        "--poem_file",
        type=Path,
        default=Path("./data/pz9_yun9.jsonl"),
        help="包含平仄标记的诗歌JSONL文件的路径"
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("./rft_pz_deepsek_quanshi/duoyinzi_rft_dataset.jsonl"),
        help="输出的JSONL数据集文件的路径"
    )
    
    args = parser.parse_args()

    process_data_for_rft(
        dictionary_file=args.dictionary_file,
        poem_file=args.poem_file,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()