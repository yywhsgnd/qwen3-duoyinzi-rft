# process_fixed_final_pingshui_complete.py
import json
import pandas as pd
import re

def load_hydcd_data(file_path):
    """加载汉语大词典数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_poem_data(file_path):
    """加载所有诗歌数据"""
    poems = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            poems.append(json.loads(line.strip()))
    return poems

def find_disambiguation_chars(original_pz, disamb_pz, content):
    disamb_chars = []
    for i, (orig, disamb, char) in enumerate(zip(original_pz, disamb_pz, content)):
        # 修正：直接判断大小写，避免转换错误
        if orig == 'A':
            if disamb in ['P', 'p']:
                correct_pingze = "平声"
            elif disamb in ['Z', 'z']:
                correct_pingze = "仄声"
            else:
                continue  # 跳过无效标记
                
            disamb_chars.append({
                'position': i,
                'character': char,
                'correct_pingze': correct_pingze
            })
    return disamb_chars

def organize_meanings_complete(character, meanings_list):
    """完整版释义整理 - 显示所有释义"""
    pingze_meanings = {"平声": [], "仄声": []}
    
    if not isinstance(meanings_list, list):
        return pingze_meanings
    
    for meaning in meanings_list:
        # 获取平仄标签
        pingze_tag = meaning.get("平仄", "")
        definition = meaning.get("定义", "")
        pinyin = meaning.get("拼音", "")
        
        if not definition:
            continue
            
        # 转换平仄标签
        if pingze_tag == "平":
            pingze = "平声"
        elif pingze_tag == "仄":
            pingze = "仄声"
        else:
            continue  # 跳过没有平仄标签的
        
        # 完整释义条目
        meaning_entry = {
            "pinyin": pinyin,
            "definition": definition,
            "word_class": meaning.get("词性", ""),
            "other": meaning.get("其他", [])
        }
        
        pingze_meanings[pingze].append(meaning_entry)
    
    return pingze_meanings

def generate_thought_process(poem_line, character, correct_pingze, pingze_meanings):
    """生成思考过程"""
    # 简短的思考过程模板
    thought_templates = [
        f"分析诗句'{poem_line['content']}'，字'{character}'在句中表示的意思与{correct_pingze}释义相符。",
        f"根据诗句语境，'{character}'字在此处应取{correct_pingze}的释义。",
        f"从诗句上下文判断，'{character}'字读{correct_pingze}更符合意境。"
    ]
    
    # 随机选择一个模板（这里简单取第一个）
    return thought_templates[0]

def create_training_example_complete(poem_line, disamb_char, pingze_meanings, index):
    """完整释义的训练样本创建 - 适配Qwen3格式"""
    
    character = disamb_char['character']
    correct_pingze = disamb_char['correct_pingze']
    
    # 生成完整的释义文本 - 显示所有释义
    explanation_text = f"{character}字释义：\n"
    
    total_meanings = 0
    for pingze, meanings in pingze_meanings.items():
        if meanings:
            explanation_text += f"\n【{pingze}】\n"
            for i, meaning in enumerate(meanings, 1):  # 移除数量限制，显示所有释义
                word_class = f"({meaning['word_class']})" if meaning['word_class'] and meaning['word_class'] != "空" else ""
                other_info = f" [{','.join(meaning['other'])}]" if meaning['other'] else ""
                explanation_text += f"{i}. {meaning['definition']}{word_class}{other_info}\n"
                total_meanings += 1
    
    # 生成思考过程
    thought_process = generate_thought_process(poem_line, character, correct_pingze, pingze_meanings)
    
    # Qwen3格式的系统提示
    system_content = """你是一个古诗平仄分析专家。请根据上下文语义判断目标字的平仄，按照以下格式输出：

<think>
【简要分析诗句意境和字义】
</think>
#### 平声 或 #### 仄声

要求：
1. <think>标签内的分析诗句意境和字义
2. 必须在</think>后换行输出最终答案
3. 答案格式必须严格遵循：#### 平声 或 #### 仄声
4. 请优先基于诗句的语境和语义进行判断"""

    # 用户提示
    user_content = f"""请分析以下诗句中指定汉字的平仄：

诗句：{poem_line['content']}
目标字：{character}

参考资料：{explanation_text}
请根据诗句上下文语义及参考资料判断此处"{character}"字应该读平声还是仄声？"""

    # 助手输出 - Qwen3格式
    assistant_content = f"<think>{thought_process}</think>\n#### {correct_pingze}"
    
    # 构建prompt（用于推理时输入）
    prompt_messages = [
        {"content": system_content, "role": "system"},
        {"content": user_content, "role": "user"}
    ]
    
    # 构建完整对话（用于训练）
    messages = prompt_messages + [
        {"content": assistant_content, "role": "assistant"}
    ]
    
    # 构建数据
    data = {
        "messages": messages,
        "prompt": prompt_messages,
        "category": "duoyinzi_qwen",
        "data_source": "duoyinzi_qwen",
        "ability": "pingze_classification",
        "reward_model": {
            "ground_truth": correct_pingze,
            "style": "rule"
        },
        "extra_info": {
            "answer": correct_pingze,
            "character": character,
            "poem": poem_line['content'],
            "category": "duoyinzi_qwen",
            "index": index,
            "question": user_content,
            "title": f"{character}字平仄判断",
            "thought_process": thought_process,
            "total_meanings": total_meanings,  # 记录释义总数
            "ping_meanings_count": len(pingze_meanings["平声"]),
            "ze_meanings_count": len(pingze_meanings["仄声"])
        }
    }
    
    return data

def process_data_complete(hydcd_file, poem_file, output_file):
    """完整版数据处理 - 显示所有释义"""
    
    print("加载数据...")
    hydcd_data = load_hydcd_data(hydcd_file)
    poem_data = load_poem_data(poem_file)
    
    print(f"词典数据: {len(hydcd_data)} 字")
    print(f"诗歌数据: {len(poem_data)} 行")
    
    training_examples = []
    char_stats = {}
    meaning_stats = []  # 统计释义数量
    
    for i, poem_line in enumerate(poem_data):
        if i % 50000 == 0 and i > 0:
            print(f"已处理 {i} 行，生成 {len(training_examples)} 个样本")
        
        # 找出多音字
        disamb_chars = find_disambiguation_chars(
            poem_line['original_pz'],
            poem_line['disamb_pz'], 
            poem_line['content']
        )
        
        for disamb_char in disamb_chars:
            character = disamb_char['character']
            
            if character in hydcd_data:
                meanings = hydcd_data[character]
                # 使用完整版释义整理函数 - 显示所有释义
                pingze_meanings = organize_meanings_complete(character, meanings)
                
                # 只有两个平仄都有释义时才创建样本
                if pingze_meanings["平声"] and pingze_meanings["仄声"]:
                    example = create_training_example_complete(
                        poem_line, disamb_char, pingze_meanings, len(training_examples)
                    )
                    training_examples.append(example)
                    
                    # 统计释义数量
                    total_ping = len(pingze_meanings["平声"])
                    total_ze = len(pingze_meanings["仄声"])
                    meaning_stats.append({
                        'character': character,
                        'ping_meanings': total_ping,
                        'ze_meanings': total_ze,
                        'total_meanings': total_ping + total_ze
                    })
                    
                    char_stats[character] = char_stats.get(character, 0) + 1
    
    print(f"\n生成完成！共 {len(training_examples)} 个训练样本")
    
    # 保存数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存为parquet
    parquet_file = output_file.replace('.jsonl', '.parquet')
    df = pd.DataFrame(training_examples)
    df.to_parquet(parquet_file, index=False)
    
    print(f"保存为: {output_file}")
    print(f"保存为: {parquet_file}")
    
    # 显示详细统计
    pingze_stats = {}
    for example in training_examples:
        pingze = example['reward_model']['ground_truth']
        pingze_stats[pingze] = pingze_stats.get(pingze, 0) + 1
    
    print("\n=== 统计信息 ===")
    print(f"平仄分布:")
    for pingze, count in pingze_stats.items():
        print(f"  {pingze}: {count} 个 ({count/len(training_examples)*100:.1f}%)")
    
    # 释义统计
    if meaning_stats:
        avg_ping = sum([m['ping_meanings'] for m in meaning_stats]) / len(meaning_stats)
        avg_ze = sum([m['ze_meanings'] for m in meaning_stats]) / len(meaning_stats)
        avg_total = sum([m['total_meanings'] for m in meaning_stats]) / len(meaning_stats)
        
        print(f"\n释义统计:")
        print(f"  平均平声释义: {avg_ping:.1f} 个")
        print(f"  平均仄声释义: {avg_ze:.1f} 个") 
        print(f"  总平均释义: {avg_total:.1f} 个")
        
        # 找出释义最多的字
        max_meanings_char = max(meaning_stats, key=lambda x: x['total_meanings'])
        print(f"  释义最多的字: '{max_meanings_char['character']}' ({max_meanings_char['total_meanings']} 个释义)")
    
    print(f"\n涉及的多音字 ({len(char_stats)} 个):")
    # 按出现次数排序
    sorted_chars = sorted(char_stats.items(), key=lambda x: x[1], reverse=True)
    for char, count in sorted_chars[:20]:
        print(f"  {char}: {count} 次")
    
    # 显示样本示例
    if training_examples:
        print(f"\n=== 前3个样本示例 ===")
        for i, example in enumerate(training_examples[:3]):
            print(f"\n示例 {i+1}:")
            print(f"诗句: {example['extra_info']['poem']}")
            print(f"目标字: {example['extra_info']['character']}")
            print(f"正确答案: {example['reward_model']['ground_truth']}")
            assistant_content = example['messages'][2]['content']
            print(f"助手输出: {assistant_content}")
            print(f"思考过程: {example['extra_info']['thought_process']}")
            user_content = example['messages'][1]['content']
            print(f"用户输入长度: {len(user_content)} 字符")
            print(f"平声释义数: {example['extra_info']['ping_meanings_count']}")
            print(f"仄声释义数: {example['extra_info']['ze_meanings_count']}")
            print(f"用户输入预览: {user_content[:200]}...")
    
    return training_examples

def validate_qwen_dataset(file_path):
    """验证Qwen3格式数据集"""
    import pandas as pd
    
    df = pd.read_parquet(file_path)
    print(f"数据集大小: {len(df)}")
    
    # 检查格式
    correct_format_count = 0
    total_length = 0
    prompt_lengths = []
    
    for i, row in df.iterrows():
        assistant_content = None
        user_content = None
        for msg in row['messages']:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
            elif msg['role'] == 'user':
                user_content = msg['content']
        
        if assistant_content and user_content:
            total_length += len(assistant_content)
            prompt_lengths.append(len(user_content))
            
            # 检查是否包含<think>和####格式
            has_think = '<think>' in assistant_content and '</think>' in assistant_content
            has_correct_answer = '#### 平声' in assistant_content or '#### 仄声' in assistant_content
            
            if has_think and has_correct_answer:
                correct_format_count += 1
            else:
                if i < 3:  # 显示前3个格式错误的样本
                    print(f"格式错误样本 {i}: {assistant_content}")
    
    print(f"\n格式正确率: {correct_format_count}/{len(df)} ({correct_format_count/len(df)*100:.1f}%)")
    print(f"平均响应长度: {total_length/len(df):.1f} 字符")
    print(f"平均提示词长度: {sum(prompt_lengths)/len(prompt_lengths):.1f} 字符")
    print(f"最大提示词长度: {max(prompt_lengths)} 字符")
    print(f"最小提示词长度: {min(prompt_lengths)} 字符")
    
    return correct_format_count == len(df)

if __name__ == "__main__":
    # 处理所有数据，使用完整释义的Qwen3格式
    dataset = process_data_complete(
        "汉语大.json", 
        "pz9_yun9.jsonl", 
        "duoyinzi_pingshui_complete.jsonl"
    )
    
    # 验证生成的数据
    print("\n=== 数据验证 ===")
    validate_qwen_dataset("duoyinzi_pingshui_complete.parquet")