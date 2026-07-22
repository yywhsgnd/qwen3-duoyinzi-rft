import os
import time
import torch
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams 
from sklearn.metrics import classification_report
from tqdm import tqdm  
import re
from sklearn.metrics import accuracy_score
# --- 0. 配置 ---

# GPU 配置
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# 测试模式：设为 True 只跑 10 个样本，设为 False 跑全部
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 10

# 模型和测试集路径（从环境变量读取，提供默认值）
MODEL_PATH = os.environ.get("MODEL_PATH", "./Qwen3-8B")
TEST_FILE_PATH = os.environ.get("TEST_FILE_PATH", "./test.parquet")

# 定义标签映射 (用于匹配 sklearn 报告)
LABEL_MAP = {
    "平声": "P",
    "仄声": "Z"
}
# 反向映射，用于案例分析
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def parse_model_output(generated_text: str) -> str:
    """
    用于查找 '####' 标记并提取答案。
    (与训练脚本 duoyinzi.py 一致的严格逻辑 )
    """
    
    #match = re.search(r"#### (平声|仄声)", generated_text)
    match = re.search(r"####\s*(平声|仄声)", generated_text)
    
    if not match:
        return "PARSE_FAIL"
        
    # 找到了，提取答案
    answer = match.group(1)
    
    if answer == "平声":
        return "P"
    elif answer == "仄声":
        return "Z"
    else:
        return "PARSE_FAIL"


# --- 1. 加载模型和分词器 ---
print(f"正在从 {MODEL_PATH} 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(f"正在从 {MODEL_PATH} 加载 VLLM 引擎...")
# 使用 VLLM 加载模型，明确指定 90% 显存占用
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    gpu_memory_utilization=0.9 
)
print("✅ 模型加载成功！")

# --- 2. 加载测试数据 ---
print(f"正在加载测试集: {TEST_FILE_PATH} ...")
try:
    df = pd.read_parquet(TEST_FILE_PATH)
except Exception as e:
    print(f" 加载 Parquet 文件失败: {e}")
    exit()

if DEBUG_MODE:
    print(f"运行在 [调试模式]，只测试 {DEBUG_SAMPLE_SIZE} 个样本。")
    df = df.head(DEBUG_SAMPLE_SIZE)

print(f"数据加载成功，共 {len(df)} 条样本。")

# --- 3. 准备所有 Prompts 和真实标签 ---
print("--- 正在准备所有 Prompts ---")

prompts_list = [] # 存放所有要推理的 prompts
y_true = []       # 存放所有真实标签 (P/Z)
raw_prompts_for_case_study = [] # 存放原始 prompt 结构

for index, row in tqdm(df.iterrows(), total=len(df), desc="准备数据"):
    try:
        # 1. 提取 Prompt 和 真实标签
        messages = row['prompt']
        ground_truth_text = row['reward_model']['ground_truth']
        
        # 2. 转换标签
        ground_truth_label = LABEL_MAP.get(ground_truth_text)
        if not ground_truth_label:
            continue
            
        # 3. 准备 Prompt 字符串
        prompt_string = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True 
        )
        
        # 4. 添加到列表
        prompts_list.append(prompt_string)
        y_true.append(ground_truth_label)
        raw_prompts_for_case_study.append(messages) # 保存原始 prompt 供案例分析

    except Exception as e:
        print(f"\n处理第 {index} 行时出错: {e}")

print(f"✅ 数据准备完毕，共 {len(prompts_list)} 条有效样本。")

# --- 4. 运行批量推理 (VLLM) ---
print("--- 开始在测试集上运行 VLLM 批量推理 ---")

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0,  # 必须是 0 (贪心采样)
    top_p=1.0,
    max_tokens=2048 # (1024 应该足够，如果不够再调大)
)

start_time = time.time()

# 一次性把所有 prompts 列表传给 vLLM
outputs = llm.generate(prompts_list, sampling_params)

end_time = time.time()

print(f"\n✅ VLLM 推理完成，耗时: {end_time - start_time:.4f} 秒")

# --- 5. 处理结果并打印报告 ---
print("--- 正在处理结果并生成报告 ---")

y_pred = []       # 存放所有预测标签 (P/Z)
case_studies = {"correct": [], "incorrect": []}
MAX_CASES = 5    

# 遍历 VLLM 的输出
for i, (output, ground_truth_label) in enumerate(tqdm(zip(outputs, y_true), total=min(len(outputs), len(y_true)), desc="处理输出")):
    # 先安全取出生成文本
    if not getattr(output, "outputs", None) or not output.outputs:
        generated_text = ""
        predicted_label = "PARSE_FAIL"
    else:
        text0 = output.outputs[0].text if output.outputs[0] and getattr(output.outputs[0], "text", None) else ""
        generated_text = text0 or ""
        predicted_label = parse_model_output(generated_text)

    y_pred.append(predicted_label)

    # 保存 Case Study
    if predicted_label == ground_truth_label:
        if len(case_studies["correct"]) < MAX_CASES:
            case_studies["correct"].append({
                "prompt": getattr(output, "prompt", None),
                "model_output": generated_text.strip(),
                "truth": REVERSE_LABEL_MAP[ground_truth_label]
            })
    else:
        #if len(case_studies["incorrect"]) < MAX_CASES:
        case_studies["incorrect"].append({
                "prompt": getattr(output, "prompt", None),
                "model_output": generated_text.strip(),
                "truth": REVERSE_LABEL_MAP[ground_truth_label],
                "predicted": REVERSE_LABEL_MAP.get(predicted_label, predicted_label)
            })


# --- 6. 打印性能报告 ---
print("\n\n--- 测试集性能报告 ---")

# 定义标签 (P=平, Z=仄)
target_names = ["P", "Z"]
# 只保留有效预测（排除 PARSE_FAIL）来计算 report 和 accuracy
valid_indices = [i for i, p in enumerate(y_pred) if p in target_names]
y_true_valid = [y_true[i] for i in valid_indices]
y_pred_valid = [y_pred[i] for i in valid_indices]
# 分类报告（只基于有效样本）
report = classification_report(
    y_true_valid,
    y_pred_valid,
    labels=target_names,
    digits=4,
    zero_division=0
)
print(report)
# 计算并打印准确率（基于有效样本）
if len(y_true_valid) > 0:
    acc = accuracy_score(y_true_valid, y_pred_valid)
    print(f"Accuracy (有效样本): {acc:.4f}  (n={len(y_true_valid)})")
else:
    print(" 没有有效样本用于计算 accuracy（所有样本均为 PARSE_FAIL 或无预测）。")

# 另外输出 PARSE_FAIL 数量（提示总样本数 vs 有效样本数）
parse_fails = sum(1 for p in y_pred if p == "PARSE_FAIL")
print(f"模型输出解析失败 (PARSE_FAIL) 数量: {parse_fails} / {len(y_pred)} (总样本)")

# --- 7. 打印 Case Study ---
print("\n\n--- Case Study (案例分析) ---")

print("\n---  5 个正确案例 (Correct Cases) ---")
for i, case in enumerate(case_studies["correct"]):
    print(f"\n[正确案例 #{i+1}] (真实: {case['truth']})")
    print(f"  --- 模型输出 ---\n{case['model_output']}")
    # print(f"  --- Prompt (部分) ---\n{case['prompt'][-500:]}") # VLLM prompt 包含了 <|im_start|> 等

total_errors = len(case_studies["incorrect"]) # <--- (!!) 新增这行
print(f"\n\n---  {total_errors} 个错误案例 (Incorrect Cases) ---")
if not case_studies["incorrect"]:
    print("太棒了！在分析的样本中没有发现错误案例。")
else:
    for i, case in enumerate(case_studies["incorrect"]):
        print(f"\n[错误案例 #{i+1}] (真实: {case['truth']}, 预测: {case['predicted']})")
        print(f"  --- 模型输出 ---\n{case['model_output']}")
        # print(f"  --- Prompt (部分) ---\n{case['prompt'][-500:]}")

print("\n--- 评测结束 ---")