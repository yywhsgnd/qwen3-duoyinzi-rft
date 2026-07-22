# split_dataset.py
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import random

def split_dataset_811(input_file, output_prefix='duoyinzi'):
    """按照8:1:1的比例分割数据集为训练集、验证集、测试集"""
    
    print("加载数据...")
    # 从parquet文件加载数据
    df = pd.read_parquet(input_file)
    print(f"总数据量: {len(df)} 个样本")
    
    # 打乱数据
    print("打乱数据...")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 第一次分割：80% 训练集，20% 临时集
    train_df, temp_df = train_test_split(
        df_shuffled, 
        test_size=0.2, 
        random_state=42,
        stratify=df_shuffled['reward_model'].apply(lambda x: x['ground_truth'])
    )
    
    # 第二次分割：临时集再分为验证集和测试集（各50%，即整体的10%）
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42,
        stratify=temp_df['reward_model'].apply(lambda x: x['ground_truth'])
    )
    
    print(f"\n分割结果:")
    print(f"训练集: {len(train_df)} 样本 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"验证集: {len(val_df)} 样本 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"测试集: {len(test_df)} 样本 ({len(test_df)/len(df)*100:.1f}%)")
    
    # 统计每个集合的平仄分布
    def get_pingze_stats(dataframe):
        pingze_counts = dataframe['reward_model'].apply(lambda x: x['ground_truth']).value_counts()
        return pingze_counts
    
    print(f"\n平仄分布统计:")
    for split_name, split_df in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
        stats = get_pingze_stats(split_df)
        print(f"{split_name}:")
        for pingze, count in stats.items():
            print(f"  {pingze}: {count} 个 ({count/len(split_df)*100:.1f}%)")
    
    # 保存为parquet格式
    print(f"\n保存数据...")
    train_df.to_parquet(f'{output_prefix}_train.parquet', index=False)
    val_df.to_parquet(f'{output_prefix}_val.parquet', index=False)
    test_df.to_parquet(f'{output_prefix}_test.parquet', index=False)
    
    # 保存为JSONL格式（便于查看）
    train_df.to_json(f'{output_prefix}_train.jsonl', orient='records', lines=True, force_ascii=False)
    val_df.to_json(f'{output_prefix}_val.jsonl', orient='records', lines=True, force_ascii=False)
    test_df.to_json(f'{output_prefix}_test.jsonl', orient='records', lines=True, force_ascii=False)
    
    print(f"\n保存完成:")
    print(f"训练集: {output_prefix}_train.parquet, {output_prefix}_train.jsonl")
    print(f"验证集: {output_prefix}_val.parquet, {output_prefix}_val.jsonl")
    print(f"测试集: {output_prefix}_test.parquet, {output_prefix}_test.jsonl")
    
    # 显示每个集合的前几个字符统计
    def get_top_chars(dataframe, top_n=5):
        char_counts = {}
        for item in dataframe['extra_info']:
            char = item['character']
            char_counts[char] = char_counts.get(char, 0) + 1
        
        return sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"\n每个集合中出现最多的5个字:")
    for split_name, split_df in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
        top_chars = get_top_chars(split_df, 5)
        char_list = ", ".join([f"'{char}'({count})" for char, count in top_chars])
        print(f"{split_name}: {char_list}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # 分割数据集
    train, val, test = split_dataset_811("./rft1009/duoyinzi_rft_dataset.jsonl", "")