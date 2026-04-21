import argparse
import json
import os
from openai import OpenAI
from tqdm import tqdm

def run_baseline2(input_file, output_file, classes_file, api_key, base_url):
    print("=== 实验(2)：纯多模态+大模型 (Qwen-VL + LLM) 基线测试启动 ===")
    
    # --- 1. API 配置 ---
    if not api_key:
        print("[错误] 未提供 API Key。请设置 DEEPSEEK_API_KEY 环境变量或通过 --api_key 传参。")
        return
        
    client = OpenAI(api_key=api_key, base_url=base_url)

    # --- 2. 读取 Food-101 的 101 个真实类别名称 ---
    if not os.path.exists(classes_file):
        print(f"[错误] 找不到类别文件: {classes_file}")
        return
        
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes_list = [line.strip() for line in f.readlines() if line.strip()]
    
    classes_str = ", ".join(classes_list)

    # --- 3. 读取多模态特征数据 ---
    if not os.path.exists(input_file):
        print(f"[错误] 找不到特征数据文件: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    # 断点续传支持
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            saved_results = json.load(f)
            for k, v in saved_results.items():
                if 'baseline2_pred' in v:
                    all_results[k] = v

    remaining_count = sum(1 for data in all_results.values() if 'baseline2_pred' not in data)
    print(f"总计 {len(all_results)} 条数据，剩余 {remaining_count} 条等待推理。开始请求...")

    def get_baseline2_decision(data):
        # 【核心修改】：完全隐瞒 edge_pred 和 cloud_pred
        prompt = f"""你是一个食品分类专家。请仅根据给定的“食材特征描述”，推断该食物属于以下哪个类别。

【候选词库】（必须且只能输出这101个词汇中的一个，不能输出中文）：
[{classes_str}]

【食材特征描述】：
{data.get('qwen_ingredients', '无')}

【决策规则】：
1. 分析上述食材特征，从候选词库中挑选一个最符合这些食材的英文类别名。
2. 严禁输出任何分析过程、标点符号或多余的解释文字。

最终的英文类别名称是："""

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个只输出候选词库中单词的机器人。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10,
                stream=False
            )
            return response.choices[0].message.content.strip().lower().replace(" ", "_").strip('`').strip("'").strip('"')
        except Exception as e:
            return "unknown" # 失败时标记为 unknown

    # --- 4. 循环推理 ---
    processed_in_this_run = 0
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    for img_path, data in tqdm(all_results.items()):
        if 'baseline2_pred' in data:
            continue
            
        decision = get_baseline2_decision(data)
        data['baseline2_pred'] = decision
        processed_in_this_run += 1

        if processed_in_this_run % 500 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)

    # --- 5. 最终保存 ---
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n[完成] 实验(2) 纯大模型基线结果已保存至: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="运行实验(2)：纯多模态+大模型基线测试")
    parser.add_argument("--input", default="./results/multimodal_features.json", help="多模态特征文件路径 (默认: ./results/multimodal_features.json)")
    parser.add_argument("--output", default="./results/baseline2_results.json", help="输出结果文件路径 (默认: ./results/baseline2_results.json)")
    parser.add_argument("--classes", default="./data/food-101/meta/classes.txt", help="类别字典文件路径 (默认: ./data/food-101/meta/classes.txt)")
    parser.add_argument("--api_key", default=os.environ.get("DEEPSEEK_API_KEY"), help="DeepSeek API Key (优先使用环境变量 DEEPSEEK_API_KEY)")
    parser.add_argument("--base_url", default="https://api.deepseek.com", help="模型 API 的 Base URL")
    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_args()
    run_baseline2(
        input_file=cli_args.input,
        output_file=cli_args.output,
        classes_file=cli_args.classes,
        api_key=cli_args.api_key,
        base_url=cli_args.base_url
    )