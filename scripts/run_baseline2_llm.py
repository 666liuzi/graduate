import json
import os
from openai import OpenAI
from tqdm import tqdm

def run_baseline2():
    print("=== 实验(2)：纯多模态+大模型 (Qwen-VL + LLM) 基线测试启动 ===")
    
    # --- 1. API 配置 ---
    API_KEY = "sk-d91799a9de38433cb3b86cffa33fba5d" # 跑完后务必去后台删除重置
    BASE_URL = "https://api.deepseek.com"
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # --- 2. 绝对路径配置 ---
    PROJECT_ROOT = '/root/autodl-tmp/graduate/graduate'
    IN_FILE = os.path.join(PROJECT_ROOT, 'results', 'multimodal_features.json')
    OUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'baseline2_results.json')
    CLASSES_FILE = os.path.join(PROJECT_ROOT, 'data', 'food-101', 'meta', 'classes.txt')

    # --- 3. 读取 Food-101 的 101 个真实类别名称 ---
    if not os.path.exists(CLASSES_FILE):
        print(f"[错误] 找不到类别文件: {CLASSES_FILE}")
        return
        
    with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
        classes_list = [line.strip() for line in f.readlines() if line.strip()]
    
    classes_str = ", ".join(classes_list)

    # --- 4. 读取多模态特征数据 ---
    with open(IN_FILE, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    # 断点续传支持
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, 'r', encoding='utf-8') as f:
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

    # --- 5. 循环推理 ---
    processed_in_this_run = 0
    for img_path, data in tqdm(all_results.items()):
        if 'baseline2_pred' in data:
            continue
            
        decision = get_baseline2_decision(data)
        data['baseline2_pred'] = decision
        processed_in_this_run += 1

        if processed_in_this_run % 500 == 0:
            with open(OUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)

    # --- 6. 最终保存 ---
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n[完成] 实验(2) 纯大模型基线结果已保存至: {OUT_FILE}")

if __name__ == "__main__":
    run_baseline2()