import json
import os
from openai import OpenAI
from tqdm import tqdm

def run_ultimate_fusion():
    print("=== 阶段3（全量版）：DeepSeek-Chat 终极决策融合启动 ===")
    
    # --- 1. API 配置 ---
    API_KEY = "sk-d91799a9de38433cb3b86cffa33fba5d" # 跑完后务必去后台删除重置
    BASE_URL = "https://api.deepseek.com"
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # --- 2. 绝对路径配置 ---
    PROJECT_ROOT = '/root/autodl-tmp/graduate/graduate'
    IN_FILE = os.path.join(PROJECT_ROOT, 'results', 'multimodal_features.json')
    # 最终结果输出路径
    OUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'final_fusion_results.json')
    CLASSES_FILE = os.path.join(PROJECT_ROOT, 'data', 'food-101', 'meta', 'classes.txt')

    # --- 3. 读取 Food-101 的 101 个真实类别名称 ---
    if not os.path.exists(CLASSES_FILE):
        print(f"[错误] 找不到类别文件: {CLASSES_FILE}")
        return
        
    with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
        classes_list = [line.strip() for line in f.readlines() if line.strip()]
    
    classes_str = ", ".join(classes_list)

    # --- 4. 读取多模态特征数据 ---
    if not os.path.exists(IN_FILE):
        print(f"[错误] 找不到特征数据文件: {IN_FILE}")
        return

    with open(IN_FILE, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    # 如果存在已经跑了一部分的最终结果文件，读取它以实现断点续传
    if os.path.exists(OUT_FILE):
        print("检测到已存在的进度文件，正在合并断点数据...")
        with open(OUT_FILE, 'r', encoding='utf-8') as f:
            saved_results = json.load(f)
            # 更新 all_results，保留已经推理过的数据
            for k, v in saved_results.items():
                if 'final_fusion_pred' in v:
                    all_results[k] = v

    # 统计还需要跑多少条
    remaining_count = sum(1 for data in all_results.values() if 'final_fusion_pred' not in data)
    print(f"总计 {len(all_results)} 条数据，剩余 {remaining_count} 条等待推理。开始请求...")

    def get_llm_decision(data):
        # 将数字索引映射为真正的英文类别名
        edge_pred_name = classes_list[data.get('edge_pred', 0)]
        cloud_pred_name = classes_list[data.get('cloud_pred', 0)]
        
        prompt = f"""你是一个严格的分类纠错系统。请根据已知信息，推断该图片的最终食物名称。

【候选词库】（你输出的单词必须且只能是这101个词汇中的一个，绝对不能自己发明，不能输出中文）：
[{classes_str}]

【已知视觉先验信息】：
1. 端侧模型认为它是：{edge_pred_name} (置信度：{data.get('edge_conf', 0):.2f})
2. 云端模型认为它是：{cloud_pred_name} (置信度：{data.get('cloud_conf', 0):.2f})

【多模态提取食材特征】：
{data.get('qwen_ingredients', '无')}

【决策规则】：
1. 如果两个视觉模型预测的名字相同且置信度高，直接输出该英文名字。
2. 如果存在分歧，对比【多模态提取食材特征】，分析哪一个候选英文类别更符合这些食材（例如面粉+油炸，符合 churros），输出最合理的那个英文类别名。
3. 如果都不符合，从【候选词库】中挑选一个最符合食材特征的英文名。

只输出英文类别名称，不要任何解释。"""

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
            # 清洗大模型输出
            return response.choices[0].message.content.strip().lower().replace(" ", "_").strip('`').strip("'").strip('"')
        except Exception as e:
            # 如果单条API偶然报错，默认降级信任云端的判断，防止整批任务崩溃
            return cloud_pred_name

    # --- 5. 循环推理与断点保护 ---
    processed_in_this_run = 0
    
    for img_path, data in tqdm(all_results.items()):
        # 跳过已经推理过的数据
        if 'final_fusion_pred' in data:
            continue
            
        final_decision = get_llm_decision(data)
        data['final_fusion_pred'] = final_decision
        processed_in_this_run += 1

        # 每跑满 500 条自动保存一次进度
        if processed_in_this_run % 500 == 0:
            with open(OUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)

    # --- 6. 最终保存 ---
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n[全量跑通] 所有数据终极融合完毕，结果已安全保存至: {OUT_FILE}")

if __name__ == "__main__":
    run_ultimate_fusion()