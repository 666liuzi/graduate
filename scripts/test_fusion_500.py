import json
import os
from openai import OpenAI
from tqdm import tqdm

def run_test_fusion():
    print("=== 阶段3（修复版）：抽取 200 条样本进行深度融合验证 ===")
    
    # --- 1. API 配置 ---
    API_KEY = "sk-d91799a9de38433cb3b86cffa33fba5d" # 跑完后记得去后台删除重置
    BASE_URL = "https://api.deepseek.com"
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # --- 2. 绝对路径配置 ---
    PROJECT_ROOT = '/root/autodl-tmp/graduate/graduate'
    IN_FILE = os.path.join(PROJECT_ROOT, 'results', 'multimodal_features.json')
    OUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'test_fusion_200.json')
    CLASSES_FILE = os.path.join(PROJECT_ROOT, 'data', 'food-101', 'meta', 'classes.txt')

    # --- 3. 读取 Food-101 的 101 个真实类别名称 ---
    if not os.path.exists(CLASSES_FILE):
        print(f"[错误] 找不到类别文件: {CLASSES_FILE}")
        return
        
    with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
        # 去除换行符，并存为列表。索引 23 刚好对应 classes_list[23]
        classes_list = [line.strip() for line in f.readlines() if line.strip()]
    
    # 把列表拼接成字符串，塞给大模型做选择题
    classes_str = ", ".join(classes_list)

    # --- 4. 读取多模态特征数据 ---
    with open(IN_FILE, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    # 抽取前 200 条
    test_keys = list(all_results.keys())[:200]
    test_results = {k: all_results[k] for k in test_keys}

    print(f"成功加载类别映射表，开始向 DeepSeek 请求推理...")

    def get_llm_decision(data):
        # 【关键修复】：将数字索引映射为真正的英文类别名
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
            return cloud_pred_name

    # --- 5. 循环推理 ---
    for img_path, data in tqdm(test_results.items()):
        final_decision = get_llm_decision(data)
        data['final_fusion_pred'] = final_decision

    # --- 6. 保存测试结果 ---
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n[测试完成] 200 条数据已保存至: {OUT_FILE}")

if __name__ == "__main__":
    run_test_fusion()