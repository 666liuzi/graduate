import os
import json
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def extract_all_ingredients():

    PROJECT_ROOT = '/root/autodl-tmp/graduate/graduate'
    
    MODEL_DIR = '/root/autodl-tmp/models/zpeng1989/Chinese_Food_Qwen25vl_3B_Model'
    

    json_in = os.path.join(PROJECT_ROOT, 'results', 'baseline_visual_results.json')
    

    json_out = os.path.join(PROJECT_ROOT, 'results', 'multimodal_features.json')
    
    img_base_dir = os.path.join(PROJECT_ROOT, 'data', 'food-101', 'images')

    print("=== 阶段2：多模态大模型特征提取启动 ===")
    print("正在加载 Qwen2.5-VL-3B 模型 (bfloat16 完全体 + FlashAttention)...")
    
    # --- 2. 加载大模型 ---
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    # --- 3. 读取基线数据 ---
    if not os.path.exists(json_in):
        print(f"[错误] 找不到基线数据文件: {json_in}")
        return

    with open(json_in, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"共加载 {len(results)} 条视觉预测数据，开始提取食材特征...")

    # --- 4. 遍历提取 ---
    for raw_img_path, data in tqdm(results.items()):
        # 智能路径解析：防止 Windows/Linux 路径斜杠冲突，提取倒数两级 (类别名/图片名.jpg)
        parts = raw_img_path.replace('\\', '/').split('/')
        class_name, img_name = parts[-2], parts[-1]
        
        # 拼接出服务器上的真实物理路径
        actual_img_path = os.path.join(img_base_dir, class_name, img_name)
        
        if not os.path.exists(actual_img_path):
            print(f"[警告] 图片丢失，跳过: {actual_img_path}")
            continue

        # 构造大模型对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": actual_img_path},
                    {"type": "text", "text": "图上有什么？请详细解析食品组成。"},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # 推理并解码
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # 将大模型提取的食材信息存入原始数据中
        data['qwen_ingredients'] = output_text[0]

    # --- 5. 结果落盘 ---
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\n[完成] 所有多模态特征已提取并保存至: {json_out}")

if __name__ == "__main__":
    extract_all_ingredients()