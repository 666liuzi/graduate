import json

def analyze_trade_off(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
        
    total_samples = len(results)
    thresholds = [0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
    
    print("=== 端云协同基线性能分析 (Baseline 1) ===")
    for t in thresholds:
        correct = 0
        cloud_calls = 0
        
        for path, data in results.items():
            true_label = data['true_label']
            edge_pred = data['edge_pred']
            edge_conf = data['edge_conf']
            cloud_pred = data['cloud_pred']
            
            # 路由逻辑：大于等于阈值端侧决断，小于阈值云端兜底
            if edge_conf >= t:
                final_pred = edge_pred
            else:
                final_pred = cloud_pred
                cloud_calls += 1
                
            if final_pred == true_label:
                correct += 1
                
        accuracy = correct / total_samples * 100
        offload_rate = cloud_calls / total_samples * 100
        
        print(f"阈值 T={t:<4} | 准确率: {accuracy:.2f}% | 云端求助率: {offload_rate:.2f}%")

if __name__ == "__main__":
    analyze_trade_off('./results/baseline_visual_results.json')