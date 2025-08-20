import os
import sys

# 添加OCR目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ocr_dir = os.path.join(current_dir, 'OCR')
sys.path.insert(0, ocr_dir)

from ocr_visualization import visualize_ocr_results_with_translation

def test_visualization():
    """测试OCR可视化功能"""
    
    # 文件路径
    image_path = "OCR/images/image1.jpg"
    #image_path = "OCR/images/image2.jpg"
    json_path = "OCR/ocr_output/output/image1_res.json"
    #json_path = "OCR/ocr_output/output/temp_ocr_image_temp_1753949138_res.json"
    output_path = "OCR/output/visualized_image1_with_translation.jpg"
    #output_path = "OCR/output/visualized_image2_with_translation.jpg"
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    if not os.path.exists(json_path):
        print(f"❌ JSON文件不存在: {json_path}")
        return
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"📁 图像文件: {image_path}")
    print(f"📁 JSON文件: {json_path}")
    print(f"📁 输出文件: {output_path}")
    
    # 模拟AI翻译结果（基于image1的实际内容）
    '''ai_response = """
[1] 大人のニキビにも -> 适用于成人痤疮  
[2] 第2类医薬品 -> 第二类药品  
[3] フェイスラインの吹き出物・ニキビも治療する -> 治疗面部轮廓的粉刺和痤疮  
[4] PAIR -> PAIR  
[5-6] 炎症をしずめ、しっかり効く -> 缓解炎症，效果显著  
[7] ペア。アクネクリーム -> Pair。痤疮膏  
[8] 肌にしっとり、透明になるクり一ム -> 润泽肌肤，使其透明的乳霜  
[9] 24g -> 24克  
[10] 税控除对象 -> 税收抵扣对象
"""'''
    ai_response = """[1] Усиленнаяверсия -> 加强版  
[2] Зкстракт трав -> 草本提取物  
[3] Без онемения -> 无麻木感  
[4] Продлевает + питает -> 延长并滋养  
[5-6] Безопасно,не вывываетпривыкания -> 安全，无依赖性  
[7] Цена -> 价格  
[8-9] CO скидкой -> 优惠  
[10] 598 -> 598  
[11] Быстрый -> 快速的  
[12-16] зффект: продление более 30 минут -> 效果：延时超过30分钟  
[17-19] Секрет мужской ВЫНОСЛИВОСТИ -> 男性耐力的秘密  
[20-21] Профессиональное средство -> 专业产品 
"""
    
    print("🎨 开始可视化OCR结果...")
    
    try:
        # 可视化OCR结果（包含翻译）
        results = visualize_ocr_results_with_translation(
            image_path, 
            json_path, 
            ai_response, 
            output_path
        )
        
        if results:
            print(f"✅ 可视化成功！共处理 {len(results)} 个文字块")
            print(f"📁 结果已保存到: {output_path}")
            
            # 打印识别结果
            print("\n📋 识别到的文字块:")
            for i, result in enumerate(results[:10]):  # 只显示前10个
                index = result['index']
                text = result['text']
                score = result['score']
                print(f"  {index}: {text} (置信度: {score:.3f})")
        else:
            print("❌ 未找到有效的OCR识别结果")
            
    except Exception as e:
        print(f"❌ 可视化失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization() 