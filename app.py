import os
import sys
import json
import base64
import io
import tempfile
import shutil
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS, cross_origin
from paddleocr import PaddleOCR
import openai
from dotenv import load_dotenv
import torch

# 添加OCR模块的导入
import sys
import os

# 添加OCR目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ocr_dir = os.path.join(current_dir, 'OCR')
sys.path.insert(0, ocr_dir)

# 导入OCR模块的函数
try:
    from ocr_test import load_mask_from_png, apply_mask_to_image
    from ocr_text_replacement import load_ocr_result, add_translated_text_enhanced, inpaint_text_area, OCRTextReplacer
    from ocr_visualization import visualize_ocr_results_with_translation
    print("✅ OCR模块导入成功")
except ImportError as e:
    print(f"警告：无法导入OCR模块: {e}")
    # 如果导入失败，定义空的占位函数
    def load_mask_from_png(mask_path):
        return None
    
    def apply_mask_to_image(image_path, mask):
        return None
    
    def load_ocr_result(json_path):
        """加载OCR结果JSON文件（备用实现）"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载OCR结果失败: {e}")
            return {}
    
    def add_translated_text_enhanced(image, box, translated_text):
        return image
    
    def inpaint_text_area(image, box, inpaint_radius=3):
        return image
    
    def visualize_ocr_results_with_translation(img_path, json_path, ai_resp, out_path):
        print("可视化功能不可用")
        return None

# 创建OCR文字替换器实例
try:
    ocr_replacer = OCRTextReplacer()
    print("✅ OCR文字替换器初始化成功")
except Exception as e:
    print(f"警告：OCR文字替换器初始化失败: {e}")
    ocr_replacer = None

def process_ocr_with_gpt_translation(image_path, json_path, output_path, translation_result):
    """使用GPT翻译结果处理OCR替换（使用新的ocr_text_replacement.py逻辑）"""
    try:
        # 获取翻译文本
        ai_response = translation_result.get('translation', '')
        
        # 使用新的ocr_text_replacement.py逻辑
        from ocr_text_replacement import process_ocr_with_translation
        
        # 调用新的处理函数，传递翻译文本
        valid_results = process_ocr_with_translation(
            image_path, 
            json_path, 
            output_path, 
            translation_text=ai_response
        )
        
        # 创建文字图层数据
        text_layers = []
        if valid_results:
            # 读取处理后的图片（inpaint后的图片）
            import cv2
            import numpy as np
            from PIL import Image
            
            # 读取原始图片
            img = cv2.imread(image_path)
            
            # 使用ocr_text_replacement.py的process_ocr_with_translation函数处理
            from ocr_text_replacement import process_ocr_with_translation
            
            # 调用process_ocr_with_translation函数，获取完整的处理结果
            process_result = process_ocr_with_translation(
                image_path, 
                json_path, 
                "temp_processed_with_translation.jpg", 
                translation_text=translation_result.get('translation', '')
            )
            
            # 添加类型检查和错误处理
            print(f"🔍 process_result 类型: {type(process_result)}")
            print(f"🔍 process_result 内容: {process_result}")
            
            # 使用返回的文字图层信息
            if process_result and isinstance(process_result, dict) and 'text_layers' in process_result:
                text_layers = process_result['text_layers']
                valid_results = process_result['valid_results']
                print(f"🔍 获取到文字图层，数量: {len(text_layers)}")
                print(f"🔍 文字图层详情:")
                for i, layer in enumerate(text_layers):
                    print(f"  图层 {i+1}: {layer}")
                
                # 对所有文字区域进行inpaint
                inpainted_img = img.copy()
                for i, result in enumerate(valid_results):
                    box = result['box']
                    # 处理不同的边界框格式
                    if isinstance(box[0], list):
                        # 格式: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                        x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])
                    else:
                        # 格式: [x1, y1, x2, y2]
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    # 使用inpaint去除原文字
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    inpainted_img = cv2.inpaint(inpainted_img, mask, 3, cv2.INPAINT_TELEA)
            else:
                text_layers = []
                valid_results = []
                print("🔍 未获取到文字图层信息")
            
            # 保存inpaint后的图片
            inpainted_output_path = "temp_inpainted_image.jpg"
            cv2.imwrite(inpainted_output_path, inpainted_img)
            
            # 不删除inpaint后的图片，因为后面还需要使用
        
        return {
            'valid_results': valid_results,
            'text_layers': text_layers,
            'inpainted_image_path': inpainted_output_path if os.path.exists(inpainted_output_path) else None
        }
        
    except Exception as e:
        print(f"处理OCR翻译失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return None

# 加载环境变量
load_dotenv()

# 如果.env文件不存在，尝试加载config.env
if not os.path.exists('.env') and os.path.exists('config.env'):
    load_dotenv('config.env')
    print("✅ 已加载config.env文件")

app = Flask(__name__)
CORS(app)

# 全局变量
model = None
ocr_model = None
TEST_GPT_RESPONSE = """
[1] -> 这是一个测试翻译结果
[2] -> 这是第二个测试翻译结果
[3-4] -> 这是合并的测试翻译结果
"""

def initialize_model():
    """初始化SAM模型"""
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        print("🔧 正在初始化SAM模型...")
        
        # 选择设备
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        print(f"使用设备: {device}")
        
        # 设置CUDA配置
        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # 构建SAM2模型
        checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        print("✅ SAM模型初始化成功")
        return predictor
    except Exception as e:
        print(f"❌ SAM模型初始化失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return None

def show_mask(mask, random_color=False, borders=True):
    """显示掩码"""
    # 确保mask是布尔类型
    mask = mask.astype(bool)
    
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, 4)
    return mask_image

def translate_ocr_results_with_gpt(ocr_results, target_language):
    """使用GPT翻译OCR结果"""
    try:
        # 构建提示词
        texts = [result['text'] for result in ocr_results]
        prompt = f"请将以下文字翻译成{target_language}，并按编号格式返回结果。\n\n"
        prompt += "重要说明：\n"
        prompt += "1. 仔细分析每个文字块的含义，只有在语义上确实需要组合才能表达完整意思时才合并\n"
        prompt += "2. 不要固定合并某些序号，要根据实际语义判断\n"
        prompt += "3. 如果单个文字块已经能表达完整意思，就不要合并\n"
        prompt += "4. 合并时要考虑语法和语义的连贯性\n\n"
        prompt += "5. 只需给出翻译，不要添加额外说明\n\n"
        prompt += "6. 如果有品牌名等难以翻译的词语，音译即可\n\n"
        prompt += "待翻译的文字：\n"
        for i, text in enumerate(texts, 1):
            prompt += f"[{i}] {text}\n"
        prompt += "\n请按以下格式返回翻译结果：\n"
        prompt += "示例格式：\n"
        prompt += "[1] 原文1 -> 翻译1\n"
        prompt += "[2-3] 原文2和3 -> 翻译2和3（只有当2和3在语义上需要组合时）\n"
        prompt += "[4] 原文4 -> 翻译4\n"
        prompt += "[5-6] 原文5和6 -> 翻译5和6（只有当5和6在语义上需要组合时）\n"
        prompt += "\n请根据实际语义判断是否需要合并，不要固定合并某些序号。"
        
        # 调用OpenAI API
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        
        translation = response.choices[0].message.content
        return {'success': True, 'translation': translation}
        
    except Exception as e:
        return {'error': f'翻译失败: {str(e)}'}

def extract_translation_from_ai_response(ai_response, original_text, block_index=None):
    """从AI响应中提取特定文本的翻译（支持新的prompt格式）"""
    try:
        lines = ai_response.strip().split('\n')
        for line in lines:
            if '[' in line and '->' in line:
                # 提取编号部分
                start = line.find('[') + 1
                end = line.find(']')
                if start > 0 and end > start:
                    num_part = line[start:end]
                    # 提取翻译部分（在 -> 之后）
                    translation_part = line.split('->', 1)[1].strip()
                    
                    # 处理合并的编号 [3-4]
                    if '-' in num_part:
                        start_num, end_num = map(int, num_part.split('-'))
                        if block_index and start_num <= block_index <= end_num:
                            return translation_part
                    else:
                        if block_index and int(num_part) == block_index:
                            return translation_part
                        
                    # 模糊匹配原文
                    if is_similar_text(original_text, translation_part):
                        return translation_part
        
        return None
    except Exception as e:
        print(f"提取翻译失败: {e}")
        return None

def get_merged_translations(ai_response):
    """解析AI响应中的合并翻译（支持新的prompt格式）"""
    merged_translations = {}
    try:
        lines = ai_response.strip().split('\n')
        for line in lines:
            if '[' in line and '->' in line and '-' in line:
                start = line.find('[') + 1
                end = line.find(']')
                if start > 0 and end > start:
                    num_part = line[start:end]
                    if '-' in num_part:
                        start_num, end_num = map(int, num_part.split('-'))
                        # 提取翻译部分（在 -> 之后）
                        translation_part = line.split('->', 1)[1].strip()
                        merged_translations[(start_num, end_num)] = translation_part
    except Exception as e:
        print(f"解析合并翻译失败: {e}")
    return merged_translations

def is_similar_text(text1, text2):
    """检查两个文本是否相似"""
    if not text1 or not text2:
        return False
    # 简单的相似度检查
    text1_clean = text1.lower().strip()
    text2_clean = text2.lower().strip()
    return text1_clean in text2_clean or text2_clean in text1_clean

def translate_image_with_gpt(image_base64, target_language):
    """使用GPT-4V翻译图像"""
    try:
        # 解码图像
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        
        # 调用OpenAI API
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"请分析这张图片中的文字内容，并将所有文字翻译成{target_language}，要求有双语对照。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        translation = response.choices[0].message.content
        return {'success': True, 'translation': translation}
        
    except Exception as e:
        return {'error': f'图像翻译失败: {str(e)}'}

def perform_ocr_and_save_json(image_path, mask_data=None, lang=None):
    """执行OCR并保存为JSON文件（完全按照ocr_test.py的逻辑）"""
    try:
        print(f"🔍 开始OCR处理...")
        print(f"  图像路径: {image_path}")
        print(f"  语言设置: {lang}")
        print(f"  掩码数据: {'有' if mask_data is not None else '无'}")
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            raise Exception(f"图像文件不存在: {image_path}")
        
        # 初始化 PaddleOCR 实例（完全按照ocr_test.py的逻辑）
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang if lang else 'ch'  # 与ocr_test.py保持一致，默认使用'ch'
        )
        
        print("✅ OCR模型初始化成功")
        
        # 处理掩码（如果提供）
        ocr_input = image_path
        if mask_data is not None:
            print("🔍 处理掩码数据...")
            # 将掩码数据转换为numpy数组
            mask = np.array(mask_data, dtype=np.uint8)
            print(f"  掩码形状: {mask.shape}")
            print(f"  掩码值范围: {mask.min()} - {mask.max()}")
            
            # 确保掩码是0-255范围
            if mask.max() <= 1:  # 如果是0-1范围，转换为0-255
                mask = mask * 255
                print(f"  掩码值范围已调整: {mask.min()} - {mask.max()}")
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("无法读取图像文件")
            
            print(f"  图像形状: {image.shape}")
            
            # 调整掩码尺寸
            if mask.shape[:2] != image.shape[:2]:
                print(f"  调整掩码尺寸从 {mask.shape[:2]} 到 {image.shape[:2]}")
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # 应用掩码（完全按照ocr_test.py的逻辑）
            white_bg = np.full_like(image, 255)  # (255,255,255) 白色背景
            masked_image = np.where(mask[:, :, None] == 255, image, white_bg)
            
            # 保存掩码图像
            masked_path = image_path.replace('.jpg', '_masked.jpg')
            cv2.imwrite(masked_path, masked_image)
            print(f"  掩码图像已保存: {masked_path}")
            ocr_input = masked_path
        else:
            print(f"  使用原始图像: {ocr_input}")
        
        # 执行OCR（完全按照ocr_test.py的逻辑）
        print("🔍 执行OCR识别...")
        try:
            result = ocr.predict(input=ocr_input)
            print(f"  OCR原始结果类型: {type(result)}")
            print(f"  OCR原始结果长度: {len(result) if result else 0}")
        except Exception as e:
            print(f"❌ OCR执行失败: {e}")
            raise
        
        # 处理OCR结果（完全按照ocr_test.py的逻辑）
        print("🔍 处理OCR结果...")
        
        if not result:
            print("  警告: OCR结果为空")
            return None, []
        
        print(f"  OCR结果数量: {len(result)}")
        
        # 创建临时输出目录
        temp_output_dir = "temp_ocr_output"
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # 保存结果（完全按照ocr_test.py的逻辑）
        for res in result:
            print(f"    处理结果: {type(res)}")
            res.print()  # 打印结果
            res.save_to_img(temp_output_dir)  # 保存图像
            res.save_to_json(temp_output_dir)  # 保存JSON
        
        # 读取保存的JSON文件
        json_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.json')]
        if json_files:
            json_path = os.path.join(temp_output_dir, json_files[0])
            print(f"  JSON文件路径: {json_path}")
            
            # 按照ocr_visualization.py的逻辑提取JSON数据
            try:
                ocr_data = load_ocr_result(json_path)
                print(f"  JSON数据键: {list(ocr_data.keys())}")
                
                # 使用zip方法提取数据（按照ocr_visualization.py的逻辑）
                rec_texts = ocr_data.get('rec_texts', [])
                rec_scores = ocr_data.get('rec_scores', [])
                rec_boxes = ocr_data.get('rec_boxes', [])
                
                print(f"  提取到 {len(rec_texts)} 个文本")
                print(f"  提取到 {len(rec_scores)} 个分数")
                print(f"  提取到 {len(rec_boxes)} 个边界框")
                
                # 使用zip方法组合数据（按照ocr_visualization.py的逻辑）
                rec = zip(rec_texts, rec_scores, rec_boxes)
                valid = [{'index': i + 1, 'text': t, 'score': s, 'box': b}
                         for i, (t, s, b) in enumerate(rec) if t.strip() and s > 0.5]  # 置信度阈值0.5
                
                print(f"  有效结果数量: {len(valid)}")
                
                # 转换为结果格式
                results = []
                for item in valid:
                    results.append({
                        'text': item['text'].strip(),
                        'confidence': float(item['score']),
                        'box': item['box']
                    })
                
                print(f"✅ 成功提取到 {len(results)} 个OCR结果")
                
                # 移动JSON文件到目标位置
                target_json_path = image_path.replace('.jpg', '_res.json')
                if os.path.exists(json_path):
                    import shutil
                    shutil.move(json_path, target_json_path)
                    print(f"✅ OCR结果已保存到: {target_json_path}")
                    
                    # 额外保存一份到OCR输出目录，方便查看
                    ocr_output_dir = "OCR/ocr_output/output"
                    os.makedirs(ocr_output_dir, exist_ok=True)
                    
                    # 生成带时间戳的文件名，避免覆盖
                    import time
                    timestamp = int(time.time())
                    base_name = os.path.basename(image_path).replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
                    extra_json_path = os.path.join(ocr_output_dir, f"{base_name}_temp_{timestamp}_res.json")
                    
                    shutil.copy2(target_json_path, extra_json_path)
                    print(f"✅ OCR结果额外保存到: {extra_json_path}")
                    
                    # 同时保存原始图像到OCR输出目录
                    extra_image_path = os.path.join(ocr_output_dir, f"{base_name}_temp_{timestamp}.jpg")
                    shutil.copy2(image_path, extra_image_path)
                    print(f"✅ 原始图像已保存到: {extra_image_path}")
                
                # 清理临时目录
                if os.path.exists(temp_output_dir):
                    import shutil
                    shutil.rmtree(temp_output_dir)
                
                # 清理临时掩码图像
                if mask_data is not None and os.path.exists(ocr_input):
                    os.remove(ocr_input)
                    print(f"✅ 清理临时掩码图像: {ocr_input}")
                
                return target_json_path, results
                
            except Exception as e:
                print(f"❌ JSON数据提取失败: {e}")
                import traceback
                print(f"错误详情: {traceback.format_exc()}")
                return None, []
        else:
            print("❌ 未找到JSON文件")
            return None, []
        
    except Exception as e:
        print(f"❌ OCR处理失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return None, []

@app.route('/api/upload', methods=['POST'])
@cross_origin()
def upload_image():
    """上传图像"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': '缺少图像数据'}), 400
        
        image_base64 = data['image']
        return jsonify({'success': True, 'message': '图像上传成功'})
        
    except Exception as e:
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/api/segment', methods=['POST'])
@cross_origin()
def segment_image():
    """分割图像"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': '缺少图像数据'}), 400
        
        image_base64 = data['image']
        points = data.get('points', [])  # 从HTML获取points
        labels = data.get('labels', [])  # 从HTML获取labels
        
        if not points or not labels:
            return jsonify({'error': '缺少点数据或标签数据'}), 400
        
        print(f"🔍 分割调试信息:")
        print(f"  点的数量: {len(points)}")
        print(f"  标签数量: {len(labels)}")
        print(f"  前景点: {sum(1 for label in labels if label == 1)}")
        print(f"  背景点: {sum(1 for label in labels if label == 0)}")
        
        # 初始化模型
        global model
        if model is None:
            model = initialize_model()
            if model is None:
                return jsonify({'error': '模型初始化失败'}), 500
        
        # 解码图像
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 设置图像到预测器
        model.set_image(image)
        
        # 转换为numpy数组
        points_array = np.array(points)
        labels_array = np.array(labels)
        
        print(f"  点坐标: {points_array}")
        print(f"  标签: {labels_array}")
        
        # 执行分割
        with torch.inference_mode():
            masks, scores, logits = model.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=True
            )
        
        # 选择最佳掩码
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        print(f"  最佳掩码索引: {best_mask_idx}")
        print(f"  最佳分数: {best_score}")
        
        # 生成掩码图像
        mask_image = show_mask(best_mask)
        mask_pil = Image.fromarray((mask_image * 255).astype(np.uint8))
        
        # 生成反转掩码
        inverted_mask = ~best_mask.astype(bool)
        inverted_mask_image = show_mask(inverted_mask)
        inverted_mask_pil = Image.fromarray((inverted_mask_image * 255).astype(np.uint8))
        
        # 生成混合图像（原图+掩码）
        image_np = np.array(image.convert("RGBA"))
        mask_overlay = show_mask(best_mask)
        overlay_uint8 = (mask_overlay * 255).astype(np.uint8)
        blended = Image.alpha_composite(Image.fromarray(image_np), Image.fromarray(overlay_uint8))
        
        # 将混合结果叠加到原图（更新canvas显示）
        blended_np = np.array(blended)
        blended_rgb = cv2.cvtColor(blended_np, cv2.COLOR_RGBA2RGB)
        blended_pil = Image.fromarray(blended_rgb)
        
        # 保存混合图像用于显示
        output_buffer = io.BytesIO()
        blended_pil.save(output_buffer, format='PNG')
        blended_base64 = base64.b64encode(output_buffer.getvalue()).decode()
        
        # 生成透明背景的掩码结果
        image_rgba = image.convert("RGBA")
        image_array = np.array(image_rgba)
        mask_3d = np.stack([best_mask.astype(bool)] * 4, axis=-1)
        masked_result = image_array * mask_3d
        masked_result_pil = Image.fromarray(masked_result.astype(np.uint8))
        
        # 转换为base64
        output_buffer = io.BytesIO()
        mask_pil.save(output_buffer, format='PNG')
        mask_base64 = base64.b64encode(output_buffer.getvalue()).decode()
        
        output_buffer = io.BytesIO()
        inverted_mask_pil.save(output_buffer, format='PNG')
        inverted_mask_base64 = base64.b64encode(output_buffer.getvalue()).decode()
        
        output_buffer = io.BytesIO()
        masked_result_pil.save(output_buffer, format='PNG')
        masked_result_base64 = base64.b64encode(output_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'mask': mask_base64,
            'inverted_mask': inverted_mask_base64,
            'blended': blended_base64,
            'masked_result': masked_result_base64,
            'score': float(best_score),
            'mask_data': best_mask.tolist()
        })
        
    except Exception as e:
        import traceback
        print(f"❌ 分割失败: {str(e)}")
        print(f"错误详情: {traceback.format_exc()}")
        return jsonify({'error': f'分割失败: {str(e)}'}), 500

@app.route('/api/ocr_translate', methods=['POST'])
@cross_origin()
def ocr_translate():
    """OCR识别并翻译图像中的文字"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '无效的请求数据'}), 400
        
        image_base64 = data.get('image_base64')
        target_language = data.get('target_language', 'Chinese')
        ocr_mode = data.get('ocr_mode', 'full')  # full, mask, mixed
        source_language = data.get('source_language', '')
        mask_data = data.get('mask_data', None)
        
        if not image_base64:
            return jsonify({'error': '缺少图像数据'}), 400
        
        # 确定OCR语言参数
        lang = 'ch'  # 默认使用中文，与ocr_test.py一致
        if source_language:
            lang_map = {
                'Korean': 'korean',
                'Russian': 'ru',
                'Japanese': 'japan',
                'English': 'en',
                'Chinese': 'ch',
                'French': 'fr'
            }
            lang = lang_map.get(source_language, 'ch')  # 如果映射失败，默认使用'ch'
        
        print(f"🔍 OCR调试信息:")
        print(f"  用户选择的原语言: {source_language}")
        print(f"  映射后的OCR语言参数: {lang}")
        print(f"  OCR模式: {ocr_mode}")
        
        # 模式1: 全图OCR翻译（带可视化）
        if ocr_mode == 'full':
            # 获取可视化语言类型
            visualization_language = data.get('visualization_language', 'chinese')  # 默认使用中文
            return handle_full_image_ocr(image_base64, target_language, lang, visualization_language)
        
        # 模式2: 背景OCR翻译并替换
        elif ocr_mode == 'mask':
            return handle_mask_ocr_replace(image_base64, target_language, lang, mask_data)
        
        # 模式3: 商品部分翻译，背景翻译并替换
        elif ocr_mode == 'mixed':
            return handle_mixed_ocr_replace(image_base64, target_language, lang, mask_data)
        
        else:
            return jsonify({'error': f'不支持的OCR模式: {ocr_mode}'}), 400
        
    except Exception as e:
        return jsonify({'error': f'OCR翻译失败: {str(e)}'}), 500

def handle_full_image_ocr(image_base64, target_language, lang, visualization_language=None):
    """处理全图OCR翻译（带可视化）
    
    Args:
        image_base64: 图像数据
        target_language: 目标翻译语言
        lang: OCR语言参数
        visualization_language: 可视化中使用的语言类型（'chinese' 或 'english'）
    """
    try:
        # 解码图像
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 保存临时图像文件（与ocr_test.py一致，直接保存原始图像）
        temp_image_path = "temp_ocr_image.jpg"
        try:
            image.save(temp_image_path, 'JPEG', quality=95)
            print(f"✅ 临时图像保存成功: {temp_image_path}")
            print(f"  文件大小: {os.path.getsize(temp_image_path)} 字节")
        except Exception as save_error:
            print(f"❌ 保存临时图像失败: {save_error}")
            return jsonify({'error': f'保存临时图像失败: {str(save_error)}'}), 500
        
        # 执行OCR并保存JSON（与ocr_test.py一致）
        json_path, results = perform_ocr_and_save_json(temp_image_path, lang=lang)
        
        if not results:
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if json_path and os.path.exists(json_path):
                os.remove(json_path)
            return jsonify({'error': '未识别到任何文字'}), 400
        
        print(f"✅ 成功提取到 {len(results)} 个OCR结果")
        
        # 翻译OCR结果
        print("🤖 开始翻译OCR结果...")
        translation_result = translate_ocr_results_with_gpt(results, target_language)
        
        # 检查翻译是否成功
        if not translation_result.get('success'):
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if json_path and os.path.exists(json_path):
                os.remove(json_path)
            return jsonify(translation_result), 500
        
        # 生成可视化
        try:
            # 使用ocr_visualization模块生成可视化
            temp_viz_path = "temp_visualization.jpg"
            
            # 确保输出目录存在
            viz_dir = os.path.dirname(temp_viz_path)
            if viz_dir:
                os.makedirs(viz_dir, exist_ok=True)
            
            print(f"🔍 开始生成可视化...")
            print(f"  图像路径: {temp_image_path}")
            print(f"  JSON路径: {json_path}")
            print(f"  输出路径: {temp_viz_path}")
            
            # 检查文件是否存在
            print(f"🔍 检查文件存在性:")
            print(f"  图像文件: {temp_image_path} - {'存在' if os.path.exists(temp_image_path) else '不存在'}")
            print(f"  JSON文件: {json_path} - {'存在' if os.path.exists(json_path) else '不存在'}")
            
            if not os.path.exists(temp_image_path):
                raise Exception(f"图像文件不存在: {temp_image_path}")
            if not os.path.exists(json_path):
                raise Exception(f"JSON文件不存在: {json_path}")
            
            # 修复路径问题：如果输出路径没有目录，使用当前目录
            if not os.path.dirname(temp_viz_path):
                temp_viz_path = f"./{temp_viz_path}"
                print(f"  修正输出路径: {temp_viz_path}")
            
            # 根据用户选择的语言类型决定可视化字体
            print(f"🎨 可视化语言类型: {visualization_language}")
            
            # 调用可视化函数，使用现有的动态边界值功能
            visualize_ocr_results_with_translation(
                temp_image_path,
                json_path,
                translation_result.get('translation', ''),
                temp_viz_path,
                visualization_language=visualization_language
            )
            
            # 检查可视化文件是否生成
            if os.path.exists(temp_viz_path):
                # 转换为base64
                with open(temp_viz_path, 'rb') as f:
                    visualization_base64 = base64.b64encode(f.read()).decode()
                
                print(f"✅ 可视化生成成功: {temp_viz_path}")
            else:
                print(f"❌ 可视化文件未生成: {temp_viz_path}")
                visualization_base64 = None
            
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if json_path and os.path.exists(json_path):
                os.remove(json_path)
            if os.path.exists(temp_viz_path):
                os.remove(temp_viz_path)
            
            response_data = {
                'success': True,
                'translation': translation_result.get('translation', ''),
                'ocr_results': results
            }
            
            if visualization_base64:
                response_data['visualization'] = visualization_base64
            
            return jsonify(response_data)
            
        except Exception as viz_error:
            print(f"❌ 可视化生成失败: {viz_error}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
            
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if json_path and os.path.exists(json_path):
                os.remove(json_path)
            
            # 返回不带可视化的结果
            return jsonify({
                'success': True,
                'translation': translation_result.get('translation', ''),
                'ocr_results': results
            })
    
    except Exception as e:
        return jsonify({'error': f'全图OCR处理失败: {str(e)}'}), 500

def handle_mask_ocr_replace(image_base64, target_language, lang, mask_data):
    """处理背景OCR翻译并替换"""
    try:
        # 解码图像
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 保存临时图像文件
        temp_image_path = "temp_ocr_image.jpg"
        image.save(temp_image_path)
        
        # 翻转mask（非商品部分翻译并替换）
        if mask_data is not None:
            import numpy as np
            inverted_mask = np.array(mask_data, dtype=np.uint8)
            # 翻转mask：255变0，0变255
            inverted_mask = 255 - inverted_mask
            print("🔄 已翻转mask（非商品部分翻译并替换）")
        else:
            inverted_mask = None
        
        # 执行OCR并保存JSON（使用翻转后的掩码）
        json_path, results = perform_ocr_and_save_json(temp_image_path, inverted_mask, lang)
        
        if not results:
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if json_path and os.path.exists(json_path):
                os.remove(json_path)
            return jsonify({'error': '未识别到任何文字'}), 400
        
        print(f"✅ 成功提取到 {len(results)} 个OCR结果")
        
        # 翻译OCR结果
        print("🤖 开始翻译OCR结果...")
        translation_result = translate_ocr_results_with_gpt(results, target_language)
        
        # 检查翻译是否成功
        if not translation_result.get('success'):
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if json_path and os.path.exists(json_path):
                os.remove(json_path)
            return jsonify(translation_result), 500
        
        # 使用ocr_text_replacement模块处理替换
        try:
            temp_output_path = "temp_replaced_image.jpg"
            process_result = process_ocr_with_gpt_translation(temp_image_path, json_path, temp_output_path, translation_result)
            
            # 添加类型检查和错误处理
            print(f"🔍 handle_mask_ocr_replace - process_result 类型: {type(process_result)}")
            print(f"🔍 handle_mask_ocr_replace - process_result 内容: {process_result}")
            
            # 转换为base64
            with open(temp_output_path, 'rb') as f:
                replaced_image_base64 = base64.b64encode(f.read()).decode()
            
            # 处理文字图层数据
            text_layers = []
            processed_image_base64 = None
            
            if process_result and process_result.get('text_layers'):
                text_layers = process_result['text_layers']
                
                # 如果有inpaint后的图片，转换为base64
                if process_result.get('inpainted_image_path') and os.path.exists(process_result['inpainted_image_path']):
                    with open(process_result['inpainted_image_path'], 'rb') as f:
                        processed_image_base64 = base64.b64encode(f.read()).decode()
            
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if json_path and os.path.exists(json_path):
                os.remove(json_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            if process_result and process_result.get('inpainted_image_path') and os.path.exists(process_result['inpainted_image_path']):
                os.remove(process_result['inpainted_image_path'])
            
            return jsonify({
                'success': True,
                'translation': translation_result.get('translation', ''),
                'ocr_results': results,
                'replaced_image': replaced_image_base64,
                'text_layers': text_layers,
                'processed_image': processed_image_base64
            })
            
        except Exception as replace_error:
            print(f"❌ 文字替换失败: {replace_error}")
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if json_path and os.path.exists(json_path):
                os.remove(json_path)
            return jsonify({'error': f'文字替换失败: {str(replace_error)}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'背景OCR处理失败: {str(e)}'}), 500

def handle_mixed_ocr_replace(image_base64, target_language, lang, mask_data):
    """处理商品部分翻译，背景翻译并替换"""
    print(f"🔍 handle_mixed_ocr_replace 开始执行")
    try:
        # 解码图像
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 保存临时图像文件
        temp_image_path = "temp_ocr_image.jpg"
        image.save(temp_image_path)
        
        # 翻转mask（商品部分翻译，非商品部分替换）
        if mask_data is not None:
            import numpy as np
            inverted_mask = np.array(mask_data, dtype=np.uint8)
            # 翻转mask：255变0，0变255
            inverted_mask = 255 - inverted_mask
            print("🔄 已翻转mask（商品部分翻译，非商品部分替换）")
        else:
            inverted_mask = None
        
        # 执行OCR并保存JSON（使用翻转后的掩码）
        json_path, results = perform_ocr_and_save_json(temp_image_path, inverted_mask, lang)
        
        if not results:
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if json_path and os.path.exists(json_path):
                os.remove(json_path)
            return jsonify({'error': '未识别到任何文字'}), 400
        
        print(f"✅ 成功提取到 {len(results)} 个OCR结果")
        
        # 翻译OCR结果
        print("🤖 开始翻译OCR结果...")
        translation_result = translate_ocr_results_with_gpt(results, target_language)
        
        # 检查翻译是否成功
        if not translation_result.get('success'):
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if json_path and os.path.exists(json_path):
                os.remove(json_path)
            return jsonify(translation_result), 500
        
        # 解析翻译结果
        ai_response = translation_result.get('translation', '')
        merged_translations = get_merged_translations(ai_response)
        
        # 读取图像进行混合处理
        img = cv2.imread(temp_image_path)
        
        # 使用翻转后的mask数据，用于区分商品和背景
        if mask_data is not None:
            # 翻转mask：255变0，0变255（与OCR时保持一致）
            inverted_mask = np.array(mask_data, dtype=np.uint8)
            inverted_mask = 255 - inverted_mask
            mask_resized = cv2.resize(inverted_mask, (img.shape[1], img.shape[0]))
        else:
            # 如果没有mask，假设所有文字都是背景
            mask_resized = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        # 第一遍：对所有block进行inpaint
        for i, result in enumerate(results):
            box = result['box']
            x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])
            
            # 使用inpaint去除原文字
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        
        # 保存inpaint后的图片用于文字图层编辑
        inpainted_img = img.copy()
        
        # 准备文字图层数据 - 使用OCR模块的统一逻辑
        text_layers = []
        
        # 使用OCR模块的文字图层生成逻辑
        from ocr_text_replacement import process_ocr_with_translation
        
        # 调用OCR模块处理，获取统一的文字图层
        try:
            process_result = process_ocr_with_translation(
                temp_image_path, 
                json_path, 
                "temp_processed_with_translation.jpg", 
                translation_text=ai_response
            )
            
            # 添加类型检查和错误处理
            print(f"🔍 process_result 类型: {type(process_result)}")
            print(f"🔍 process_result 内容: {process_result}")
            
            if process_result and isinstance(process_result, dict) and 'text_layers' in process_result:
                # 使用OCR模块生成的文字图层，但根据商品区域调整显示文本
                ocr_text_layers = process_result['text_layers']
                
                print(f"🔍 从OCR模块获取的文字图层信息:")
                print(f"  原始图层数量: {len(ocr_text_layers)}")
                for i, layer in enumerate(ocr_text_layers):
                    print(f"  原始图层 {i+1}: {layer['original_text']} -> {layer['translated_text']}")
                    print(f"    颜色: {layer['text_color']}, 字体大小: {layer['font_size']}")
                
                for i, layer in enumerate(ocr_text_layers):
                    box = layer['box']
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    # 检查这个区域是否在商品mask内（使用翻转后的mask）
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    # 由于mask已翻转，现在255表示商品区域，0表示背景区域
                    is_in_product = mask_resized[center_y, center_x] > 0 if center_y < mask_resized.shape[0] and center_x < mask_resized.shape[1] else False
                    
                    # 根据是否在商品区域内调整显示文本
                    if is_in_product:
                        # 商品区域：只翻译，不替换（显示双语）
                        display_text = f"{layer['original_text']} -> {layer['translated_text']}"
                        # 使用红色突出显示
                        text_color = [255, 0, 0]
                    else:
                        # 背景区域：替换为翻译
                        display_text = layer['translated_text']
                        # 直接使用OCR模块返回的颜色，确保完全一致
                        text_color = layer['text_color']
                    
                    # 确保颜色格式正确
                    if isinstance(text_color, tuple):
                        text_color = list(text_color)
                    elif not isinstance(text_color, list):
                        text_color = [0, 0, 0]  # 默认黑色
                    
                    # 创建文字图层数据，完全使用OCR模块的信息
                    text_layer = {
                        'id': f'layer_{i}',
                        'original_text': layer['original_text'],
                        'translated_text': display_text,
                        'box': layer['box'],
                        'text_color': text_color,
                        'font_size': layer['font_size'],
                        'visible': True,
                        'is_in_product': is_in_product,
                        'font_family': layer.get('font_family', 'Microsoft YaHei'),
                        'spacing': layer.get('spacing', 0),
                        'text_height': layer.get('text_height', layer['font_size'])  # 添加文字高度
                    }
                    text_layers.append(text_layer)
            else:
                # 如果OCR模块处理失败，使用备用逻辑
                print(f"⚠️ OCR模块处理失败或返回意外数据类型，使用备用逻辑")
                print(f"  process_result 类型: {type(process_result) if process_result else 'None'}")
                
                for i, result in enumerate(results):
                    box = result['box']
                    x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])
                
                # 检查这个区域是否在商品mask内
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                is_in_product = mask_resized[center_y, center_x] > 0 if center_y < mask_resized.shape[0] and center_x < mask_resized.shape[1] else False
                
                # 查找翻译
                translation = None
                block_index = i + 1
                
                # 检查是否在合并翻译中
                for (start, end), merged_trans in merged_translations.items():
                    if start <= block_index <= end:
                        translation = merged_trans
                        break
                
                # 如果没找到合并翻译，尝试单独翻译
                if not translation:
                    translation = extract_translation_from_ai_response(ai_response, result['text'], block_index)
                
                if translation:
                    # 根据是否在商品区域内选择颜色和显示文本
                    if is_in_product:
                        text_color = [255, 0, 0]  # 红色
                        display_text = f"{result['text']} -> {translation}"
                    else:
                        # 使用OCR模块的颜色计算逻辑
                        from ocr_text_replacement import get_dominant_color, get_contrast_color
                        box = [x1, y1, x2, y2]
                        bg_color = get_dominant_color(img, box)
                        text_color = list(get_contrast_color(bg_color))
                        display_text = translation
                    
                    # 创建文字图层数据
                    text_layer = {
                        'id': f'layer_{i}',
                        'original_text': result['text'],
                        'translated_text': display_text,
                        'box': [x1, y1, x2, y2],
                        'text_color': text_color,
                        'font_size': 20,  # 默认字体大小
                        'visible': True,
                        'is_in_product': is_in_product,
                        'font_family': 'Microsoft YaHei',
                        'spacing': 0
                    }
                    text_layers.append(text_layer)
        except Exception as ocr_error:
            print(f"❌ OCR模块处理异常: {ocr_error}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
            # 使用备用逻辑
            for i, result in enumerate(results):
                box = result['box']
                x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])
                
                # 检查这个区域是否在商品mask内
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                is_in_product = mask_resized[center_y, center_x] > 0 if center_y < mask_resized.shape[0] and center_x < mask_resized.shape[1] else False
                
                # 查找翻译
                translation = None
                block_index = i + 1
                
                # 检查是否在合并翻译中
                for (start, end), merged_trans in merged_translations.items():
                    if start <= block_index <= end:
                        translation = merged_trans
                        break
                
                # 如果没找到合并翻译，尝试单独翻译
                if not translation:
                    translation = extract_translation_from_ai_response(ai_response, result['text'], block_index)
                
                if translation:
                    # 根据是否在商品区域内选择颜色和显示文本
                    if is_in_product:
                        text_color = [255, 0, 0]  # 红色
                        display_text = f"{result['text']} -> {translation}"
                    else:
                        # 使用OCR模块的颜色计算逻辑
                        from ocr_text_replacement import get_dominant_color, get_contrast_color
                        box = [x1, y1, x2, y2]
                        bg_color = get_dominant_color(img, box)
                        text_color = list(get_contrast_color(bg_color))
                        display_text = translation
                    
                    # 创建文字图层数据
                    text_layer = {
                        'id': f'layer_{i}',
                        'original_text': result['text'],
                        'translated_text': display_text,
                        'box': [x1, y1, x2, y2],
                        'text_color': text_color,
                        'font_size': 20,  # 默认字体大小
                        'visible': True,
                        'is_in_product': is_in_product,
                        'font_family': 'Microsoft YaHei',
                        'spacing': 0
                    }
                    text_layers.append(text_layer)
        
        # 现在绘制文字到最终图片（用于显示）- 使用OCR模块的统一逻辑
        from ocr_text_replacement import add_translated_text_enhanced
        
        # 使用inpaint后的图片作为背景
        img_result = inpainted_img.copy()
        
        # 使用OCR模块的文字绘制逻辑
        for layer in text_layers:
            box = layer['box']
            text = layer['translated_text']
            text_color = layer['text_color']
            font_size = layer['font_size']
            font_family = layer.get('font_family', 'Microsoft YaHei')
            spacing = layer.get('spacing', 0)
            
            # 使用OCR模块的统一绘制函数
            img_result = add_translated_text_enhanced(
                img_result, 
                box, 
                text, 
                text_color=text_color, 
                font_size=font_size,
                font_family=font_family,
                spacing=spacing
            )
        
        # img_result已经是OpenCV格式，无需转换
        
        # 保存结果
        output_path = "temp_mixed_replaced_image.jpg"
        cv2.imwrite(output_path, img_result)
        
        # 转换为base64
        with open(output_path, 'rb') as f:
            replaced_image_base64 = base64.b64encode(f.read()).decode()
        
        # 清理临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if json_path and os.path.exists(json_path):
            os.remove(json_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # 转换inpaint后的图片为base64（用于文字图层编辑）
        inpainted_img_pil = Image.fromarray(cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB))
        inpainted_output_path = "temp_inpainted_image.jpg"
        inpainted_img_pil.save(inpainted_output_path)
        
        with open(inpainted_output_path, 'rb') as f:
            inpainted_image_base64 = base64.b64encode(f.read()).decode()
        
        # 清理临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if json_path and os.path.exists(json_path):
            os.remove(json_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        if os.path.exists(inpainted_output_path):
            os.remove(inpainted_output_path)
        
        print(f"🔍 返回数据调试信息:")
        print(f"  文字图层数量: {len(text_layers)}")
        print(f"  text_layers 类型: {type(text_layers)}")
        print(f"  processed_image 存在: {'是' if inpainted_image_base64 else '否'}")
        for i, layer in enumerate(text_layers):
            print(f"  图层 {i}: {layer['original_text']} -> {layer['translated_text']}")
            print(f"    颜色: {layer['text_color']}, 字体大小: {layer['font_size']}, 字体类型: {layer.get('font_family', 'N/A')}, 字间距: {layer.get('spacing', 0)}")
            print(f"    是否在商品区域: {layer.get('is_in_product', False)}")
        
        response_data = {
            'success': True,
            'translation': ai_response,
            'ocr_results': results,
            'replaced_image': replaced_image_base64,
            'text_layers': text_layers,
            'processed_image': inpainted_image_base64
        }
        
        print(f"🔍 最终返回数据键: {list(response_data.keys())}")
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'混合OCR处理失败: {str(e)}'}), 500

@app.route('/api/ocr_replace', methods=['POST'])
@cross_origin()
def ocr_replace():
    """OCR识别并替换图像中的文字（商品部分翻译，背景翻译并替换）"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '无效的请求数据'}), 400
        
        image_base64 = data.get('image_base64')
        target_language = data.get('target_language', 'Chinese')
        source_language = data.get('source_language', '')
        mask_data = data.get('mask_data', None)
        
        if not image_base64:
            return jsonify({'error': '缺少图像数据'}), 400
        
        if not mask_data:
            return jsonify({'error': '缺少掩码数据'}), 400
        
        # 确定OCR语言参数
        lang = None  # 默认不设置，让PaddleOCR自动检测中日英
        if source_language:
            lang_map = {
                'Korean': 'korean',
                'Russian': 'ru',
                'Japanese': 'japan',
                'English': 'en',
                'Chinese': 'ch',
                'French': 'fr'
            }
            lang = lang_map.get(source_language, None)
        
        print(f"🔍 OCR替换调试信息:")
        print(f"  用户选择的原语言: {source_language}")
        print(f"  映射后的OCR语言参数: {lang}")
        
        # 调用混合OCR处理函数
        return handle_mixed_ocr_replace(image_base64, target_language, lang, mask_data)
        
    except Exception as e:
        return jsonify({'error': f'OCR替换失败: {str(e)}'}), 500

@app.route('/api/translate', methods=['POST'])
@cross_origin()
def translate_image():
    """翻译图像中的文字"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '无效的请求数据'}), 400
        
        image_base64 = data.get('image_base64')
        target_language = data.get('target_language', 'Chinese')
        
        if not image_base64:
            return jsonify({'error': '缺少图像数据'}), 400
        
        # 调用图像翻译函数
        result = translate_image_with_gpt(image_base64, target_language)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        # 如果有翻译结果，尝试进行OCR替换
        response_data = {
            'success': True,
            'translation': result.get('translation', '')
        }
        
        # 如果翻译成功，尝试进行OCR替换
        if result.get('translation'):
            try:
                # 解码图像
                if ',' in image_base64:
                    image_base64 = image_base64.split(',')[1]
                
                # 保存临时图像文件
                image_bytes = base64.b64decode(image_base64)
                temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_image.jpg')
                with open(temp_image_path, 'wb') as f:
                    f.write(image_bytes)
                
                # 进行OCR检测
                ocr_result = perform_ocr_and_save_json(temp_image_path, lang='en')
                
                if ocr_result and ocr_result.get('json_path'):
                    # 使用OCR模块进行文字替换
                    from ocr_text_replacement import process_ocr_with_translation
                    
                    # 生成输出路径
                    output_path = os.path.join(tempfile.gettempdir(), 'translated_ocr.jpg')
                    
                    # 处理OCR替换
                    text_layers = process_ocr_with_translation(
                        temp_image_path, 
                        ocr_result['json_path'], 
                        output_path, 
                        translation_text=result.get('translation', '')
                    )
                    
                    # 读取处理后的图像
                    if os.path.exists(output_path):
                        with open(output_path, 'rb') as f:
                            processed_image_bytes = f.read()
                            processed_image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')
                            response_data['replaced_image'] = processed_image_base64
                            response_data['processed_image'] = processed_image_base64
                    
                    # 添加文字图层信息
                    if text_layers:
                        response_data['text_layers'] = text_layers
                
                # 清理临时文件
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
                    
            except Exception as e:
                print(f"OCR替换处理失败: {e}")
                # 即使OCR替换失败，也返回翻译结果
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'翻译失败: {str(e)}'}), 500

@app.route('/api/download_mask', methods=['POST'])
@cross_origin()
def download_mask():
    """下载掩码图像"""
    try:
        data = request.get_json()
        if not data or 'mask_data' not in data:
            return jsonify({'error': '缺少掩码数据'}), 400
        
        mask = np.array(data['mask_data'], dtype=bool) * 255
        mask_img = Image.fromarray(mask.astype(np.uint8), mode='L')
        
        output_buffer = io.BytesIO()
        mask_img.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        return send_file(
            output_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='mask.png'
        )
    except Exception as e:
        return jsonify({'error': f'下载掩码失败: {str(e)}'}), 500

@app.route('/api/download_masked_image', methods=['POST'])
@cross_origin()
def download_masked_image():
    """下载透明背景的掩码图像"""
    try:
        data = request.get_json()
        if not data or 'mask_data' not in data or 'image' not in data:
            return jsonify({'error': '缺少掩码数据或图像数据'}), 400
        
        # 解码图像
        image_base64 = data['image']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 确保图像是RGBA格式
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 获取掩码
        mask = np.array(data['mask_data'], dtype=bool)
        
        # 创建透明背景的掩码结果
        image_array = np.array(image)
        mask_3d = np.stack([mask] * 4, axis=-1)
        masked_result = image_array * mask_3d
        masked_result_pil = Image.fromarray(masked_result.astype(np.uint8))
        
        output_buffer = io.BytesIO()
        masked_result_pil.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        return send_file(
            output_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='masked_image.png'
        )
    except Exception as e:
        return jsonify({'error': f'下载掩码图像失败: {str(e)}'}), 500

@app.route('/api/download_inverted_mask', methods=['POST'])
@cross_origin()
def download_inverted_mask():
    """下载反转掩码图像"""
    try:
        data = request.get_json()
        if not data or 'mask_data' not in data:
            return jsonify({'error': '缺少掩码数据'}), 400
        
        mask = np.array(data['mask_data'], dtype=bool)
        inverted_mask = ~mask
        inverted_mask_img = Image.fromarray((inverted_mask * 255).astype(np.uint8), mode='L')
        
        output_buffer = io.BytesIO()
        inverted_mask_img.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        return send_file(
            output_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='inverted_mask.png'
        )
    except Exception as e:
        return jsonify({'error': f'下载反转掩码失败: {str(e)}'}), 500

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/mobile-test')
def mobile_test():
    """移动端坐标测试页面"""
    return render_template('mobile_test.html')

@app.route('/coordinate-test')
def coordinate_test():
    """坐标系统测试页面"""
    return render_template('coordinate_test.html')

@app.route('/info')
def service_info():
    """服务信息页面"""
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        info_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAM2 OCR翻译系统 - 服务信息</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; text-align: center; }}
                .info-box {{ background: #e8f4fd; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .url-box {{ background: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .status {{ color: #28a745; font-weight: bold; }}
                .warning {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
                .api-endpoint {{ background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 SAM2 OCR翻译系统</h1>
                
                <div class="info-box">
                    <h2>📊 服务状态</h2>
                    <p><span class="status">✅ 服务运行正常</span></p>
                    <p><strong>本机IP地址:</strong> {local_ip}</p>
                    <p><strong>端口:</strong> 5000</p>
                </div>
                
                <div class="info-box">
                    <h2>🌐 访问地址</h2>
                    <div class="url-box">
                        <strong>本地访问:</strong> <a href="http://localhost:5000">http://localhost:5000</a>
                    </div>
                    <div class="url-box">
                        <strong>网络访问:</strong> <a href="http://{local_ip}:5000">http://{local_ip}:5000</a>
                    </div>
                    <p class="warning">⚠️ 其他设备可以通过网络访问地址连接到此服务</p>
                </div>
                
                <div class="info-box">
                    <h2>🔧 API端点</h2>
                    <div class="api-endpoint">GET /api/health - 健康检查</div>
                    <div class="api-endpoint">POST /api/segment - 图像分割</div>
                    <div class="api-endpoint">POST /api/ocr_translate - OCR翻译</div>
                    <div class="api-endpoint">POST /api/ocr_replace - OCR替换</div>
                    <div class="api-endpoint">POST /api/translate - 图像翻译</div>
                </div>
                
                <div class="info-box">
                    <h2>📱 移动端测试</h2>
                    <div class="url-box">
                        <strong>坐标测试:</strong> <a href="/mobile-test">/mobile-test</a>
                    </div>
                    <p class="warning">⚠️ 用于测试移动端触摸坐标处理</p>
                </div>
                
                <div class="info-box">
                    <h2>🎯 坐标系统测试</h2>
                    <div class="url-box">
                        <strong>坐标一致性测试:</strong> <a href="/coordinate-test">/coordinate-test</a>
                    </div>
                    <p class="warning">⚠️ 用于测试交互编辑和最终生成的坐标一致性</p>
                </div>
                
                <div class="info-box">
                    <h2>💡 使用说明</h2>
                    <ul>
                        <li>确保防火墙允许5000端口访问</li>
                        <li>在同一网络下的其他设备可以通过本机IP访问</li>
                        <li>如需外网访问，请配置端口转发</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        return info_html
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAM2 OCR翻译系统 - 服务信息</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; text-align: center; }}
                .error {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 SAM2 OCR翻译系统</h1>
                <p class="error">❌ 无法获取本机IP地址: {str(e)}</p>
                <p>请手动查看本机IP地址后访问: http://[您的IP]:5000</p>
            </div>
        </body>
        </html>
        """
        return error_html

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return jsonify({
            'status': 'healthy',
            'service': 'SAM2 OCR翻译系统',
            'local_ip': local_ip,
            'access_urls': [
                f'http://localhost:5000',
                f'http://{local_ip}:5000'
            ],
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'healthy',
            'service': 'SAM2 OCR翻译系统',
            'error': f'无法获取IP地址: {str(e)}',
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })

@app.route('/api/test_ocr', methods=['POST'])
@cross_origin()
def test_ocr():
    """测试OCR功能"""
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({'error': '缺少图像数据'}), 400
        
        image_base64 = data['image_base64']
        lang = data.get('lang', None)
        
        # 解码图像
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 保存临时图像文件
        temp_image_path = "test_ocr_image.jpg"
        image.save(temp_image_path)
        
        print("🧪 开始测试OCR功能...")
        
        # 执行OCR并保存JSON
        json_path, results = perform_ocr_and_save_json(temp_image_path, lang=lang)
        
        # 清理临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        if json_path and os.path.exists(json_path):
            # 读取JSON文件内容
            with open(json_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
            os.remove(json_path)
        else:
            json_content = {}
        
        return jsonify({
            'success': True,
            'results_count': len(results),
            'results': results,
            'json_content': json_content
        })
        
    except Exception as e:
        import traceback
        print(f"❌ 测试OCR失败: {str(e)}")
        print(f"错误详情: {traceback.format_exc()}")
        return jsonify({'error': f'测试OCR失败: {str(e)}'}), 500

# OCR文字替换相关API端点
@app.route('/api/ocr/detect', methods=['POST'])
@cross_origin()
def ocr_detect():
    """检测图片中的文字"""
    try:
        data = request.json
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'error': '没有提供图片数据'}), 400
        
        # 解码图片
        image_data = base64.b64decode(image_base64.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        if ocr_replacer is None:
            return jsonify({'error': 'OCR服务未初始化'}), 500
        
        # 检测文字
        text_regions = ocr_replacer.detect_text(image)
        
        return jsonify({
            'success': True,
            'text_regions': text_regions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ocr/process', methods=['POST'])
@cross_origin()
def ocr_process():
    """处理图片：移除原文字，准备文字图层"""
    try:
        data = request.json
        image_base64 = data.get('image')
        translation_mapping = data.get('translation_mapping', {})
        
        if not image_base64:
            return jsonify({'error': '没有提供图片数据'}), 400
        
        # 解码图片
        image_data = base64.b64decode(image_base64.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        if ocr_replacer is None:
            return jsonify({'error': 'OCR服务未初始化'}), 500
        
        # 处理图片
        processed_image, text_layers = ocr_replacer.process_image(image, translation_mapping)
        
        # 转换处理后的图片为base64
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'processed_image': f'data:image/png;base64,{img_str}',
            'text_layers': text_layers
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ocr/generate', methods=['POST'])
@cross_origin()
def ocr_generate():
    """生成最终图片：应用文字图层"""
    try:
        data = request.json
        image_base64 = data.get('image')
        text_layers = data.get('text_layers', [])
        
        if not image_base64:
            return jsonify({'error': '没有提供图片数据'}), 400
        
        # 解码图片
        image_data = base64.b64decode(image_base64.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # 保存原始图片尺寸
        original_width, original_height = image.size
        
        # 转换为OpenCV格式
        import cv2
        import numpy as np
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 应用文字图层
        for layer in text_layers:
            if not layer.get('visible', True):
                continue
            
            box = layer['box']
            text = layer['translated_text']
            text_color = layer['text_color']
            font_size = layer['font_size']
            
            print(f"🔍 处理图层: {layer.get('id', 'unknown')}")
            print(f"  原始坐标: {box}")
            print(f"  文字: '{text}'")
            print(f"  字体大小: {font_size}")
            print(f"  文字颜色: {text_color}")
            
            # 处理边界框格式
            if isinstance(box[0], list):
                x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])
                print(f"  转换后坐标: [{x1}, {y1}, {x2}, {y2}]")
            else:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                print(f"  使用坐标: [{x1}, {y1}, {x2}, {y2}]")
            
            # 使用add_translated_text_enhanced函数添加文字，传递所有文字属性
            from ocr_text_replacement import add_translated_text_enhanced
            font_family = layer.get('font_family', 'Microsoft YaHei')
            spacing = layer.get('spacing', 0)
            image_cv = add_translated_text_enhanced(
                image_cv, 
                [x1, y1, x2, y2], 
                text, 
                text_color=text_color, 
                font_size=font_size,
                font_family=font_family,
                spacing=spacing
            )
        
        # 转换回PIL格式
        final_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        
        # 确保最终图片尺寸与原始图片一致
        if final_image.size != (original_width, original_height):
            final_image = final_image.resize((original_width, original_height), Image.Resampling.LANCZOS)
        
        # 确保图片尺寸正确
        print(f"🔍 最终结果图片尺寸检查:")
        print(f"  原始图片尺寸: {original_width}x{original_height}")
        print(f"  最终图片尺寸: {final_image.size}")
        print(f"  尺寸是否一致: {final_image.size == (original_width, original_height)}")
        
        # 转换最终图片为base64
        buffered = io.BytesIO()
        final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'final_image': f'data:image/png;base64,{img_str}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ocr/translate_text', methods=['POST'])
@cross_origin()
def ocr_translate_text():
    """翻译文字（简单示例）"""
    try:
        data = request.json
        text = data.get('text', '')
        
        # 这里可以集成真实的翻译API
        # 现在只是简单的示例
        translated_text = f"[翻译] {text}"
        
        return jsonify({
            'success': True,
            'translated_text': translated_text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 启动SAM2 OCR翻译系统...")
    print("🌐 服务将在以下地址启动:")
    print("   本地访问: http://localhost:5000")
    print("   网络访问: http://0.0.0.0:5000")
    print("   本机IP访问: http://[您的本机IP]:5000")
    print("📱 其他设备可以通过本机IP地址访问此服务")
    print("=" * 50)
    
    # 获取本机IP地址
    import socket
    try:
        # 获取本机IP地址
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"🖥️  本机IP地址: {local_ip}")
        print(f"🌐 其他设备访问地址: http://{local_ip}:5000")
    except Exception as e:
        print(f"⚠️  无法获取本机IP地址: {e}")
        print("💡 您可以通过以下命令查看本机IP:")
        print("   Windows: ipconfig")
        print("   Linux/Mac: ifconfig 或 ip addr")
    
    print("=" * 50)
    print("🚀 启动服务...")
    
    # 启动Flask应用，允许外部访问
    app.run(
        debug=True,           # 开发模式
        host='0.0.0.0',      # 允许所有IP访问
        port=5000,           # 端口号
        threaded=True        # 启用多线程
    ) 