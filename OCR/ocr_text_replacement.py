import json
import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
from PIL import Image, ImageDraw, ImageFont
import requests
import re

def load_ocr_result(json_path):
    """加载OCR结果JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_merged_translations(translation_text):
    """解析翻译文本中的合并信息"""
    merged_regions = {}  # 存储合并区域信息
    merged_blocks = set()  # 存储被合并的块索引
    
    # 解析翻译文本中的合并信息
    lines = translation_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('['):
            continue
            
        # 匹配 [x-y] 格式的合并区域
        import re
        match = re.match(r'\[(\d+)-(\d+)\]\s*(.+?)\s*->\s*(.+)', line)
        if match:
            start_block = int(match.group(1))
            end_block = int(match.group(2))
            original_text = match.group(3).strip()
            translated_text = match.group(4).strip()
            
            # 记录合并区域
            merged_regions[start_block] = {
                'start_block': start_block,
                'end_block': end_block,
                'translation': translated_text
            }
            
            # 记录被合并的块
            for i in range(start_block, end_block + 1):
                merged_blocks.add(i)
    
    print(f"🔍 解析合并翻译信息:")
    print(f"  原始翻译文本行数: {len(lines)}")
    print(f"  合并区域数量: {len(merged_regions)}")
    print(f"  被合并的块: {sorted(merged_blocks)}")
    
    return merged_regions, merged_blocks

def merge_boxes(boxes):
    """合并多个边界框为一个大的边界框"""
    if not boxes:
        return None
    
    print(f"🔍 合并边界框调试信息:")
    print(f"  边界框数量: {len(boxes)}")
    for i, box in enumerate(boxes):
        print(f"  边界框 {i}: {box} (类型: {type(box)})")
    
    # 检查边界框格式并统一处理
    processed_boxes = []
    for box in boxes:
        if isinstance(box, list) and len(box) == 4:
            # 如果是 [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] 格式
            if isinstance(box[0], list):
                processed_boxes.append(box)
            # 如果是 [x1, y1, x2, y2] 格式
            elif len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
                x1, y1, x2, y2 = box
                processed_boxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            else:
                print(f"  警告: 未知的边界框格式: {box}")
                continue
        else:
            print(f"  警告: 无效的边界框格式: {box}")
            continue
    
    if not processed_boxes:
        print("❌ 没有有效的边界框可以合并")
        return None
    
    # 找到所有边界框的最小和最大坐标
    min_x = min(box[0][0] for box in processed_boxes)
    min_y = min(box[0][1] for box in processed_boxes)
    max_x = max(box[2][0] for box in processed_boxes)
    max_y = max(box[2][1] for box in processed_boxes)
    
    # 返回 [x1, y1, x2, y2] 格式，与前端期望一致
    merged_box = [min_x, min_y, max_x, max_y]
    print(f"  合并后边界框: {merged_box}")
    
    return merged_box

def get_translation_by_index_simple(index, translation_text):
    """根据索引获取翻译文本（简化版）"""
    if translation_text is None:
        return f"[翻译{index}]"
    
    # 解析翻译文本，查找对应索引的翻译
    lines = translation_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('['):
            continue
            
        # 匹配 [x] 或 [x-y] 格式
        import re
        match = re.match(r'\[(\d+)(?:-(\d+))?\]\s*(.+?)\s*->\s*(.+)', line)
        if match:
            start_block = int(match.group(1))
            end_block = int(match.group(2)) if match.group(2) else start_block
            translated_text = match.group(4).strip()
            
            # 如果索引在范围内，返回翻译
            if start_block <= index <= end_block:
                return translated_text
    
    # 如果没找到翻译，返回默认文本
    return f"[翻译{index}]"

def translate_text(text, target_lang='zh', translation_text=None):
    """简单的翻译函数（这里使用模拟翻译，实际使用时可以接入翻译API）"""
    if translation_text is None:
        # 这里是一个简单的翻译映射，实际使用时可以替换为真实的翻译API
        translations = '''
        [1] Усиленнаяверсия -> 加强版  
        [2] Зкстракт трав -> 草本提取物  
        [3] Без онемения -> 无麻木感  
        [4] Продлевает + питает -> 延长+滋养  
        [5-6] Безопасно,не вывываетпривыкания -> 安全，不会产生依赖  
        [7] Цена -> 价格  
        [8-9] CO скидкой -> 有折扣  
        [10] 598 -> 598  
        [11] Быстрый -> 快速  
        [12-16] зффект: продление более 30 минут -> 效果：延长超过30分钟  
        [17-19] Секрет мужской ВЫНОСЛИВОСТИ -> 男性耐力的秘密  
        [20-21] Профессиональное средство -> 专业产品
        '''
    else:
        translations = translation_text
    
    # 解析翻译文本，查找对应的翻译
    lines = translations.strip().split('\n')
    print(f"🔍 查找翻译: '{text}'")
    print(f"  翻译文本行数: {len(lines)}")
    
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('['):
            continue
            
        # 匹配 [x] 或 [x-y] 格式
        import re
        match = re.match(r'\[(\d+)(?:-(\d+))?\]\s*(.+?)\s*->\s*(.+)', line)
        if match:
            start_block = int(match.group(1))
            end_block = int(match.group(2)) if match.group(2) else start_block
            original_text = match.group(3).strip()
            translated_text = match.group(4).strip()
            
            print(f"    解析行: '{line}'")
            print(f"      原文: '{original_text}'")
            print(f"      翻译: '{translated_text}'")
            
            # 如果原文匹配，返回翻译
            if original_text.strip() == text.strip():
                print(f"      匹配成功！")
                return translated_text
    
    print(f"  未找到匹配的翻译")
    # 如果没找到翻译，返回原文
    return text

def create_mask_for_text_removal(image, box):
    """为文字区域创建掩码"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 统一边界框格式
    if isinstance(box, list) and len(box) == 4:
        if isinstance(box[0], list):
            # 如果是 [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] 格式，转换为 [x1,y1,x2,y2]
            x1, y1 = box[0][0], box[0][1]
            x2, y2 = box[2][0], box[2][1]
        elif len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
            # 已经是 [x1, y1, x2, y2] 格式
            x1, y1, x2, y2 = box
        else:
            print(f"警告: 无效的边界框格式: {box}")
            return mask
    else:
        print(f"警告: 无效的边界框格式: {box}")
        return mask
    
    # 确保坐标在图像范围内
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(image.shape[1], int(x2))
    y2 = min(image.shape[0], int(y2))
    
    # 在掩码上填充文字区域
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    return mask

def inpaint_text_area(image, box, inpaint_radius=3):
    """使用inpaint方法移除文字区域"""
    mask = create_mask_for_text_removal(image, box)
    
    # 使用inpaint方法填充文字区域
    inpainted = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
    
    return inpainted

def get_dominant_color(image, box):
    """获取框内主要背景颜色"""
    # 统一边界框格式
    if isinstance(box, list) and len(box) == 4:
        if isinstance(box[0], list):
            # 如果是 [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] 格式，转换为 [x1,y1,x2,y2]
            x1, y1 = box[0][0], box[0][1]
            x2, y2 = box[2][0], box[2][1]
        elif len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
            # 已经是 [x1, y1, x2, y2] 格式
            x1, y1, x2, y2 = box
        else:
            print(f"警告: 无效的边界框格式: {box}")
            return (128, 128, 128)  # 默认灰色
    else:
        print(f"警告: 无效的边界框格式: {box}")
        return (128, 128, 128)  # 默认灰色
    
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
    
    # 提取框内区域
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return (128, 128, 128)  # 默认灰色
    
    # 计算平均颜色
    avg_color = np.mean(roi, axis=(0, 1))
    return tuple(avg_color.astype(int))

def contrast_bw(bg_rgb):
    """返回 0 表示用黑字，1 表示用白字"""
    r, g, b = [v / 255 for v in bg_rgb]

    # 将 sRGB 转为线性值
    def lin(c):  # IEC 61966-2-1
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin, g_lin, b_lin = map(lin, (r, g, b))

    # 计算相对亮度 (relative luminance)
    L = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    # 简化阈值：0.179 ≈ (0.05^2.4)；高于此说明底色偏亮
    return 0 if L > 0.179 else 1

def get_contrast_color(bg_color):
    """根据背景颜色计算对比度高的文字颜色"""
    # 使用新的对比度算法
    contrast_result = contrast_bw(bg_color)
    
    if contrast_result == 0:
        # 返回黑色文字
        return (0, 0, 0)
    else:
        # 返回白色文字
        return (255, 255, 255)

def calculate_font_size_and_spacing(text, box, max_font_size=80, min_font_size=8):
    """根据框的大小计算合适的字体大小和字间距，支持非固定比例"""
    # 统一边界框格式
    if isinstance(box, list) and len(box) == 4:
        if isinstance(box[0], list):
            # 如果是 [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] 格式，转换为 [x1,y1,x2,y2]
            x1, y1 = box[0][0], box[0][1]
            x2, y2 = box[2][0], box[2][1]
        elif len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
            # 已经是 [x1, y1, x2, y2] 格式
            x1, y1, x2, y2 = box
        else:
            print(f"警告: 无效的边界框格式: {box}")
            return min_font_size, 0, min_font_size
    else:
        print(f"警告: 无效的边界框格式: {box}")
        return min_font_size, 0, min_font_size
    
    box_width = x2 - x1
    box_height = y2 - y1
    
    # 设置目标宽度为框宽度的80%
    target_width = box_width * 0.8
    target_height = box_height * 0.8
    
    # 分别计算基于宽度和高度的字体大小
    if len(text) > 0:
        # 基于宽度的字体大小
        width_based_size = target_width / len(text) * 0.8
        # 基于高度的字体大小
        height_based_size = target_height * 0.8
        # 选择较小的值，确保文字完全适应框
        font_size = min(width_based_size, height_based_size)
    else:
        font_size = min(target_width * 0.8, target_height * 0.8)
    
    font_size = max(min_font_size, min(max_font_size, font_size))
    
    # 尝试加载字体来计算实际文字宽度和高度
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", int(font_size))
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", int(font_size))
        except:
            font = ImageFont.load_default()
    
    # 计算文字实际宽度和高度
    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # 如果文字太宽或太高，调整字体大小
    width_scale = target_width / text_width if text_width > 0 else 1
    height_scale = target_height / text_height if text_height > 0 else 1
    
    # 根据框的形状选择缩放策略
    aspect_ratio = box_width / box_height if box_height > 0 else 1
    
    if aspect_ratio > 2:  # 很宽的框，优先考虑宽度
        scale_factor = min(width_scale, height_scale * 1.2) * 0.95
    elif aspect_ratio < 0.5:  # 很高的框，优先考虑高度
        scale_factor = min(width_scale * 1.2, height_scale) * 0.95
    else:  # 正常比例的框，平衡考虑
        scale_factor = min(width_scale, height_scale) * 0.95
    
    if scale_factor < 1:
        font_size = int(font_size * scale_factor)
        font_size = max(min_font_size, font_size)
        
        # 重新计算文字尺寸
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
        except:
            font = ImageFont.load_default()
        text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    
    # 计算字间距策略
    if len(text) > 1:
        # 检查文字宽度是否远小于目标宽度
        if text_width < target_width * 0.6:  # 如果文字宽度小于目标宽度的60%
            # 使用正常字间距，居中显示
            spacing = 0  # 正常字间距
        else:
            # 计算字间距，使文字总宽度等于目标宽度
            spacing = (target_width - text_width) / (len(text) - 1) if len(text) > 1 else 0
            spacing = max(0, spacing)  # 确保间距不为负数
    else:
        spacing = 0
    
    return int(font_size), spacing, text_height

def calculate_optimized_box_for_text(text, original_box, font_size, spacing=0):
    """根据翻译后的文字计算优化的边框，使其更紧密贴合文字（支持多行文本）"""
    # 统一边界框格式
    if isinstance(original_box, list) and len(original_box) == 4:
        if isinstance(original_box[0], list):
            x1, y1 = original_box[0][0], original_box[0][1]
            x2, y2 = original_box[2][0], original_box[2][1]
        elif len(original_box) == 4 and all(isinstance(x, (int, float)) for x in original_box):
            x1, y1, x2, y2 = original_box
        else:
            return original_box
    else:
        return original_box
    
    # 计算原始框的中心点
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # 尝试加载字体来计算实际文字宽度和高度
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", int(font_size))
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", int(font_size))
        except:
            font = ImageFont.load_default()
    
    # 计算文字实际宽度和高度（支持多行文本）
    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    
    # 检查是否包含换行符
    if '\n' in text:
        # 多行文本处理
        lines = text.split('\n')
        line_height = font_size * 1.2  # 行高为字体大小的1.2倍
        
        # 计算每行的宽度
        line_widths = []
        for line in lines:
            if line.strip():  # 跳过空行
                text_bbox = dummy_draw.textbbox((0, 0), line, font=font)
                line_width = text_bbox[2] - text_bbox[0]
                # 计算字间距
                total_spacing = spacing * (len(line) - 1) if len(line) > 1 else 0
                line_widths.append(line_width + total_spacing)
            else:
                line_widths.append(0)
        
        # 找到最大宽度
        max_line_width = max(line_widths) if line_widths else 0
        # 计算总高度
        total_height = len([line for line in lines if line.strip()]) * line_height
        
        text_width = max_line_width
        text_height = total_height
    else:
        # 单行文本处理
        text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 计算字间距的总宽度
        total_spacing = spacing * (len(text) - 1) if len(text) > 1 else 0
        text_width += total_spacing
    
    # 计算新的边框大小，添加一些内边距
    padding = max(10, font_size * 0.3)  # 内边距为字体大小的30%，最小10像素
    new_width = text_width + padding * 2
    new_height = text_height + padding * 2
    
    # 计算新的边框坐标，保持中心点不变
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2
    
    # 确保新边框不会超出原始边框太多
    max_expansion = min(original_box[2] - original_box[0], original_box[3] - original_box[1]) * 0.5
    current_expansion_x = (new_width - (x2 - x1)) / 2
    current_expansion_y = (new_height - (y2 - y1)) / 2
    
    if current_expansion_x > max_expansion:
        # 如果扩展太多，保持原始宽度，只调整高度
        new_x1 = x1
        new_x2 = x2
        center_x = (x1 + x2) / 2
        new_y1 = center_y - new_height / 2
        new_y2 = center_y + new_height / 2
    
    if current_expansion_y > max_expansion:
        # 如果扩展太多，保持原始高度，只调整宽度
        new_y1 = y1
        new_y2 = y2
        center_y = (y1 + y2) / 2
        new_x1 = center_x - new_width / 2
        new_x2 = center_x + new_width / 2
    
    return [new_x1, new_y1, new_x2, new_y2]

def add_translated_text_enhanced(image, box, translated_text, text_color=None, font_size=None, font_family=None, spacing=None):
    """在指定位置添加增强的翻译文字"""
    # 转换为PIL图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 打印调试信息
    print(f"🔍 add_translated_text_enhanced 调试信息:")
    print(f"  输入坐标: {box}")
    print(f"  文字: '{translated_text}'")
    print(f"  图片尺寸: {pil_image.size}")
    
    # 统一边界框格式
    if isinstance(box, list) and len(box) == 4:
        if isinstance(box[0], list):
            # 如果是 [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] 格式，转换为 [x1,y1,x2,y2]
            x1, y1 = box[0][0], box[0][1]
            x2, y2 = box[2][0], box[2][1]
            box = [x1, y1, x2, y2]
            print(f"  转换后坐标: [{x1}, {y1}, {x2}, {y2}]")
        elif len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
            # 已经是 [x1, y1, x2, y2] 格式
            print(f"  使用原始坐标: {box}")
            pass
        else:
            print(f"警告: 无效的边界框格式: {box}")
            return image
    
    # 如果没有提供颜色，则自动获取背景颜色
    if text_color is None:
        bg_color = get_dominant_color(image, box)
        text_color = get_contrast_color(bg_color)
        contrast_result = contrast_bw(bg_color)
        print(f"  背景颜色: RGB{bg_color}")
        print(f"  对比度算法结果: {contrast_result} ({'黑字' if contrast_result == 0 else '白字'})")
        print(f"  选择的文字颜色: RGB{text_color}")
    else:
        # 如果提供了颜色，直接使用
        text_color = tuple(text_color) if isinstance(text_color, list) else text_color
        print(f"  使用提供的文字颜色: RGB{text_color}")
    
    # 如果没有提供字体大小，则自动计算
    if font_size is None:
        font_size, spacing, text_height = calculate_font_size_and_spacing(translated_text, box)
    else:
        # 使用提供的字体大小
        font_size = int(font_size)
        if spacing is None:
            spacing = 0  # 使用提供的字体大小时，默认不计算字间距
        text_height = font_size  # 简化估算
    
    # 重新计算文字宽度用于调试信息
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
        except:
            font = ImageFont.load_default()
    
    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    text_bbox = dummy_draw.textbbox((0, 0), translated_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    
    # 打印调试信息
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    target_width = box_width * 0.8
    target_height = box_height * 0.8
    aspect_ratio = box_width / box_height if box_height > 0 else 1
    print(f"  框尺寸: {box_width}x{box_height} (宽高比: {aspect_ratio:.2f})")
    print(f"  目标尺寸: {target_width:.1f}x{target_height:.1f}")
    print(f"  字体大小: {font_size}, 字间距: {spacing:.1f}, 文字高度: {text_height:.1f}")
    if len(translated_text) > 1:
        if text_width < target_width * 0.6:
            print(f"  字间距策略: 正常字间距 (文字宽度 {text_width:.1f} < 目标宽度 {target_width:.1f} 的60%)")
        else:
            print(f"  字间距策略: 扩展字间距以占满80%宽度")
    print(f"  文字: '{translated_text}', 字符数: {len(translated_text)}")
    
    # 根据框的形状选择合适的字体
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    aspect_ratio = box_width / box_height if box_height > 0 else 1
    
    # 尝试加载字体
    try:
        if font_family:
            # 如果提供了字体类型，尝试加载
            if font_family == 'Microsoft YaHei':
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
            elif font_family == 'SimHei':
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
            else:
                font = ImageFont.truetype(font_family, font_size)
        else:
            # 根据框的形状选择合适的字体
            if aspect_ratio > 3:  # 很宽的框，使用较扁的字体
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
            elif aspect_ratio < 0.5:  # 很高的框，使用较方的字体
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
            else:  # 正常比例的框
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
        except:
            font = ImageFont.load_default()
    
    # 计算文字位置（居中）
    x1, y1, x2, y2 = box
    dummy_draw = ImageDraw.Draw(pil_image)
    
    # 计算居中位置
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2 - text_height // 2
    
    # 确保文字不超出边界
    center_y = max(y1, min(center_y, y2 - text_height))
    
    # 创建透明图层用于绘制文字
    text_layer = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    
    # 绘制文字（支持字间距）
    if len(translated_text) > 1 and spacing > 0:
        # 计算总宽度（包括字间距）
        total_width = len(translated_text) * font_size + (len(translated_text) - 1) * spacing
        # 计算起始位置，使文字在框内居中
        start_x = center_x - total_width // 2
        
        for i, char in enumerate(translated_text):
            char_x = start_x + i * (font_size + spacing)
            draw.text((char_x, center_y), char, fill=(*text_color, 255), font=font)
    else:
        # 单个字符或正常字间距的情况
        # 重新计算文字宽度用于居中
        dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        text_bbox = dummy_draw.textbbox((0, 0), translated_text, font=font)
        actual_text_width = text_bbox[2] - text_bbox[0]
        
        # 计算居中位置
        start_x = center_x - actual_text_width // 2
        draw.text((start_x, center_y), translated_text, fill=(*text_color, 255), font=font)
    
    # 将文字图层合并到原图
    pil_image = Image.alpha_composite(pil_image.convert('RGBA'), text_layer)
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)

def process_ocr_with_translation(image_path, json_path, output_path="output/translated_ocr.jpg", translation_text=None):
    """处理OCR结果，移除原文字并添加翻译"""
    
    # 加载OCR结果
    ocr_result = load_ocr_result(json_path)
    
    # 读取原始图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return
    
    # 获取识别到的文字信息
    rec_texts = ocr_result.get('rec_texts', [])
    rec_scores = ocr_result.get('rec_scores', [])
    rec_boxes = ocr_result.get('rec_boxes', [])
    
    # 过滤掉空文本和置信度很低的文本
    valid_results = []
    for i, (text, score, box) in enumerate(zip(rec_texts, rec_scores, rec_boxes)):
        if text.strip() and score > 0.5:  # 过滤空文本和低置信度
            valid_results.append({
                'index': len(valid_results) + 1,
                'text': text,
                'score': score,
                'box': box
            })
    
    print(f"找到 {len(valid_results)} 个有效文字区域")
    
    # 第一步：先对所有block进行inpaint移除原文字
    print("🔍 第一步：对所有block进行inpaint移除原文字")
    processed_image = image.copy()
    for i, result in enumerate(valid_results):
        block_index = result['index']
        text = result['text']
        box = result['box']
        score = result['score']
        
        print(f"  Inpaint块 {block_index}: '{text}' (置信度: {score:.3f})")
        
        # 使用inpaint移除原文字
        processed_image = inpaint_text_area(processed_image, box)
    
    # 第二步：解析合并翻译信息
    print("\n🔍 第二步：解析合并翻译信息")
    if translation_text is None:
        translation_text = '''
        [1] Усиленнаяверсия -> 加强版  
        [2] Зкстракт трав -> 草本提取物  
        [3] Без онемения -> 无麻木感  
        [4] Продлевает + питает -> 延长+滋养  
        [5-6] Безопасно,не вывываетпривыкания -> 安全，不会产生依赖  
        [7] Цена -> 价格  
        [8-9] CO скидкой -> 有折扣  
        [10] 598 -> 598  
        [11] Быстрый -> 快速  
        [12-16] зффект: продление более 30 минут -> 效果：延长超过30分钟  
        [17-19] Секрет мужской ВЫНОСЛИВОСТИ -> 男性耐力的秘密  
        [20-21] Профессиональное средство -> 专业产品
        '''
    
    merged_regions, merged_blocks = parse_merged_translations(translation_text)
    print(f"解析到 {len(merged_regions)} 个合并区域")
    print(f"被合并的块: {sorted(merged_blocks)}")
    
    # 第三步：添加翻译文字
    print("\n🔍 第三步：添加翻译文字")
    for i, result in enumerate(valid_results):
        block_index = result['index']
        text = result['text']
        box = result['box']
        score = result['score']
        
        # 检查是否是被合并的块（不是合并区域的起始块）
        is_merged_block = False
        for region_start, region_info in merged_regions.items():
            if block_index != region_start and region_info['start_block'] <= block_index <= region_info['end_block']:
                is_merged_block = True
                break
        
        # 如果是被合并的块，跳过添加翻译文字
        if is_merged_block:
            print(f"  跳过被合并的块 {block_index}: '{text}'")
            continue
        
        print(f"  处理第 {block_index} 个文字: '{text}' (置信度: {score:.3f})")
        
        # 检查是否是合并区域的起始块
        if block_index in merged_regions:
            region_info = merged_regions[block_index]
            start_block = region_info['start_block']
            end_block = region_info['end_block']
            translation = region_info['translation']
            
            print(f"    合并区域 [{start_block}-{end_block}]: '{translation}'")
            
            # 收集合并区域的所有边界框
            merged_boxes = []
            for j in range(start_block, end_block + 1):
                if j - 1 < len(valid_results):
                    merged_boxes.append(valid_results[j - 1]['box'])
            
            # 合并边界框
            merged_box = merge_boxes(merged_boxes)
            if merged_box:
                # 在合并区域添加翻译文字
                processed_image = add_translated_text_enhanced(processed_image, merged_box, translation)
        else:
            # 普通块的处理
            translated_text = translate_text(text, translation_text=translation_text)
            print(f"    翻译结果: '{translated_text}'")
            
            # 添加翻译后的文字（增强版）
            processed_image = add_translated_text_enhanced(processed_image, box, translated_text)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:  # 只有当目录不为空时才创建
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果图像
    cv2.imwrite(output_path, processed_image)
    print(f"\n处理完成！结果已保存到: {output_path}")
    
    # 创建文字图层信息（使用第三步处理后的结果）
    text_layers = []
    layer_id = 1
    
    for i, result in enumerate(valid_results):
        block_index = result['index']
        text = result['text']
        box = result['box']
        
        # 检查是否是被合并的块（不是合并区域的起始块）
        is_merged_block = False
        for region_start, region_info in merged_regions.items():
            if block_index != region_start and region_info['start_block'] <= block_index <= region_info['end_block']:
                is_merged_block = True
                break
        
        # 如果是被合并的块，跳过创建文字图层
        if is_merged_block:
            continue
        
        # 检查是否是合并区域的起始块
        if block_index in merged_regions:
            region_info = merged_regions[block_index]
            start_block = region_info['start_block']
            end_block = region_info['end_block']
            translation = region_info['translation']
            
            # 收集合并区域的所有边界框
            merged_boxes = []
            for j in range(start_block, end_block + 1):
                if j - 1 < len(valid_results):
                    box = valid_results[j - 1]['box']
                    # 统一坐标格式为 [x1, y1, x2, y2]
                    if isinstance(box, list) and len(box) == 4:
                        if isinstance(box[0], list):
                            # 如果是 [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] 格式，转换为 [x1,y1,x2,y2]
                            x1, y1 = box[0][0], box[0][1]
                            x2, y2 = box[2][0], box[2][1]
                            unified_box = [x1, y1, x2, y2]
                        elif len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
                            # 已经是 [x1, y1, x2, y2] 格式
                            unified_box = box
                        else:
                            print(f"警告: 无效的边界框格式: {box}")
                            continue
                    else:
                        print(f"警告: 无效的边界框格式: {box}")
                        continue
                    merged_boxes.append(unified_box)
            
            # 合并边界框
            merged_box = merge_boxes(merged_boxes)
            if merged_box:
                # 获取背景色和文字颜色
                bg_color = get_dominant_color(processed_image, merged_box)
                text_color = get_contrast_color(bg_color)
                
                # 计算字体大小
                font_size, spacing, text_height = calculate_font_size_and_spacing(translation, merged_box)
                
                # 计算优化的边框，使其更紧密贴合翻译后的文字
                optimized_box = calculate_optimized_box_for_text(translation, merged_box, font_size, spacing)
                
                # 创建合并区域的文字图层
                text_layer = {
                    'id': f'layer_{layer_id}',
                    'original_text': f"合并区域[{start_block}-{end_block}]",
                    'translated_text': translation,
                    'box': optimized_box,  # 使用优化的边框
                    'text_color': list(text_color),  # 确保是列表格式
                    'font_size': font_size,
                    'font_family': 'Microsoft YaHei',
                    'spacing': spacing,
                    'visible': True,
                    'is_in_product': False,  # 添加商品区域标识
                    'text_height': text_height  # 添加文字高度信息
                }
                text_layers.append(text_layer)
                layer_id += 1
        else:
            # 普通块的处理
            translated_text = get_translation_by_index_simple(block_index, translation_text)
            if not translated_text:
                translated_text = text  # 如果没找到翻译，使用原文
            
            # 统一坐标格式为 [x1, y1, x2, y2]
            if isinstance(box, list) and len(box) == 4:
                if isinstance(box[0], list):
                    # 如果是 [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] 格式，转换为 [x1,y1,x2,y2]
                    x1, y1 = box[0][0], box[0][1]
                    x2, y2 = box[2][0], box[2][1]
                    unified_box = [x1, y1, x2, y2]
                elif len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
                    # 已经是 [x1, y1, x2, y2] 格式
                    unified_box = box
                else:
                    print(f"警告: 无效的边界框格式: {box}")
                    continue
            else:
                print(f"警告: 无效的边界框格式: {box}")
                continue
            
            # 获取背景色和文字颜色
            bg_color = get_dominant_color(processed_image, unified_box)
            text_color = get_contrast_color(bg_color)
            
            # 计算字体大小
            font_size, spacing, text_height = calculate_font_size_and_spacing(translated_text, unified_box)
            
            # 计算优化的边框，使其更紧密贴合翻译后的文字
            optimized_box = calculate_optimized_box_for_text(translated_text, unified_box, font_size, spacing)
            
            # 创建普通块的文字图层
            text_layer = {
                'id': f'layer_{layer_id}',
                'original_text': text,
                'translated_text': translated_text,
                'box': optimized_box,  # 使用优化的边框
                'text_color': list(text_color),  # 确保是列表格式
                'font_size': font_size,
                'font_family': 'Microsoft YaHei',
                'spacing': spacing,
                'visible': True,
                'is_in_product': False,  # 添加商品区域标识
                'text_height': text_height  # 添加文字高度信息
            }
            text_layers.append(text_layer)
            layer_id += 1
    
    # 添加调试信息
    print(f"\n🔍 OCR模块返回的文字图层信息:")
    print(f"  文字图层数量: {len(text_layers)}")
    for i, layer in enumerate(text_layers):
        print(f"  图层 {i+1}:")
        print(f"    ID: {layer['id']}")
        print(f"    原文: {layer['original_text']}")
        print(f"    翻译: {layer['translated_text']}")
        print(f"    边界框: {layer['box']}")
        print(f"    颜色: {layer['text_color']}")
        print(f"    字体大小: {layer['font_size']}")
        print(f"    字体类型: {layer['font_family']}")
        print(f"    字间距: {layer['spacing']}")
        print(f"    文字高度: {layer['text_height']}")
        print(f"    是否在商品区域: {layer['is_in_product']}")
    
    return {
        'valid_results': valid_results,
        'text_layers': text_layers,
        'processed_image_path': output_path
    }

def print_translation_results(results, translation_text=None):
    """打印翻译结果"""
    print("\n" + "="*60)
    print("文字翻译结果:")
    print("="*60)
    
    for result in results:
        index = result['index']
        original_text = result['text']
        translated_text = translate_text(original_text, translation_text=translation_text)
        score = result['score']
        print(f"序号 {index:2d}: '{original_text}' -> '{translated_text}' (置信度: {score:.3f})")
    
    print("="*60)

class OCRTextReplacer:
    """OCR文字替换器类"""
    
    def __init__(self):
        """初始化OCR文字替换器"""
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        print("✅ OCR文字替换器初始化完成")
    
    def contrast_bw(self, bg_rgb):
        """计算文字颜色（黑或白）基于背景色的相对亮度"""
        def lin(c):  # IEC 61966-2-1
            c = c / 255.0
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        
        # 计算相对亮度
        luminance = 0.2126 * lin(bg_rgb[0]) + 0.7152 * lin(bg_rgb[1]) + 0.0722 * lin(bg_rgb[2])
        
        # 根据亮度选择文字颜色
        return [0, 0, 0] if luminance > 0.5 else [255, 255, 255]  # 黑或白
    
    def get_contrast_color(self, bg_color):
        """获取对比色"""
        return self.contrast_bw(bg_color)
    
    def get_dominant_color(self, image, box):
        """获取指定区域的主要颜色"""
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return [128, 128, 128]  # 默认灰色
        
        # 计算平均颜色
        avg_color = np.mean(roi, axis=(0, 1))
        return [int(c) for c in avg_color]
    
    def create_mask_for_text_removal(self, image, box):
        """为文字移除创建掩码"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
        return mask
    
    def inpaint_text_area(self, image, box, inpaint_radius=3):
        """使用图像修复移除文字区域"""
        mask = self.create_mask_for_text_removal(image, box)
        inpainted = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
        return inpainted
    
    def calculate_font_size_and_spacing(self, text, box, max_font_size=80, min_font_size=8):
        """计算字体大小和间距"""
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        
        # 估算字符数（包括空格）
        char_count = len(text)
        if char_count == 0:
            return min_font_size, 0
        
        # 计算合适的字体大小
        font_size = min(box_width // char_count, box_height, max_font_size)
        font_size = max(font_size, min_font_size)
        
        # 计算间距
        spacing = (box_width - char_count * font_size) // (char_count + 1)
        spacing = max(spacing, 0)
        
        return font_size, spacing
    
    def add_text_layer(self, image, box, text, text_color=None, font_size=None):
        """添加文字图层"""
        x1, y1, x2, y2 = box
        
        # 如果没有指定颜色，获取背景色并计算对比色
        if text_color is None:
            bg_color = self.get_dominant_color(image, box)
            text_color = self.get_contrast_color(bg_color)
        
        # 如果没有指定字体大小，计算合适的字体大小
        if font_size is None:
            font_size, _ = self.calculate_font_size_and_spacing(text, box)
        
        # 创建PIL图像用于文字渲染
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # 计算文字位置（居中）
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 绘制文字
        draw.text((center_x - text_width//2, center_y - text_height//2), 
                  text, fill=tuple(text_color), font=font)
        
        # 转换回OpenCV格式
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result_image
    
    def detect_text(self, image):
        """检测图片中的文字"""
        if isinstance(image, Image.Image):
            # 转换为OpenCV格式
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_cv = image
        
        # 使用PaddleOCR检测文字
        result = self.ocr.ocr(image_cv, cls=True)
        
        text_regions = []
        if result and result[0]:
            for line in result[0]:
                box = line[0]  # 边界框坐标
                text = line[1][0]  # 识别的文字
                confidence = line[1][1]  # 置信度
                
                # 转换为 [x1, y1, x2, y2] 格式
                x_coords = [point[0] for point in box]
                y_coords = [point[1] for point in box]
                x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                
                text_regions.append({
                    'box': [x1, y1, x2, y2],
                    'text': text,
                    'confidence': confidence
                })
        
        return text_regions
    
    def process_image(self, image, translation_mapping=None):
        """处理图片：移除原文字，准备文字图层"""
        if isinstance(image, Image.Image):
            # 转换为OpenCV格式
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_cv = image
        
        # 检测文字
        text_regions = self.detect_text(image_cv)
        
        # 处理后的图片（移除原文字）
        processed_image = image_cv.copy()
        text_layers = []
        
        for i, region in enumerate(text_regions):
            box = region['box']
            original_text = region['text']
            
            # 移除原文字
            processed_image = self.inpaint_text_area(processed_image, box)
            
            # 获取背景色
            bg_color = self.get_dominant_color(processed_image, box)
            text_color = self.get_contrast_color(bg_color)
            
            # 翻译文字
            if translation_mapping and original_text in translation_mapping:
                translated_text = translation_mapping[original_text]
            else:
                translated_text = f"[翻译] {original_text}"
            
            # 计算字体大小
            font_size, _ = self.calculate_font_size_and_spacing(translated_text, box)
            
            # 创建文字图层
            text_layer = {
                'id': f'layer_{i}',
                'original_text': original_text,
                'translated_text': translated_text,
                'box': box,
                'text_color': text_color,
                'font_size': font_size,
                'visible': True
            }
            text_layers.append(text_layer)
        
        # 转换回PIL格式
        processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        
        return processed_pil, text_layers
    
    def apply_text_layers(self, image, text_layers):
        """应用文字图层到图片"""
        if isinstance(image, Image.Image):
            # 转换为OpenCV格式
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_cv = image
        
        result_image = image_cv.copy()
        
        for layer in text_layers:
            if not layer['visible']:
                continue
            
            box = layer['box']
            text = layer['translated_text']
            text_color = layer['text_color']
            font_size = layer['font_size']
            
            # 添加文字图层
            result_image = self.add_text_layer(
                result_image, box, text, text_color, font_size
            )
        
        # 转换回PIL格式
        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        return result_pil

def main():
    """主函数"""
    # 文件路径
    image_path = "images/image2.jpg"
    json_path = "ocr_output/output/temp_ocr_image_temp_1753959854_res.json"
    output_path = "output/translated_ocr.jpg"
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：图像文件不存在 {image_path}")
        return
    
    if not os.path.exists(json_path):
        print(f"错误：JSON文件不存在 {json_path}")
        return
    
    # 处理OCR结果并翻译
    print("正在处理OCR结果并进行翻译...")
    results = process_ocr_with_translation(image_path, json_path, output_path)
    
    # 打印翻译结果
    if results:
        # 使用相同的翻译文本
        translation_text = '''
        [1] Усиленнаяверсия -> 加强版  
        [2] Зкстракт трав -> 草本提取物  
        [3] Без онемения -> 无麻木感  
        [4] Продлевает + питает -> 延长+滋养  
        [5-6] Безопасно,не вывываетпривыкания -> 安全，不会产生依赖  
        [7] Цена -> 价格  
        [8-9] CO скидкой -> 有折扣  
        [10] 598 -> 598  
        [11] Быстрый -> 快速  
        [12-16] зффект: продление более 30 минут -> 效果：延长超过30分钟  
        [17-19] Секрет мужской ВЫНОСЛИВОСТИ -> 男性耐力的秘密  
        [20-21] Профессиональное средство -> 专业产品
        '''
        print_translation_results(results, translation_text)
    else:
        print("未找到有效的OCR识别结果")

if __name__ == "__main__":
    main() 