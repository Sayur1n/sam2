import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Optional
import requests
from flask import Flask, request, jsonify
import logging

logger = logging.getLogger(__name__)

class OCRTextReplacer:
    """OCR文字替换器"""
    
    def __init__(self):
        self.ocr = None
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        except ImportError:
            logger.warning("PaddleOCR not available, OCR functionality will be disabled")
    
    def contrast_bw(self, bg_rgb):
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

    def get_contrast_color(self, bg_color):
        """根据背景颜色计算对比度高的文字颜色"""
        contrast_result = self.contrast_bw(bg_color)
        
        if contrast_result == 0:
            return (0, 0, 0)  # 黑色文字
        else:
            return (255, 255, 255)  # 白色文字

    def get_dominant_color(self, image, box):
        """获取框内主要背景颜色"""
        x1, y1, x2, y2 = box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
        
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return (128, 128, 128)
        
        avg_color = np.mean(roi, axis=(0, 1))
        return tuple(avg_color.astype(int))

    def create_mask_for_text_removal(self, image, box):
        """为文字区域创建掩码"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = box
        
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.shape[1], int(x2))
        y2 = min(image.shape[0], int(y2))
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def inpaint_text_area(self, image, box, inpaint_radius=3):
        """使用inpaint方法移除文字区域"""
        mask = self.create_mask_for_text_removal(image, box)
        inpainted = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
        return inpainted

    def calculate_font_size_and_spacing(self, text, box, max_font_size=80, min_font_size=8):
        """根据框的大小计算合适的字体大小和字间距"""
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        
        target_width = box_width * 0.8
        target_height = box_height * 0.8
        
        if len(text) > 0:
            width_based_size = target_width / len(text) * 0.8
            height_based_size = target_height * 0.8
            font_size = min(width_based_size, height_based_size)
        else:
            font_size = min(target_width * 0.8, target_height * 0.8)
        
        font_size = max(min_font_size, min(max_font_size, font_size))
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", int(font_size))
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", int(font_size))
            except:
                font = ImageFont.load_default()
        
        # 计算文字尺寸
        dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 调整字体大小
        width_scale = target_width / text_width if text_width > 0 else 1
        height_scale = target_height / text_height if text_height > 0 else 1
        scale_factor = min(width_scale, height_scale) * 0.95
        
        if scale_factor < 1:
            font_size = int(font_size * scale_factor)
            font_size = max(min_font_size, font_size)
        
        # 计算字间距
        if len(text) > 1:
            if text_width < target_width * 0.6:
                spacing = 0
            else:
                spacing = (target_width - text_width) / (len(text) - 1) if len(text) > 1 else 0
                spacing = max(0, spacing)
        else:
            spacing = 0
        
        return int(font_size), spacing, text_height

    def add_text_layer(self, image, box, text, text_color=None, font_size=None):
        """添加文字图层"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        
        # 获取背景颜色和文字颜色
        bg_color = self.get_dominant_color(image, box)
        if text_color is None:
            text_color = self.get_contrast_color(bg_color)
        
        # 计算字体大小
        if font_size is None:
            font_size, spacing, text_height = self.calculate_font_size_and_spacing(text, box)
        else:
            spacing = 0
            text_height = font_size
        
        # 加载字体
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
            except:
                font = ImageFont.load_default()
        
        # 计算文字位置
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2 - text_height // 2
        center_y = max(y1, min(center_y, y2 - text_height))
        
        # 创建文字图层
        text_layer = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        
        # 绘制文字
        if len(text) > 1 and spacing > 0:
            total_width = len(text) * font_size + (len(text) - 1) * spacing
            start_x = center_x - total_width // 2
            
            for i, char in enumerate(text):
                char_x = start_x + i * (font_size + spacing)
                draw.text((char_x, center_y), char, fill=(*text_color, 255), font=font)
        else:
            dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
            text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
            actual_text_width = text_bbox[2] - text_bbox[0]
            start_x = center_x - actual_text_width // 2
            draw.text((start_x, center_y), text, fill=(*text_color, 255), font=font)
        
        # 合并图层
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), text_layer)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)

    def detect_text(self, image):
        """检测图像中的文字"""
        if self.ocr is None:
            return []
        
        try:
            result = self.ocr.ocr(image, cls=True)
            if result is None or len(result) == 0:
                return []
            
            detected_texts = []
            for line in result[0]:
                if len(line) >= 2:
                    box, (text, confidence) = line
                    if text.strip() and confidence > 0.5:
                        # 转换边界框格式
                        x_coords = [point[0] for point in box]
                        y_coords = [point[1] for point in box]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                        
                        detected_texts.append({
                            'text': text.strip(),
                            'confidence': float(confidence),
                            'box': [x1, y1, x2, y2],
                            'original_box': box
                        })
            
            return detected_texts
        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            return []

    def process_image(self, image_data, translation_mapping=None):
        """处理图像：检测文字、移除原文字、添加翻译"""
        # 解码图像
        if isinstance(image_data, str):
            # Base64编码的图像
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = image_data
        
        if image is None:
            return None, []
        
        # 检测文字
        detected_texts = self.detect_text(image)
        
        # 移除原文字
        processed_image = image.copy()
        for text_info in detected_texts:
            box = text_info['box']
            processed_image = self.inpaint_text_area(processed_image, box)
        
        # 添加翻译文字
        text_layers = []
        for i, text_info in enumerate(detected_texts):
            original_text = text_info['text']
            box = text_info['box']
            
            # 获取翻译
            translated_text = original_text
            if translation_mapping and original_text in translation_mapping:
                translated_text = translation_mapping[original_text]
            
            # 创建文字图层信息
            layer_info = {
                'id': f"layer_{i}",
                'original_text': original_text,
                'translated_text': translated_text,
                'box': box,
                'text_color': self.get_contrast_color(self.get_dominant_color(processed_image, box)),
                'font_size': self.calculate_font_size_and_spacing(translated_text, box)[0],
                'visible': True
            }
            text_layers.append(layer_info)
        
        return processed_image, text_layers

    def apply_text_layers(self, image, text_layers):
        """应用文字图层到图像"""
        result_image = image.copy()
        
        for layer in text_layers:
            if layer.get('visible', True):
                box = layer['box']
                text = layer['translated_text']
                text_color = layer.get('text_color', (0, 0, 0))
                font_size = layer.get('font_size', 20)
                
                result_image = self.add_text_layer(result_image, box, text, text_color, font_size)
        
        return result_image

# 全局OCR实例
ocr_replacer = OCRTextReplacer()

def create_ocr_routes(app: Flask):
    """创建OCR相关的路由"""
    
    @app.route("/api/ocr/detect", methods=["POST"])
    def detect_text():
        """检测图像中的文字"""
        try:
            data = request.get_json()
            image_data = data.get('image')
            
            if not image_data:
                return jsonify({'error': 'No image data provided'}), 400
            
            # 解码图像
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Invalid image data'}), 400
            
            # 检测文字
            detected_texts = ocr_replacer.detect_text(image)
            
            return jsonify({
                'success': True,
                'detected_texts': detected_texts
            })
        
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route("/api/ocr/process", methods=["POST"])
    def process_ocr():
        """处理OCR：移除原文字并准备文字图层"""
        try:
            data = request.get_json()
            image_data = data.get('image')
            translation_mapping = data.get('translation_mapping', {})
            
            if not image_data:
                return jsonify({'error': 'No image data provided'}), 400
            
            # 处理图像
            processed_image, text_layers = ocr_replacer.process_image(image_data, translation_mapping)
            
            if processed_image is None:
                return jsonify({'error': 'Failed to process image'}), 500
            
            # 编码处理后的图像
            _, buffer = cv2.imencode('.jpg', processed_image)
            processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'processed_image': f"data:image/jpeg;base64,{processed_image_b64}",
                'text_layers': text_layers
            })
        
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route("/api/ocr/generate", methods=["POST"])
    def generate_final_image():
        """生成最终图像"""
        try:
            data = request.get_json()
            image_data = data.get('image')
            text_layers = data.get('text_layers', [])
            
            if not image_data:
                return jsonify({'error': 'No image data provided'}), 400
            
            # 解码图像
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Invalid image data'}), 400
            
            # 应用文字图层
            final_image = ocr_replacer.apply_text_layers(image, text_layers)
            
            # 编码最终图像
            _, buffer = cv2.imencode('.jpg', final_image)
            final_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'final_image': f"data:image/jpeg;base64,{final_image_b64}"
            })
        
        except Exception as e:
            logger.error(f"Final image generation failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route("/api/ocr/translate", methods=["POST"])
    def translate_text():
        """翻译文字（这里可以集成翻译API）"""
        try:
            data = request.get_json()
            text = data.get('text', '')
            target_lang = data.get('target_lang', 'zh')
            
            # 这里可以集成真实的翻译API
            # 目前使用简单的映射
            translation_mapping = {
                'Усиленнаяверсия': '加强版',
                'Зкстракт трав': '草本提取物',
                'Без онемения': '无麻木感',
                'Продлевает + питает': '延长+滋养',
                'Безопасно,не вывываетпривыкания': '安全，不会产生依赖',
                'Цена': '价格',
                'CO скидкой': '有折扣',
                'Быстрый': '快速',
                'зффект: продление более 30 минут': '效果：延长超过30分钟',
                'Секрет мужской ВЫНОСЛИВОСТИ': '男性耐力的秘密',
                'Профессиональное средство': '专业产品'
            }
            
            translated_text = translation_mapping.get(text, text)
            
            return jsonify({
                'success': True,
                'original_text': text,
                'translated_text': translated_text
            })
        
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return jsonify({'error': str(e)}), 500 