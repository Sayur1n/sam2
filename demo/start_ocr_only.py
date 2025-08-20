#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import json

# 导入OCR功能
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'OCR'))
from ocr_text_replacement import OCRTextReplacer

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 创建OCR实例
ocr_replacer = OCRTextReplacer()

@app.route('/')
def index():
    """主页 - 简单的测试页面"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCR文字替换服务</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .status { color: green; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 OCR文字替换服务</h1>
            <p class="status">✅ 服务运行正常</p>
            
            <h2>📡 API端点</h2>
            <div class="endpoint">
                <strong>POST /api/ocr/detect</strong> - 检测图片中的文字
            </div>
            <div class="endpoint">
                <strong>POST /api/ocr/process</strong> - 处理图片（移除原文字，准备文字图层）
            </div>
            <div class="endpoint">
                <strong>POST /api/ocr/generate</strong> - 生成最终图片（应用文字图层）
            </div>
            <div class="endpoint">
                <strong>POST /api/ocr/translate</strong> - 翻译文字
            </div>
            <div class="endpoint">
                <strong>GET /healthy</strong> - 健康检查
            </div>
            
            <h2>🔗 前端应用</h2>
            <p>请访问前端应用：<a href="http://localhost:3000/ocr" target="_blank">http://localhost:3000/ocr</a></p>
            
            <h2>📋 使用说明</h2>
            <ol>
                <li>启动前端：<code>cd frontend && npm run dev</code></li>
                <li>访问：<a href="http://localhost:3000/ocr" target="_blank">http://localhost:3000/ocr</a></li>
                <li>上传图片并开始编辑</li>
            </ol>
        </div>
    </body>
    </html>
    '''

@app.route('/api/ocr/detect', methods=['POST'])
def detect_text():
    """检测图片中的文字"""
    try:
        data = request.json
        image_data = data['image']
        
        # 解码base64图片
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # 检测文字
        text_regions = ocr_replacer.detect_text(image)
        
        return jsonify({
            'success': True,
            'text_regions': text_regions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ocr/process', methods=['POST'])
def process_image():
    """处理图片：移除原文字，准备文字图层"""
    try:
        data = request.json
        image_data = data['image']
        
        # 解码base64图片
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # 处理图片
        processed_image, text_layers = ocr_replacer.process_image(image)
        
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ocr/generate', methods=['POST'])
def generate_final_image():
    """生成最终图片：应用文字图层"""
    try:
        data = request.json
        image_data = data['image']
        text_layers = data['text_layers']
        
        # 解码base64图片
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # 应用文字图层
        final_image = ocr_replacer.apply_text_layers(image, text_layers)
        
        # 转换最终图片为base64
        buffered = io.BytesIO()
        final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'final_image': f'data:image/png;base64,{img_str}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ocr/translate', methods=['POST'])
def translate_text():
    """翻译文字（简单示例）"""
    try:
        data = request.json
        text = data['text']
        
        # 这里可以集成真实的翻译API
        # 现在只是简单的示例
        translated_text = f"[翻译] {text}"
        
        return jsonify({
            'success': True,
            'translated_text': translated_text
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/healthy')
def healthy():
    """健康检查"""
    return jsonify({'status': 'OK'})

if __name__ == '__main__':
    print("🚀 启动OCR专用服务...")
    print("📡 服务地址: http://localhost:5000")
    print("🔗 API端点:")
    print("  - POST /api/ocr/detect - 检测文字")
    print("  - POST /api/ocr/process - 处理图片")
    print("  - POST /api/ocr/generate - 生成最终图片")
    print("  - POST /api/ocr/translate - 翻译文字")
    print("  - GET  /healthy - 健康检查")
    print("\n按 Ctrl+C 停止服务")
    
    app.run(host='0.0.0.0', port=5000, debug=True)