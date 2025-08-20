#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标一致性测试脚本
验证交互编辑和最终生成的坐标系统是否一致
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

def create_test_image(width=800, height=600):
    """创建测试图片"""
    # 创建白色背景
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 添加一些测试文字
    cv2.putText(image, "Test Text 1", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, "Test Text 2", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, "Test Text 3", (500, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return image

def simulate_interactive_coordinates(image, test_boxes):
    """模拟交互编辑时的坐标系统"""
    height, width = image.shape[:2]
    
    # 模拟Canvas缩放（假设显示尺寸是原始尺寸的一半）
    display_width = width // 2
    display_height = height // 2
    scale_x = display_width / width
    scale_y = display_height / height
    
    print(f"🔍 交互编辑坐标系统:")
    print(f"  原始图片尺寸: {width}x{height}")
    print(f"  显示尺寸: {display_width}x{display_height}")
    print(f"  缩放比例: scaleX={scale_x:.3f}, scaleY={scale_y:.3f}")
    
    interactive_coords = []
    for i, box in enumerate(test_boxes):
        x1, y1, x2, y2 = box
        # 缩放坐标到显示尺寸
        scaled_x1 = x1 * scale_x
        scaled_y1 = y1 * scale_y
        scaled_x2 = x2 * scale_x
        scaled_y2 = y2 * scale_y
        
        interactive_coords.append({
            'id': f'layer_{i+1}',
            'original_box': [x1, y1, x2, y2],
            'scaled_box': [scaled_x1, scaled_y1, scaled_x2, scaled_y2],
            'text': f'测试文字 {i+1}',
            'font_size': 20
        })
        
        print(f"  图层 {i+1}:")
        print(f"    原始坐标: [{x1}, {y1}, {x2}, {y2}]")
        print(f"    显示坐标: [{scaled_x1:.1f}, {scaled_y1:.1f}, {scaled_x2:.1f}, {scaled_y2:.1f}]")
    
    return interactive_coords

def simulate_final_generation(image, interactive_coords):
    """模拟最终生成时的坐标系统"""
    height, width = image.shape[:2]
    
    print(f"\n🔍 最终生成坐标系统:")
    print(f"  图片尺寸: {width}x{height}")
    
    final_coords = []
    for coord in interactive_coords:
        # 使用原始坐标（不缩放）
        original_box = coord['original_box']
        x1, y1, x2, y2 = original_box
        
        final_coords.append({
            'id': coord['id'],
            'box': original_box,
            'text': coord['text'],
            'font_size': coord['font_size'],
            'text_color': [255, 0, 0]  # 红色
        })
        
        print(f"  图层 {coord['id']}:")
        print(f"    最终坐标: [{x1}, {y1}, {x2}, {y2}]")
        print(f"    文字: '{coord['text']}'")
    
    return final_coords

def add_text_to_image_canvas_style(image, coords):
    """使用Canvas风格的文字渲染（模拟前端）"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
        except:
            font = ImageFont.load_default()
    
    for coord in coords:
        x1, y1, x2, y2 = coord['box']
        text = coord['text']
        
        # 计算中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 计算文字边界框
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 居中绘制文字
        start_x = center_x - text_width // 2
        start_y = center_y - text_height // 2
        
        # 绘制文字
        draw.text((start_x, start_y), text, fill=(255, 0, 0), font=font)
        
        print(f"  Canvas风格 - 图层 {coord['id']}:")
        print(f"    框中心: ({center_x}, {center_y})")
        print(f"    文字尺寸: {text_width}x{text_height}")
        print(f"    绘制位置: ({start_x}, {start_y})")
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def add_text_to_image_pil_style(image, coords):
    """使用PIL风格的文字渲染（模拟后端）"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
        except:
            font = ImageFont.load_default()
    
    for coord in coords:
        x1, y1, x2, y2 = coord['box']
        text = coord['text']
        
        # 计算中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 计算文字边界框
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 居中绘制文字
        start_x = center_x - text_width // 2
        start_y = center_y - text_height // 2
        
        # 绘制文字
        draw.text((start_x, start_y), text, fill=(0, 255, 0), font=font)  # 绿色
        
        print(f"  PIL风格 - 图层 {coord['id']}:")
        print(f"    框中心: ({center_x}, {center_y})")
        print(f"    文字尺寸: {text_width}x{text_height}")
        print(f"    绘制位置: ({start_x}, {start_y})")
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """主测试函数"""
    print("🎯 坐标一致性测试")
    print("=" * 50)
    
    # 创建测试图片
    test_image = create_test_image(800, 600)
    
    # 定义测试框
    test_boxes = [
        [100, 100, 300, 150],   # 第一个文字框
        [300, 200, 500, 250],   # 第二个文字框
        [500, 300, 700, 350],   # 第三个文字框
    ]
    
    # 模拟交互编辑坐标
    interactive_coords = simulate_interactive_coordinates(test_image, test_boxes)
    
    # 模拟最终生成坐标
    final_coords = simulate_final_generation(test_image, interactive_coords)
    
    # 创建Canvas风格的图片
    canvas_image = add_text_to_image_canvas_style(test_image.copy(), final_coords)
    
    # 创建PIL风格的图片
    pil_image = add_text_to_image_pil_style(test_image.copy(), final_coords)
    
    # 保存结果
    cv2.imwrite('test_canvas_style.jpg', canvas_image)
    cv2.imwrite('test_pil_style.jpg', pil_image)
    
    print(f"\n✅ 测试完成!")
    print(f"  Canvas风格图片: test_canvas_style.jpg")
    print(f"  PIL风格图片: test_pil_style.jpg")
    print(f"  请比较两张图片，检查文字位置是否一致")

if __name__ == "__main__":
    main() 