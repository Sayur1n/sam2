#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM-V2 交互式测试脚本
基于用户点击的图像分割测试
"""

import requests
import base64
import json
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('.env')

class InteractiveSAMV2Tester:
    """交互式SAM-V2 API 测试类"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化交互式SAM-V2 API测试器
        
        Args:
            api_key: API密钥，如果为None则从环境变量读取
        """
        self.api_key = api_key or os.getenv('SAM_V2_API_KEY')
        self.url = "https://api.segmind.com/v1/sam-v2-image"
        
        if not self.api_key:
            raise ValueError("请设置SAM_V2_API_KEY环境变量或在初始化时提供api_key参数")
        
        # 交互式变量
        self.clicked_points = []
        self.original_image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.window_name = "SAM-V2 交互式测试 - 点击选择坐标，按ESC退出，按Enter确认"
    
    def resize_image_to_max_dimension(self, image: np.ndarray, max_dimension: int = 1024) -> Tuple[np.ndarray, float]:
        """
        将图像等比例缩放到最大尺寸内（用于显示）
        
        Args:
            image: 输入图像
            max_dimension: 最大尺寸
            
        Returns:
            缩放后的图像和缩放因子
        """
        height, width = image.shape[:2]
        
        # 计算缩放因子
        scale_factor = min(max_dimension / width, max_dimension / height)
        
        if scale_factor < 1.0:
            # 需要缩放
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"📏 显示图像已缩放: {width}x{height} -> {new_width}x{new_height} (缩放因子: {scale_factor:.3f})")
            return resized_image, scale_factor
        else:
            # 不需要缩放
            print(f"📏 显示图像尺寸合适: {width}x{height}")
            return image, 1.0
    
    def load_and_prepare_image(self, image_path: str) -> bool:
        """
        加载并准备图像用于交互
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            是否成功加载
        """
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            return False
        
        # 读取图像
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return False
        
        # 缩放图像
        self.display_image, self.scale_factor = self.resize_image_to_max_dimension(self.original_image)
        
        # 创建显示图像的副本
        self.display_image = self.display_image.copy()
        
        print(f"✅ 图像加载成功: {image_path}")
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        鼠标回调函数
        
        Args:
            event: 鼠标事件
            x, y: 鼠标坐标
            flags: 标志
            param: 参数
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键点击，添加坐标点
            self.clicked_points.append([x, y])
            
            # 在图像上绘制点击点
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.display_image, f"{len(self.clicked_points)}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 计算原始图像坐标
            original_x = int(x / self.scale_factor)
            original_y = int(y / self.scale_factor)
            
            print(f"📍 添加坐标点 {len(self.clicked_points)}: 显示[{x}, {y}] -> 原始[{original_x}, {original_y}]")
            
            # 在图像上显示坐标点统计信息
            self.draw_coordinates_info()
            
            # 更新显示
            cv2.imshow(self.window_name, self.display_image)
    
    def run_interactive_session(self, image_path: str):
        """
        运行交互式会话
        
        Args:
            image_path: 图像文件路径
        """
        # 加载图像
        if not self.load_and_prepare_image(image_path):
            return
        
        # 创建窗口和鼠标回调
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # 设置窗口大小，确保能看到整张图片
        height, width = self.display_image.shape[:2]
        screen_width = 1920  # 假设屏幕宽度
        screen_height = 1080  # 假设屏幕高度
        
        # 计算合适的窗口大小，留出一些边距
        max_window_width = min(width, screen_width - 100)
        max_window_height = min(height, screen_height - 100)
        
        # 等比例缩放窗口
        window_scale = min(max_window_width / width, max_window_height / height)
        window_width = int(width * window_scale)
        window_height = int(height * window_scale)
        
        cv2.resizeWindow(self.window_name, window_width, window_height)
        print(f"🖥️ 窗口大小设置为: {window_width}x{window_height}")
        
        # 在初始图像上显示坐标点信息
        self.draw_coordinates_info()
        
        # 显示图像
        cv2.imshow(self.window_name, self.display_image)
        
        print("\n🎯 交互式操作说明:")
        print("- 左键点击: 添加坐标点")
        print("- ESC键: 退出程序")
        print("- Enter键: 确认并执行SAM-V2分割")
        print("- R键: 重置所有坐标点")
        print("- S键: 保存当前图像")
        print("- 可以拖拽窗口边缘调整大小")
        print(f"📏 当前显示图像尺寸: {self.display_image.shape[1]}x{self.display_image.shape[0]}")
        print(f"📏 原始图像尺寸: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC键
                print("👋 用户退出")
                break
            elif key == 13:  # Enter键
                if self.clicked_points:
                    print(f"\n🚀 执行SAM-V2分割，坐标点: {self.clicked_points}")
                    self.execute_sam_v2_segmentation(image_path)
                    break
                else:
                    print("⚠️ 请先点击添加坐标点")
            elif key == ord('r') or key == ord('R'):  # R键重置
                self.reset_coordinates()
            elif key == ord('s') or key == ord('S'):  # S键保存
                self.save_current_image()
        
        cv2.destroyAllWindows()
    
    def draw_coordinates_info(self):
        """在图像上绘制坐标点统计信息"""
        # 创建信息文本
        info_text = f"坐标点: {len(self.clicked_points)}"
        
        # 在图像左上角绘制信息
        cv2.putText(self.display_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.display_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    def reset_coordinates(self):
        """重置所有坐标点"""
        self.clicked_points = []
        self.display_image = self.original_image.copy()
        self.display_image, _ = self.resize_image_to_max_dimension(self.display_image)
        self.draw_coordinates_info()
        cv2.imshow(self.window_name, self.display_image)
        print("🔄 已重置所有坐标点")
    
    def save_current_image(self):
        """保存当前图像"""
        output_file = "sam_v2_output/interactive_image.jpg"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cv2.imwrite(output_file, self.display_image)
        print(f"💾 当前图像已保存到: {output_file}")
    
    def execute_sam_v2_segmentation(self, image_path: str):
        """
        执行SAM-V2分割
        
        Args:
            image_path: 原始图像路径
        """
        if not self.clicked_points:
            print("❌ 没有坐标点")
            return
        
        print(f"🔍 执行SAM-V2分割...")
        print(f"坐标点数量: {len(self.clicked_points)}")
        print(f"坐标点: {self.clicked_points}")
        
        # 将显示坐标转换为原始图像坐标
        original_coordinates = []
        for point in self.clicked_points:
            # 反向缩放坐标
            original_x = int(point[0] / self.scale_factor)
            original_y = int(point[1] / self.scale_factor)
            original_coordinates.append([original_x, original_y])
        
        print(f"原始图像坐标: {original_coordinates}")
        
        # 调用SAM-V2 API
        try:
            result = self.call_sam_v2_api(image_path, original_coordinates)
            print("✅ SAM-V2分割完成")
        except Exception as e:
            print(f"❌ SAM-V2分割失败: {e}")
    
    def call_sam_v2_api(self, image_path: str, coordinates: List[List[int]]) -> dict:
        """
        调用SAM-V2 API
        
        Args:
            image_path: 图像文件路径
            coordinates: 坐标列表
            
        Returns:
            API响应结果
        """
        # 转换图像为base64
        image_base64 = self.image_file_to_base64(image_path)
        
        # 构建请求数据
        data = {
            "base64": False,
            "image": image_base64,
            "overlay_mask": True,
            "coordinates": json.dumps(coordinates)
        }
        
        # 设置请求头
        headers = {'x-api-key': self.api_key}
        
        print(f"📡 发送API请求...")
        print(f"图像路径: {image_path}")
        print(f"坐标: {coordinates}")
        
        # 发送请求
        response = requests.post(self.url, json=data, headers=headers)
        response.raise_for_status()
        
        print("✅ API调用成功!")
        
        # 分析并保存响应
        self.analyze_and_save_response(response)
        
        return {
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type'),
            "content_length": len(response.content)
        }
    
    def image_file_to_base64(self, image_path: str) -> str:
        """
        将图像文件转换为base64编码
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            base64编码的图像数据
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    
    def analyze_and_save_response(self, response: requests.Response):
        """
        分析并保存API响应
        
        Args:
            response: API响应对象
        """
        print(f"\n📊 响应分析:")
        print(f"状态码: {response.status_code}")
        print(f"内容类型: {response.headers.get('content-type')}")
        print(f"内容长度: {len(response.content)} 字节")
        
        content_type = response.headers.get('content-type', '')
        
        if 'image/' in content_type:
            # 图像响应
            print(f"图像响应 - 格式: {content_type}")
            print(f"图像大小: {len(response.content)} 字节")
            
            # 保存图像
            self.save_image_response(response)
        else:
            # 其他类型响应
            print(f"其他类型响应: {content_type}")
            print(f"内容预览: {response.content[:200]}...")
    
    def save_image_response(self, response: requests.Response, output_dir: str = "sam_v2_output"):
        """
        保存图像响应
        
        Args:
            response: API响应对象
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 从content-type推断文件扩展名
        content_type = response.headers.get('content-type', '')
        if 'png' in content_type:
            ext = '.png'
        elif 'jpeg' in content_type or 'jpg' in content_type:
            ext = '.jpg'
        else:
            ext = '.png'  # 默认使用PNG
        
        # 生成输出文件名
        output_file = os.path.join(output_dir, f"sam_v2_interactive_mask{ext}")
        
        try:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✅ 分割结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存图像失败: {e}")


def main():
    """主函数"""
    print("🚀 SAM-V2 交互式测试脚本")
    print("=" * 50)
    
    # 检查环境变量
    api_key = os.getenv('SAM_V2_API_KEY')
    if not api_key:
        print("❌ 请设置SAM_V2_API_KEY环境变量")
        print("您可以在.env文件中设置: SAM_V2_API_KEY=your_api_key_here")
        return
    
    # 创建测试器
    try:
        tester = InteractiveSAMV2Tester()
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 测试图像路径
    image_path = "OCR/images/image2.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ 测试图像不存在: {image_path}")
        return
    
    print(f"📸 使用测试图像: {image_path}")
    
    # 运行交互式会话
    tester.run_interactive_session(image_path)
    
    print(f"\n{'='*50}")
    print("🎉 测试完成!")


if __name__ == "__main__":
    main() 