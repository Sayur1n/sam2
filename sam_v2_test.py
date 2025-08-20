#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM-V2 API 测试脚本
基于Segmind SAM-V2 API的图像分割测试
"""

import requests
import base64
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('.env')

class SAMV2Tester:
    """SAM-V2 API 测试类"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化SAM-V2 API测试器
        
        Args:
            api_key: API密钥，如果为None则从环境变量读取
        """
        self.api_key = api_key or os.getenv('SAM_V2_API_KEY')
        self.url = "https://api.segmind.com/v1/sam-v2-image"
        
        if not self.api_key:
            raise ValueError("请设置SAM_V2_API_KEY环境变量或在初始化时提供api_key参数")
    
    def image_file_to_base64(self, image_path: str) -> str:
        """
        将图像文件转换为base64编码
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            base64编码的图像数据
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    
    def image_url_to_base64(self, image_url: str) -> str:
        """
        从URL获取图像并转换为base64编码
        
        Args:
            image_url: 图像URL
            
        Returns:
            base64编码的图像数据
        """
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = response.content
        return base64.b64encode(image_data).decode('utf-8')
    
    def test_sam_v2_with_coordinates(self, 
                                   image_path: str, 
                                   coordinates: List[List[int]], 
                                   remove_coordinates: Optional[List[List[int]]] = None,
                                   overlay_mask: bool = True,
                                   output_base64: bool = False) -> dict:
        """
        使用坐标测试SAM-V2 API
        
        Args:
            image_path: 输入图像路径
            coordinates: 坐标列表，格式为 [[x1,y1], [x2,y2], ...]
            remove_coordinates: 要移除的坐标列表（可选）
            overlay_mask: 是否叠加mask
            output_base64: 是否输出base64编码
            
        Returns:
            API响应结果
        """
        print(f"🔍 测试SAM-V2 API...")
        print(f"图像路径: {image_path}")
        print(f"坐标: {coordinates}")
        print(f"移除坐标: {remove_coordinates}")
        print(f"叠加mask: {overlay_mask}")
        print(f"输出base64: {output_base64}")
        
        # 转换图像为base64
        image_base64 = self.image_file_to_base64(image_path)
        
        # 构建请求数据
        data = {
            "base64": output_base64,
            "image": image_base64,
            "overlay_mask": overlay_mask,
            "coordinates": json.dumps(coordinates)  # 坐标需要转换为字符串
        }
        
        # 添加可选的移除坐标
        if remove_coordinates:
            data["remove_coordinates"] = json.dumps(remove_coordinates)
        
        # 设置请求头
        headers = {'x-api-key': self.api_key}
        
        try:
            # 发送请求
            response = requests.post(self.url, json=data, headers=headers)
            response.raise_for_status()
            
            print("✅ SAM-V2 API调用成功!")
            
            # 分析响应内容
            self.analyze_response(response)
            
            return {
                "status_code": response.status_code,
                "content_type": response.headers.get('content-type'),
                "content_length": len(response.content),
                "response": response.content
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ SAM-V2 API调用失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text[:500]}...")  # 只显示前500字符
            raise
    
    def analyze_response(self, response: requests.Response):
        """
        分析API响应内容
        
        Args:
            response: API响应对象
        """
        print(f"\n📊 响应分析:")
        print(f"状态码: {response.status_code}")
        print(f"内容类型: {response.headers.get('content-type')}")
        print(f"内容长度: {len(response.content)} 字节")
        
        content_type = response.headers.get('content-type', '')
        
        if 'application/json' in content_type:
            # JSON响应
            try:
                json_data = response.json()
                print(f"JSON响应: {json.dumps(json_data, indent=2, ensure_ascii=False)}")
            except json.JSONDecodeError:
                print("❌ JSON解析失败")
                print(f"原始内容: {response.text[:500]}...")
        
        elif 'image/' in content_type:
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
        output_file = os.path.join(output_dir, f"sam_v2_mask{ext}")
        
        try:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✅ 图像已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存图像失败: {e}")
    
    def test_multiple_coordinates(self, image_path: str):
        """
        测试多个坐标点
        
        Args:
            image_path: 图像文件路径
        """
        print(f"\n🧪 多坐标测试")
        print("=" * 50)
        
        # 定义不同的坐标测试
        test_cases = [
            {
                "name": "单点测试",
                "coordinates": [[400, 300]],
                "description": "测试单个坐标点"
            },
            {
                "name": "多点测试",
                "coordinates": [[400, 300], [500, 400], [600, 350]],
                "description": "测试多个坐标点"
            },
            {
                "name": "边界测试",
                "coordinates": [[100, 100], [800, 600]],
                "description": "测试边界坐标"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n测试 {i}: {test_case['name']}")
            print(f"描述: {test_case['description']}")
            print(f"坐标: {test_case['coordinates']}")
            
            try:
                result = self.test_sam_v2_with_coordinates(
                    image_path=image_path,
                    coordinates=test_case['coordinates'],
                    overlay_mask=True,
                    output_base64=False
                )
                print(f"✅ 测试 {i} 成功")
                
            except Exception as e:
                print(f"❌ 测试 {i} 失败: {e}")
            
            print("-" * 30)


def main():
    """主函数"""
    print("🚀 SAM-V2 API 测试脚本")
    print("=" * 50)
    
    # 检查环境变量
    api_key = os.getenv('SAM_V2_API_KEY')
    if not api_key:
        print("❌ 请设置SAM_V2_API_KEY环境变量")
        print("您可以在.env文件中设置: SAM_V2_API_KEY=your_api_key_here")
        return
    
    # 创建测试器
    try:
        tester = SAMV2Tester()
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 测试图像路径
    image_path = "OCR/images/image1.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ 测试图像不存在: {image_path}")
        return
    
    print(f"📸 使用测试图像: {image_path}")
    
    # 基本测试
    print(f"\n🔍 基本测试")
    print("=" * 30)
    
    try:
        # 使用图像中心附近的坐标
        coordinates = [2048, 1536]
        
        result = tester.test_sam_v2_with_coordinates(
            image_path=image_path,
            coordinates=coordinates,
            overlay_mask=True,
            output_base64=False
        )
        
        print(f"✅ 基本测试完成")
        
    except Exception as e:
        print(f"❌ 基本测试失败: {e}")
    
    print(f"\n{'='*50}")
    print("🎉 测试完成!")


if __name__ == "__main__":
    main() 