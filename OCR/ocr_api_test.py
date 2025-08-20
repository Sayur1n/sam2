#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR API 测试脚本
基于星河API的OCR功能测试
"""

import base64
import os
import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 加载环境变量（从上一级目录的env文件）
load_dotenv('.env')

# 默认配置
DEFAULT_API_URL = "https://jdebf6gbs2x6g0y4.aistudio-hub.baidu.com/ocr"

class OCRAPITester:
    """OCR API 测试类"""
    
    def __init__(self, api_url: Optional[str] = None, token: Optional[str] = None):
        """
        初始化OCR API测试器
        
        Args:
            api_url: API地址，如果为None则从环境变量读取
            token: 访问令牌，如果为None则从环境变量读取
        """
        self.api_url = api_url or os.getenv('OCR_API_URL', DEFAULT_API_URL)
        self.token = token or os.getenv('OCR_TOKEN')
        
        if not self.token:
            raise ValueError("请设置OCR_TOKEN环境变量或在初始化时提供token参数")
    
    def test_image_ocr(self, file_path: str, file_type: int = 1) -> Dict[str, Any]:
        """
        测试图像OCR识别
        
        Args:
            file_path: 图像文件路径
            file_type: 文件类型，0为PDF，1为图像
            
        Returns:
            API响应结果
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取文件并编码
        with open(file_path, "rb") as file:
            file_bytes = file.read()
            file_data = base64.b64encode(file_bytes).decode("ascii")
        
        # 设置请求头
        headers = {
            "Authorization": f"token {self.token}",
            "Content-Type": "application/json"
        }
        
        # 设置请求体
        payload = {
            "file": file_data, 
            "fileType": file_type
        }
        
        print(f"正在测试OCR API...")
        print(f"API URL: {self.api_url}")
        print(f"文件路径: {file_path}")
        print(f"文件类型: {'PDF' if file_type == 0 else '图像'}")
        
        # 发送请求
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            print("✅ OCR API调用成功!")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ OCR API调用失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            raise
    
    def print_ocr_results(self, result: Dict[str, Any]):
        """
        打印OCR识别结果
        
        Args:
            result: API响应结果
        """
        # 检查是否为API原始格式
        if "result" in result and "ocrResults" in result["result"]:
            # API原始格式
            ocr_results = result["result"].get("ocrResults", [])
            
            if not ocr_results:
                print("⚠️ 没有识别到任何文本")
                return
            
            print(f"\n📝 识别到 {len(ocr_results)} 个文本区域:")
            print("=" * 50)
            
            for i, res in enumerate(ocr_results, 1):
                print(f"\n区域 {i}:")
                pruned_result = res.get('prunedResult', {})
                rec_texts = pruned_result.get('rec_texts', [])
                rec_scores = pruned_result.get('rec_scores', [])
                
                if rec_texts:
                    print(f"识别文本: {rec_texts}")
                    if rec_scores:
                        print(f"置信度: {rec_scores}")
                else:
                    print(f"识别文本: {res.get('prunedResult', 'N/A')}")
                
                print(f"图像URL: {res.get('ocrImage', 'N/A')}")
                
        else:
            # 标准格式
            rec_texts = result.get("rec_texts", [])
            rec_scores = result.get("rec_scores", [])
            
            if not rec_texts:
                print("⚠️ 没有识别到任何文本")
                return
            
            print(f"\n📝 识别到 {len(rec_texts)} 个文本区域:")
            print("=" * 50)
            
            for i, (text, score) in enumerate(zip(rec_texts, rec_scores), 1):
                print(f"\n区域 {i}:")
                print(f"识别文本: {text}")
                print(f"置信度: {score}")
    
    def convert_api_result_to_standard_format(self, api_result: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """
        将星河API返回的结果转换为标准格式
        
        Args:
            api_result: API返回的原始结果
            file_path: 输入文件路径
            
        Returns:
            转换后的标准格式结果
        """
        if "result" not in api_result or "ocrResults" not in api_result["result"]:
            raise ValueError("API返回结果格式不正确")
        
        # 获取第一个OCR结果（通常只有一个）
        ocr_result = api_result["result"]["ocrResults"][0]
        pruned_result = ocr_result.get("prunedResult", {})
        
        # 构建标准格式
        standard_result = {
            "input_path": file_path,
            "page_index": None,
            "model_settings": {
                "use_doc_preprocessor": pruned_result.get("model_settings", {}).get("use_doc_preprocessor", True),
                "use_textline_orientation": pruned_result.get("model_settings", {}).get("use_textline_orientation", False)
            },
            "doc_preprocessor_res": {
                "input_path": None,
                "page_index": None,
                "model_settings": {
                    "use_doc_orientation_classify": pruned_result.get("doc_preprocessor_res", {}).get("model_settings", {}).get("use_doc_orientation_classify", False),
                    "use_doc_unwarping": pruned_result.get("doc_preprocessor_res", {}).get("model_settings", {}).get("use_doc_unwarping", False)
                },
                "angle": pruned_result.get("doc_preprocessor_res", {}).get("angle", -1)
            },
            "dt_polys": pruned_result.get("dt_polys", []),
            "text_det_params": pruned_result.get("text_det_params", {}),
            "text_type": pruned_result.get("text_type", "general"),
            "textline_orientation_angles": pruned_result.get("textline_orientation_angles", []),
            "text_rec_score_thresh": pruned_result.get("text_rec_score_thresh", 0.0),
            "rec_texts": pruned_result.get("rec_texts", []),
            "rec_scores": pruned_result.get("rec_scores", []),
            "rec_polys": pruned_result.get("rec_polys", []),
            "rec_boxes": pruned_result.get("rec_boxes", [])
        }
        
        return standard_result
    
    def save_results(self, result: Dict[str, Any], output_file: str, file_path: str = None):
        """
        保存OCR结果到文件
        
        Args:
            result: API响应结果
            output_file: 输出文件路径
            file_path: 输入文件路径（用于转换格式）
        """
        try:
            # 保存原始API结果
            raw_output_file = output_file.replace('.json', '_raw.json')
            with open(raw_output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ 原始API结果已保存到: {raw_output_file}")
            
            # 转换为标准格式并保存
            if file_path:
                standard_result = self.convert_api_result_to_standard_format(result, file_path)
            else:
                standard_result = result
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(standard_result, f, ensure_ascii=False, indent=2)
            print(f"✅ 标准格式结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")


def main():
    """主函数"""
    print("🚀 OCR API 测试脚本")
    print("=" * 50)
    
    # 检查环境变量
    token = os.getenv('OCR_TOKEN')
    if not token:
        print("❌ 请设置OCR_TOKEN环境变量")
        print("您可以在.env文件中设置: OCR_TOKEN=your_token_here")
        return
    
    # 创建测试器
    try:
        tester = OCRAPITester()
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 测试文件路径
    test_files = [
        "OCR/images/image1.jpg",  # 使用OCR目录下的测试图像
        "OCR/images/image2.jpg"
    ]
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 测试文件不存在: {file_path}")
            continue
        
        print(f"\n🔍 测试文件: {file_path}")
        print("-" * 30)
        
        try:
            # 执行OCR测试
            result = tester.test_image_ocr(file_path)
            
            # 打印结果
            tester.print_ocr_results(result)
            
            # 保存结果
            output_file = f"ocr_output/test_result_{Path(file_path).stem}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            tester.save_results(result, output_file, file_path)
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
        
        print("\n" + "=" * 50)


if __name__ == "__main__":
    main() 