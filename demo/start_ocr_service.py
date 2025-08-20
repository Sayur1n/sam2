#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = [
        'opencv-python',
        'pillow',
        'flask',
        'flask-cors',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_paddleocr():
    """检查PaddleOCR是否可用"""
    try:
        from paddleocr import PaddleOCR
        print("✅ PaddleOCR 可用")
        return True
    except ImportError:
        print("⚠️  PaddleOCR 未安装，OCR功能将不可用")
        print("   如需OCR功能，请运行: pip install paddlepaddle paddleocr")
        return False

def start_backend():
    """启动后端服务"""
    print("🚀 启动后端服务...")
    
    # 切换到后端目录
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    # 启动Flask应用
    try:
        from server.app import app
        print("✅ 后端服务启动成功")
        print("   服务地址: http://localhost:5000")
        print("   OCR API: http://localhost:5000/api/ocr/")
        
        # 启动Flask开发服务器
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        print(f"❌ 后端服务启动失败: {e}")
        return False

def check_frontend():
    """检查前端是否已构建"""
    frontend_dir = Path(__file__).parent / "frontend"
    dist_dir = frontend_dir / "dist"
    
    if dist_dir.exists():
        print("✅ 前端已构建")
        return True
    else:
        print("⚠️  前端未构建，请先构建前端:")
        print("   cd frontend && npm install && npm run build")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("OCR文字替换服务启动器")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查PaddleOCR
    check_paddleocr()
    
    print("\n" + "=" * 50)
    print("服务信息:")
    print("=" * 50)
    print("后端服务: http://localhost:5000")
    print("前端服务: http://localhost:3000 (需要单独启动)")
    print("OCR功能: http://localhost:3000/ocr")
    print("\n启动说明:")
    print("1. 后端服务将在本脚本中启动")
    print("2. 前端服务需要单独启动:")
    print("   cd frontend && npm run dev")
    print("3. 访问 http://localhost:3000/ocr 使用OCR功能")
    
    print("\n" + "=" * 50)
    print("正在启动后端服务...")
    print("=" * 50)
    
    # 启动后端服务
    start_backend()

if __name__ == "__main__":
    main() 