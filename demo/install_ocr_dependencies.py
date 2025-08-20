#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import os

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def install_python_dependencies():
    """安装Python依赖"""
    dependencies = [
        "opencv-python",
        "pillow",
        "flask",
        "flask-cors",
        "numpy",
        "requests",
        "strawberry-graphql>=0.243.0"
    ]
    
    print("📦 安装Python依赖...")
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"安装 {dep}"):
            return False
    
    return True

def install_paddleocr():
    """安装PaddleOCR"""
    print("🤖 安装PaddleOCR...")
    
    # 先安装PaddlePaddle
    if not run_command("pip install paddlepaddle", "安装PaddlePaddle"):
        print("⚠️  PaddlePaddle安装失败，尝试CPU版本...")
        if not run_command("pip install paddlepaddle-cpu", "安装PaddlePaddle CPU版本"):
            return False
    
    # 安装PaddleOCR
    if not run_command("pip install paddleocr", "安装PaddleOCR"):
        return False
    
    return True

def install_node_dependencies():
    """安装Node.js依赖"""
    print("📦 安装Node.js依赖...")
    
    # 检查是否在frontend目录
    if not os.path.exists("frontend"):
        print("❌ 未找到frontend目录")
        return False
    
    os.chdir("frontend")
    
    # 安装npm依赖
    if not run_command("npm install", "安装npm依赖"):
        return False
    
    os.chdir("..")
    return True

def check_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ Python版本过低，需要Python 3.7或更高版本")
        return False
    
    print(f"✅ Python版本: {sys.version}")
    
    # 检查pip
    try:
        subprocess.run(["pip", "--version"], check=True, capture_output=True)
        print("✅ pip可用")
    except:
        print("❌ pip不可用")
        return False
    
    # 检查Node.js
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
        print("✅ Node.js可用")
    except:
        print("❌ Node.js不可用，请先安装Node.js")
        return False
    
    # 检查npm
    try:
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
        print("✅ npm可用")
    except:
        print("❌ npm不可用")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("OCR文字替换功能依赖安装器")
    print("=" * 60)
    
    # 检查系统要求
    if not check_requirements():
        print("\n❌ 系统要求检查失败，请解决上述问题后重试")
        return
    
    print("\n" + "=" * 60)
    print("开始安装依赖...")
    print("=" * 60)
    
    # 安装Python依赖
    if not install_python_dependencies():
        print("\n❌ Python依赖安装失败")
        return
    
    # 安装PaddleOCR
    if not install_paddleocr():
        print("\n❌ PaddleOCR安装失败")
        return
    
    # 安装Node.js依赖
    if not install_node_dependencies():
        print("\n❌ Node.js依赖安装失败")
        return
    
    print("\n" + "=" * 60)
    print("✅ 所有依赖安装完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 启动后端服务: python start_ocr_service.py")
    print("2. 启动前端服务: cd frontend && npm run dev")
    print("3. 访问: http://localhost:3000/ocr")
    print("\n如果遇到问题，请查看 demo/OCR_README.md")

if __name__ == "__main__":
    main() 