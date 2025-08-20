#!/usr/bin/env python3
"""
SAM2 图像分割 Web 应用启动脚本
"""

import os
import sys

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, initialize_model

if __name__ == '__main__':
    print("🚀 启动 SAM2 图像分割 Web 应用...")
    print("📁 检查模型文件...")
    
    # 检查模型文件是否存在
    checkpoint_path = "checkpoints/sam2.1_hiera_base_plus.pt"
    config_path = "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
    
    # 转换为绝对路径
    current_dir = os.getcwd()
    abs_checkpoint_path = os.path.join(current_dir, checkpoint_path)
    abs_config_path = os.path.join(current_dir, config_path)
    
    if not os.path.exists(abs_checkpoint_path):
        print(f"❌ 模型文件不存在: {abs_checkpoint_path}")
        print("请确保已下载 SAM2 模型文件")
        sys.exit(1)
    
    if not os.path.exists(abs_config_path):
        print(f"❌ 配置文件不存在: {abs_config_path}")
        print("请确保 SAM2 配置文件存在")
        sys.exit(1)
    
    print("✅ 模型文件检查通过")
    
    # 初始化模型
    if initialize_model():
        print("✅ 模型加载成功")
        print("🌐 启动 Web 服务器...")
        print("📱 请在浏览器中访问: http://localhost:5000")
        print("⏹️  按 Ctrl+C 停止服务器")
        
        try:
            app.run(debug=False, host='0.0.0.0', port=5000)
        except KeyboardInterrupt:
            print("\n👋 服务器已停止")
    else:
        print("❌ 模型加载失败，请检查模型文件和依赖")
        sys.exit(1) 