#!/usr/bin/env python3
"""
SAM2 图像分割 Web 应用 - 快速启动
"""

import os
import sys

def main():
    print("🚀 SAM2 图像分割 Web 应用")
    print("=" * 40)
    
    # 检查模型文件
    checkpoint_path = "checkpoints/sam2.1_hiera_base_plus.pt"
    config_path = "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        print("请确保已下载 SAM2 模型文件")
        input("按回车键退出...")
        return
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        print("请确保 SAM2 配置文件存在")
        input("按回车键退出...")
        return
    
    print("✅ 模型文件检查通过")
    print()
    print("🌐 启动 Web 服务器...")
    print("📱 请在浏览器中访问: http://localhost:5000")
    print("⏹️  按 Ctrl+C 停止服务器")
    print("=" * 40)
    
    try:
        # 直接运行 run.py
        from run import app, initialize_model
        
        if initialize_model():
            print("✅ 模型加载成功")
            app.run(debug=False, host='0.0.0.0', port=5000)
        else:
            print("❌ 模型加载失败")
            input("按回车键退出...")
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        input("按回车键退出...")

if __name__ == '__main__':
    main()