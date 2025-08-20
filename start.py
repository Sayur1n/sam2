#!/usr/bin/env python3
"""
SAM2 图像分割 Web 应用启动脚本
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """检查 Python 版本"""
    print("🐍 检查 Python 版本...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python 版本过低: {version.major}.{version.minor}")
        print("请使用 Python 3.7 或更高版本")
        return False
    print(f"✅ Python 版本: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """检查并安装依赖"""
    print("📦 检查依赖...")
    
    required_packages = [
        'flask',
        'flask_cors', 
        'torch',
        'torchvision',
        'opencv-python',
        'pillow',
        'matplotlib',
        'numpy',
        'openai',
        'python-dotenv',
        'paddlepaddle',
        'paddleocr'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        # 处理包名映射
        import_name = package.replace('-', '_')
        if package == 'opencv-python':
            import_name = 'cv2'
        elif package == 'pillow':
            import_name = 'PIL'
        
        try:
            importlib.import_module(import_name)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📥 安装缺失的依赖: {', '.join(missing_packages)}")
        try:
            # 首先尝试使用 --user 选项安装
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--user', '-r', 'requirements.txt'
            ])
            print("✅ 依赖安装完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖安装失败: {e}")
            print("💡 提示: 如果遇到权限问题，请尝试以管理员身份运行")
            return False
    else:
        print("✅ 所有依赖已安装")
    
    return True

def check_model_files():
    """检查模型文件"""
    print("📁 检查模型文件...")
    
    checkpoint_path = "checkpoints/sam2.1_hiera_base_plus.pt"
    config_path = "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        print("请确保已下载 SAM2 模型文件")
        return False
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        print("请确保 SAM2 配置文件存在")
        return False
    
    print("✅ 模型文件检查通过")
    return True

def start_application():
    """启动应用"""
    print("🚀 启动 SAM2 图像分割 Web 应用...")
    print("=" * 50)
    
    # 检查 Python 版本
    if not check_python_version():
        input("按回车键退出...")
        return
    
    print()
    
    # 检查依赖
    if not check_dependencies():
        input("按回车键退出...")
        return
    
    print()
    
    # 检查模型文件
    if not check_model_files():
        input("按回车键退出...")
        return
    
    print()
    
    # 启动应用
    print("🌐 启动 Web 服务器...")
    print("📱 请在浏览器中访问: http://localhost:5000")
    print("⏹️  按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    try:
        # 导入并运行应用
        from app import app, initialize_model
        
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
    start_application() 