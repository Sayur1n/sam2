#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG图片大小调整脚本
将指定的PNG图片调整到小于4MB的大小
"""

import os
import sys
from PIL import Image
import argparse

def get_file_size_mb(file_path):
    """获取文件大小（MB）"""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)

def resize_image(input_path, output_path, max_size_mb=4.0):
    """
    调整图片大小，确保输出文件小于指定大小
    
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        max_size_mb: 最大文件大小（MB）
    """
    try:
        # 打开图片
        with Image.open(input_path) as img:
            # 转换为RGB模式（如果不是的话）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 获取原始尺寸
            original_width, original_height = img.size
            print(f"原始尺寸: {original_width}x{original_height}")
            print(f"原始文件大小: {get_file_size_mb(input_path):.2f} MB")
            
            # 如果原始文件已经小于目标大小，直接复制
            if get_file_size_mb(input_path) <= max_size_mb:
                img.save(output_path, 'PNG', optimize=True)
                print(f"原始文件已小于 {max_size_mb}MB，直接保存")
                return
            
            # 计算初始缩放比例
            scale_factor = 0.9
            
            while True:
                # 计算新的尺寸
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                
                # 调整图片大小
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 保存到临时文件以检查大小
                temp_path = output_path + '.temp'
                resized_img.save(temp_path, 'PNG', optimize=True)
                
                file_size_mb = get_file_size_mb(temp_path)
                print(f"当前尺寸: {new_width}x{new_height}, 文件大小: {file_size_mb:.2f} MB")
                
                # 如果文件大小符合要求，保存最终文件
                if file_size_mb <= max_size_mb:
                    os.rename(temp_path, output_path)
                    print(f"调整完成！最终尺寸: {new_width}x{new_height}")
                    print(f"最终文件大小: {file_size_mb:.2f} MB")
                    break
                
                # 删除临时文件
                os.remove(temp_path)
                
                # 继续缩小尺寸
                scale_factor *= 0.9
                
                # 防止无限循环
                if scale_factor < 0.1:
                    print("警告：无法将文件压缩到目标大小，使用最小尺寸保存")
                    resized_img.save(output_path, 'PNG', optimize=True)
                    break
                    
    except Exception as e:
        print(f"处理图片时出错: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='调整PNG图片大小，确保小于4MB')
    parser.add_argument('input', help='输入PNG图片路径')
    parser.add_argument('-o', '--output', help='输出图片路径（可选，默认添加_resized后缀）')
    parser.add_argument('-s', '--size', type=float, default=4.0, help='最大文件大小（MB，默认4.0）')
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误：输入文件 '{args.input}' 不存在")
        sys.exit(1)
    
    # 检查文件扩展名
    if not args.input.lower().endswith('.png'):
        print("警告：输入文件不是PNG格式")
    
    # 设置输出路径
    if args.output:
        output_path = args.output
    else:
        name, ext = os.path.splitext(args.input)
        output_path = f"{name}_resized{ext}"
    
    print(f"开始处理图片: {args.input}")
    print(f"目标文件大小: {args.size} MB")
    print(f"输出文件: {output_path}")
    print("-" * 50)
    
    # 调整图片
    success = resize_image(args.input, output_path, args.size)
    
    if success:
        print(f"\n✅ 图片调整完成！")
        print(f"输出文件: {output_path}")
        print(f"最终大小: {get_file_size_mb(output_path):.2f} MB")
    else:
        print("\n❌ 图片调整失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 