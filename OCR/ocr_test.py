from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
import argparse

def load_mask_from_png(mask_path):
    """从PNG文件加载掩码"""
    if not mask_path or not os.path.exists(mask_path):
        print(f"错误：掩码文件不存在 {mask_path}")
        return None
    
    print(f"正在加载掩码文件: {mask_path}")
    print(f"文件大小: {os.path.getsize(mask_path)} 字节")
    
    # 尝试读取掩码文件
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"错误：OpenCV无法读取掩码文件 {mask_path}")
        print("可能的原因：")
        print("1. 文件格式不支持")
        print("2. 文件损坏")
        print("3. 文件路径包含特殊字符")
        
        # 尝试使用PIL读取
        try:
            from PIL import Image
            pil_image = Image.open(mask_path)
            mask = np.array(pil_image.convert('L'))
            print("使用PIL成功读取掩码文件")
            return mask
        except Exception as e:
            print(f"PIL也无法读取文件: {e}")
            return None
    
    print(f"掩码尺寸: {mask.shape}")
    print(f"掩码数据类型: {mask.dtype}")
    print(f"掩码值范围: {mask.min()} - {mask.max()}")
    
    return mask

def apply_mask_to_image(image_path, mask):
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return None

    print(f"图像尺寸: {image.shape}")
    print(f"掩码尺寸: {mask.shape}")

    if mask.shape[:2] != image.shape[:2]:
        print("警告：掩码尺寸与图像不匹配，将调整掩码尺寸")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # 创建一个全白背景图像
    white_bg = np.full_like(image, 255)  # (255,255,255) 白色背景

    # 将掩码区域内容保留，其余为白色
    masked_image = np.where(mask[:, :, None] == 255, image, white_bg)

    return masked_image

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='OCR处理脚本，支持掩码功能')
    parser.add_argument('--mask', type=str, default='', 
                       help='PNG格式的掩码文件路径（可选）')
    parser.add_argument('--input', type=str, default='images/image1.jpg',
                       help='输入图像路径')
    parser.add_argument('--output', type=str, default='ocr_output/output',
                       help='输出目录')
    parser.add_argument('--lang', type=str, default='ch',
                       help='OCR语言设置 (ch:中文, en:英文, japan:日文, korean:韩文, ru:俄文，fr:法文)')
    
    args = parser.parse_args()
    
    # 初始化 PaddleOCR 实例
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=args.lang)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误：输入图像文件不存在 {args.input}")
        return
    
    # 加载掩码（如果提供）
    mask = None
    if args.mask:
        mask = load_mask_from_png(args.mask)
        if mask is None:
            print("警告：掩码加载失败，将处理整个图像")
    
    # 处理图像
    if mask is not None:
        # 使用掩码
        print(f"使用掩码文件: {args.mask}")
        masked_image = apply_mask_to_image(args.input, mask)
        if masked_image is None:
            return
        
        # 保存掩码图像
        os.makedirs(args.output, exist_ok=True)
        masked_path = os.path.join(args.output, "masked_image.jpg")
        cv2.imwrite(masked_path, masked_image)
        print(f"掩码图像已保存到: {masked_path}")
        
        # 对掩码图像执行OCR
        print("正在对掩码区域进行OCR识别...")
        result = ocr.predict(input=masked_path)
    else:
        # 不使用掩码，处理整个图像
        print("处理整个图像...")
        result = ocr.predict(input=args.input)
    
    # 保存结果
    os.makedirs(args.output, exist_ok=True)
    for res in result:
        res.print()
        res.save_to_img(args.output)
        res.save_to_json(args.output)
    
    print(f"OCR结果已保存到: {args.output}")

if __name__ == "__main__":
    main()