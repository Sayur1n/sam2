# PNG图片大小调整脚本

这个脚本可以将PNG图片调整到指定的大小（默认小于4MB）。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法
```bash
python resize_png.py input.png
```

### 指定输出文件
```bash
python resize_png.py input.png -o output.png
```

### 指定最大文件大小
```bash
python resize_png.py input.png -s 2.0
```

### 完整参数示例
```bash
python resize_png.py input.png -o output.png -s 3.5
```

## 参数说明

- `input`: 输入PNG图片路径（必需）
- `-o, --output`: 输出图片路径（可选，默认添加_resized后缀）
- `-s, --size`: 最大文件大小，单位MB（可选，默认4.0）

## 功能特点

- 自动调整图片尺寸，确保文件大小小于指定值
- 保持图片宽高比
- 使用高质量的重采样算法
- 支持PNG格式的无损压缩优化
- 防止无限循环，确保程序正常结束

## 示例

```bash
# 将图片调整到小于2MB
python resize_png.py large_image.png -s 2.0

# 指定输出文件名
python resize_png.py image.png -o small_image.png -s 1.5
``` 