# OCR API 测试脚本使用说明

本目录包含了基于星河API的OCR功能测试脚本。

## 文件说明

- `ocr_api_test.py` - 完整的OCR API测试脚本，包含详细的错误处理和结果保存功能
- `simple_ocr_test.py` - 简化版测试脚本，更接近原始代码示例
- `config.example` - 配置文件示例
- `requirements.txt` - Python依赖包列表

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置环境变量

1. 复制配置文件示例：
```bash
cp config.example config.env
```

2. 编辑 `config.env` 文件，填入您的配置：
```
OCR_API_URL=https://jdebf6gbs2x6g0y4.aistudio-hub.baidu.com/ocr
OCR_TOKEN=your_access_token_here
```

**注意**: 请前往 https://aistudio.baidu.com/index/accessToken 获取您的访问令牌

## 使用方法

### 方法1: 使用简化版脚本
```bash
python simple_ocr_test.py
```

### 方法2: 使用完整版脚本
```bash
python ocr_api_test.py
```

## 功能特性

### 简化版脚本 (`simple_ocr_test.py`)
- 基于您提供的原始代码
- 支持从环境变量读取配置
- 自动测试多个图像文件
- 简洁的错误处理

### 完整版脚本 (`ocr_api_test.py`)
- 面向对象的API封装
- 详细的错误处理和日志输出
- 自动保存结果到JSON文件
- 支持PDF和图像文件
- 完整的类型注解

## 测试文件

脚本会自动测试以下文件：
- `images/2.jpg`
- `images/原图-min.png`

如果这些文件不存在，脚本会跳过并显示警告信息。

## 输出结果

- 控制台会显示识别到的文本内容
- 完整版脚本会保存两种格式的结果：
  - `*_raw.json` - 星河API的原始返回格式
  - `*.json` - 转换后的标准格式（与 `image1_res.json` 格式一致）
- 结果保存在 `ocr_output/` 目录下

## 错误处理

脚本包含以下错误处理：
- 文件不存在检查
- 网络请求异常处理
- API响应格式验证
- 环境变量配置检查

## 注意事项

1. 确保您有有效的星河API访问令牌
2. 测试图像文件应放在 `images/` 目录下
3. 网络连接正常，能够访问星河API
4. 图像文件大小应在API限制范围内

## 故障排除

### 常见问题

1. **"请设置OCR_TOKEN环境变量"**
   - 检查 `config.env` 文件是否存在
   - 确认 `OCR_TOKEN` 已正确设置

2. **"文件不存在"**
   - 检查测试文件是否在正确位置
   - 确认文件路径正确

3. **网络请求失败**
   - 检查网络连接
   - 确认API地址正确
   - 验证访问令牌有效

4. **响应格式错误**
   - 检查API响应内容
   - 确认API版本兼容性 