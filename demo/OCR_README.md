# OCR文字替换功能

## 功能概述

OCR文字替换功能是一个集成的图像处理工具，可以：

1. **自动识别图像中的文字** - 使用PaddleOCR进行文字检测
2. **移除原文字** - 使用inpaint技术移除图像中的原文字
3. **添加翻译文字** - 支持文字图层编辑，可调整位置、大小、颜色等
4. **生成最终图像** - 将文字图层合并到图像中

## 主要特性

### 1. 智能对比度算法
- 使用基于相对亮度的对比度算法
- 自动选择黑色或白色文字以确保最佳可读性
- 支持自定义文字颜色

### 2. 文字图层编辑
- 可拖拽调整文字位置
- 可调整文字大小
- 可修改文字颜色
- 可编辑翻译内容
- 可控制图层可见性

### 3. 实时预览
- 在编辑过程中实时预览效果
- 支持图层选择和高亮显示

## 技术实现

### 后端API

#### `/api/ocr/detect` - 文字检测
```javascript
POST /api/ocr/detect
{
  "image": "base64_encoded_image"
}
```

#### `/api/ocr/process` - 图像处理
```javascript
POST /api/ocr/process
{
  "image": "base64_encoded_image",
  "translation_mapping": {
    "original_text": "translated_text"
  }
}
```

#### `/api/ocr/generate` - 生成最终图像
```javascript
POST /api/ocr/generate
{
  "image": "base64_encoded_image",
  "text_layers": [
    {
      "id": "layer_1",
      "original_text": "原文",
      "translated_text": "翻译",
      "box": [x1, y1, x2, y2],
      "text_color": [r, g, b],
      "font_size": 20,
      "visible": true
    }
  ]
}
```

#### `/api/ocr/translate` - 翻译文字
```javascript
POST /api/ocr/translate
{
  "text": "原文",
  "target_lang": "zh"
}
```

### 前端组件

#### OCRProcessor
- 主要的OCR处理组件
- 处理图像上传和OCR处理流程
- 管理文字图层状态

#### OCREditor
- 文字图层编辑器
- 支持拖拽、调整大小、修改属性
- 实时预览功能

## 使用方法

### 1. 启动服务
```bash
# 启动后端服务
cd demo/backend
python server/app.py

# 启动前端服务
cd demo/frontend
npm install
npm run dev
```

### 2. 访问OCR功能
- 打开浏览器访问 `http://localhost:3000`
- 点击导航栏中的"OCR文字替换"
- 或者直接访问 `http://localhost:3000/ocr`

### 3. 使用流程
1. **上传图像** - 点击上传区域或拖拽图像文件
2. **自动处理** - 系统自动识别文字并移除原文字
3. **编辑图层** - 在文字编辑器中调整文字属性
4. **生成结果** - 点击"生成最终图像"查看结果
5. **下载图像** - 下载处理后的图像

## 配置说明

### 翻译映射
在 `ocr_api.py` 中可以配置翻译映射：

```python
translation_mapping = {
    'Усиленнаяверсия': '加强版',
    'Зкстракт трав': '草本提取物',
    'Без онемения': '无麻木感',
    # ... 更多映射
}
```

### 字体配置
系统会尝试加载以下字体：
1. `C:/Windows/Fonts/simhei.ttf` - 黑体
2. `C:/Windows/Fonts/msyh.ttc` - 微软雅黑
3. 默认字体（如果上述字体不可用）

## 依赖要求

### 后端依赖
```bash
pip install opencv-python
pip install pillow
pip install paddlepaddle
pip install paddleocr
pip install flask
pip install flask-cors
```

### 前端依赖
```bash
npm install @stylexjs/stylex
npm install react-router-dom
```

## 注意事项

1. **图像格式** - 支持常见的图像格式（JPEG、PNG等）
2. **文字识别** - 主要支持英文和俄文识别
3. **翻译功能** - 目前使用简单的映射，可集成真实翻译API
4. **性能考虑** - 大图像处理可能需要较长时间
5. **浏览器兼容性** - 需要支持Canvas API的现代浏览器

## 扩展功能

### 1. 集成真实翻译API
可以替换 `ocr_api.py` 中的翻译函数：

```python
def translate_text_with_api(text, target_lang='zh'):
    # 集成Google Translate、百度翻译等API
    pass
```

### 2. 支持更多语言
修改PaddleOCR配置：

```python
self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 中文
```

### 3. 添加更多编辑功能
- 文字旋转
- 文字特效（阴影、描边等）
- 批量操作
- 撤销/重做功能

## 故障排除

### 常见问题

1. **OCR识别失败**
   - 检查PaddleOCR是否正确安装
   - 确认图像质量足够清晰

2. **字体显示问题**
   - 检查系统字体文件是否存在
   - 尝试使用默认字体

3. **图像处理失败**
   - 检查图像格式是否支持
   - 确认图像大小合理

4. **前端显示异常**
   - 检查浏览器控制台错误
   - 确认所有依赖已正确安装

## 开发计划

- [ ] 支持更多语言识别
- [ ] 集成真实翻译API
- [ ] 添加文字特效
- [ ] 支持批量处理
- [ ] 添加历史记录功能
- [ ] 优化性能
- [ ] 添加更多导出格式 