# SAM 2: 图像和视频中的通用分割模型

**[Meta AI, FAIR](https://ai.meta.com/research/)**

[Nikhila Ravi](https://nikhilaravi.com/), [Valentin Gabeur](https://gabeur.github.io/), [Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en), [Ronghang Hu](https://ronghanghu.com/), [Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en), [Tengyu Ma](https://scholar.google.com/citations?user=VeTSl0wAAAAJ&hl=en), [Haitham Khedr](https://hkhedr.com/), [Roman Rädle](https://scholar.google.de/citations?user=Tpt57v0AAAAJ&hl=en), [Chloe Rolland](https://scholar.google.com/citations?hl=fr&user=n-SnMhoAAAAJ), [Laura Gustafson](https://scholar.google.com/citations?user=c8IpF9gAAAAJ&hl=en), [Eric Mintun](https://ericmintun.github.io/), [Junting Pan](https://junting.github.io/), [Kalyan Vasudev Alwala](https://scholar.google.co.in/citations?user=m34oaWEAAAAJ&hl=en), [Nicolas Carion](https://www.nicolascarion.com/), [Chao-Yuan Wu](https://chaoyuan.org/), [Ross Girshick](https://www.rossgirshick.info/), [Piotr Dollár](https://pdollar.github.io/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/)

[[`论文`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`项目主页`](https://ai.meta.com/sam2)] [[`在线演示`](https://sam2.metademolab.com/)] [[`数据集`](https://ai.meta.com/datasets/segment-anything-video)] [[`博客`](https://ai.meta.com/blog/segment-anything-2)] [[`引用`](#引用-sam-2)]

![SAM 2 架构](assets/model_diagram.png?raw=true)

**Segment Anything Model 2 (SAM 2)** 是一个基础模型，用于解决图像和视频中的可提示视觉分割问题。我们将SAM扩展到视频领域，将图像视为单帧视频。模型设计采用简单的transformer架构，具有流式内存，支持实时视频处理。我们构建了一个模型在环数据引擎，通过用户交互改进模型和数据，收集了[**SA-V数据集**](https://ai.meta.com/datasets/segment-anything-video)，这是迄今为止最大的视频分割数据集。在我们的数据上训练的SAM 2在广泛的任务和视觉领域都表现出色。

![SA-V 数据集](assets/sa_v_dataset.jpg?raw=true)

## 最新更新

**2024年12月11日 -- 完整模型编译实现重大VOS加速和新的`SAM2VideoPredictor`以更好地处理多对象跟踪**

- 我们现在支持整个SAM 2模型的`torch.compile`，可以通过在`build_sam2_video_predictor`中设置`vos_optimized=True`来开启，这将显著提升VOS推理速度。
- 我们更新了`SAM2VideoPredictor`的实现，支持独立的每对象推理，允许我们放宽多对象跟踪的提示假设，并在跟踪开始后添加新对象。
- 详情请参见[`RELEASE_NOTES.md`](RELEASE_NOTES.md)。

**2024年9月30日 -- SAM 2.1 开发者套件发布（新检查点、训练代码、Web演示）**

- 发布了一套改进的模型检查点（标记为**SAM 2.1**）。详情请参见[模型描述](#模型描述)。
  * 要使用新的SAM 2.1检查点，您需要从此仓库获取最新的模型代码。如果您已安装此仓库的早期版本，请先通过`pip uninstall SAM-2`卸载之前的版本，从此仓库拉取最新代码（使用`git pull`），然后按照下面的[安装](#安装)说明重新安装。
- 训练（和微调）代码已发布。请参见[`training/README.md`](training/README.md)了解如何开始。
- SAM 2 Web演示的前端+后端代码已发布。详情请参见[`demo/README.md`](demo/README.md)。

## 🚀 快速启动

### 启动我们的SAM2 OCR翻译系统

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动Web应用
python run.py

# 3. 访问Web界面
# 本地访问: http://localhost:5000
# 网络访问: http://[您的IP]:5000
```

### 直接启动Flask应用

```bash
# 直接运行app.py
python app.py

# 访问地址: http://localhost:5000
```

### 启动原有SAM2项目

```bash
# 1. 克隆项目
git clone https://github.com/facebookresearch/sam2.git && cd sam2

# 2. 安装依赖
pip install -e .
pip install -e ".[notebooks]"

# 3. 安装OCR依赖
cd OCR && pip install -r requirements.txt && cd ..

# 4. 下载模型检查点
cd checkpoints && ./download_ckpts.sh && cd ..

# 5. 启动Web演示界面
cd demo && docker-compose up
```

### 分步启动

#### 启动SAM 2核心功能
```bash
# 安装SAM 2
pip install -e .

# 下载模型
cd checkpoints && ./download_ckpts.sh && cd ..

# 运行示例
jupyter notebook notebooks/image_predictor_example.ipynb
```

#### 启动OCR功能
```bash
# 安装OCR依赖
cd OCR && pip install -r requirements.txt && cd ..

# 运行OCR
python OCR/ocr_processor.py --image path/to/image.jpg
```

#### 启动Web演示
```bash
# 启动Web界面
cd demo && docker-compose up

# 访问 http://localhost:3000
```

## 项目概述

本项目结合了SAM 2（通用分割模型2）和OCR（光学字符识别）功能，提供了一个全面的计算机视觉解决方案。主要功能包括：

- **图像和视频分割**：使用SAM 2进行对象分割、交互式分割、自动掩码生成
- **OCR文本提取**：从图像中提取文本，支持多语言识别
- **Web演示界面**：提供用户友好的交互式工具
- **训练和微调**：支持自定义数据集训练和模型优化

## 安装

SAM 2需要先安装才能使用。代码要求`python>=3.10`，以及`torch>=2.5.1`和`torchvision>=0.20.1`。请按照[这里](https://pytorch.org/get-started/locally/)的说明安装PyTorch和TorchVision依赖。您可以在GPU机器上使用以下命令安装SAM 2：

```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```

如果您在Windows上安装，强烈建议使用[Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install)和Ubuntu。

要使用SAM 2预测器并运行示例notebook，需要安装`jupyter`和`matplotlib`：

```bash
pip install -e ".[notebooks]"
```

注意：
1. 建议通过[Anaconda](https://www.anaconda.com/)为此安装创建新的Python环境，并通过`pip`安装PyTorch 2.5.1（或更高版本），按照https://pytorch.org/的说明。如果您当前环境中的PyTorch版本低于2.5.1，上述安装命令将尝试使用`pip`将其升级到最新的PyTorch版本。
2. 上述步骤需要使用`nvcc`编译器编译自定义CUDA内核。如果您的机器上还没有，请安装与PyTorch CUDA版本匹配的[CUDA工具包](https://developer.nvidia.com/cuda-toolkit-archive)。
3. 如果在安装过程中看到类似`Failed to build the SAM 2 CUDA extension`的消息，您可以忽略它，仍然可以使用SAM 2（某些后处理功能可能有限，但在大多数情况下不会影响结果）。

有关潜在问题和解决方案的常见问题，请参见[`INSTALL.md`](./INSTALL.md)。

## 快速开始

### 下载模型检查点

首先，我们需要下载模型检查点。可以通过运行以下命令下载所有模型检查点：

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

或单独下载：

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

（注意：这些是标记为SAM 2.1的改进检查点；详情请参见[模型描述](#模型描述）。）

然后，SAM 2可以用于图像和视频预测，如下所示：

### 图像预测

SAM 2具有[SAM](https://github.com/facebookresearch/segment-anything)在静态图像上的所有功能，我们提供与SAM非常相似的图像预测API，用于图像用例。`SAM2ImagePredictor`类为图像提示提供了简单的接口。

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<您的图像>)
    masks, _, _ = predictor.predict(<输入提示>)
```

有关静态图像用例的示例，请参考[image_predictor_example.ipynb](./notebooks/image_predictor_example.ipynb)（也可在Colab中查看[这里](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)）。

SAM 2还支持图像的自动掩码生成，就像SAM一样。有关图像自动掩码生成的详细信息，请参见[automatic_mask_generator_example.ipynb](./notebooks/automatic_mask_generator_example.ipynb)（也可在Colab中查看[这里](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb)）。

### 视频预测

对于视频中的可提示分割和跟踪，我们提供了一个视频预测器，具有API，例如添加提示并在整个视频中传播掩码。SAM 2支持视频上的多对象推理，并使用推理状态来跟踪每个视频中的交互。

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<您的视频>)

    # 添加新提示并立即在同一帧上获得输出
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <您的提示>):

    # 传播提示以在整个视频中获得掩码
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

有关如何添加点击或框提示、进行细化以及在视频中跟踪多个对象的详细信息，请参考[video_predictor_example.ipynb](./notebooks/video_predictor_example.ipynb)中的示例（也可在Colab中查看[这里](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb)）。

## 从🤗 Hugging Face加载

或者，模型也可以从[Hugging Face](https://huggingface.co/models?search=facebook/sam2)加载（需要`pip install huggingface_hub`）。

对于图像预测：

```python
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<您的图像>)
    masks, _, _ = predictor.predict(<输入提示>)
```

对于视频预测：

```python
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<您的视频>)

    # 添加新提示并立即在同一帧上获得输出
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <您的提示>):

    # 传播提示以在整个视频中获得掩码
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

## OCR功能

本项目还包括OCR（光学字符识别）功能，用于从图像中提取文本。OCR模块提供了处理图像和提取文本内容的工具。

### OCR特性

- **文本识别**：从各种图像格式中提取文本
- **图像处理**：自动图像大小调整和优化
- **多语言支持**：支持包括中文和英文在内的多种语言
- **输出格式**：生成包含文本和坐标的结构化JSON输出

### OCR安装

安装OCR依赖：

```bash
cd OCR
pip install -r requirements.txt
```

### OCR使用方法

#### 基本OCR处理

```python
from OCR.ocr_processor import OCRProcessor

# 初始化OCR处理器
processor = OCRProcessor()

# 处理图像
result = processor.process_image("path/to/image.jpg")

# 获取提取的文本
print(result.text)
print(result.confidence)
```

#### 图像大小调整

OCR模块包含一个PNG图像大小调整脚本，可以自动调整图像大小：

```bash
# 基本用法
python OCR/resize_png.py input.png

# 指定输出文件
python OCR/resize_png.py input.png -o output.png

# 指定最大文件大小（MB）
python OCR/resize_png.py input.png -s 2.0

# 完整示例
python OCR/resize_png.py input.png -o output.png -s 3.5
```

更多详细信息，请参见[OCR README](OCR/README.md)。

## 项目使用指南

本项目结合了SAM 2（通用分割模型2）和OCR功能，提供了一个全面的计算机视觉解决方案。以下是不同组件的使用方法：

### 1. 图像和视频分割（SAM 2）

**使用场景：**
- 图像和视频中的对象分割
- 交互式分割提示
- 自动掩码生成
- 视频中的多对象跟踪

**快速开始：**
```bash
# 安装依赖
pip install -e .

# 下载模型检查点
cd checkpoints && ./download_ckpts.sh && cd ..

# 运行示例
jupyter notebook notebooks/image_predictor_example.ipynb
jupyter notebook notebooks/video_predictor_example.ipynb
```

### 2. OCR文本提取

**使用场景：**
- 从图像中提取文本
- 处理文档和截图
- 多语言文本识别
- 生成结构化文本数据

**快速开始：**
```bash
# 安装OCR依赖
cd OCR && pip install -r requirements.txt && cd ..

# 在图像上运行OCR
python OCR/ocr_processor.py --image path/to/image.jpg
```

### 3. Web演示界面

**使用场景：**
- SAM 2的交互式Web界面
- 实时视频处理
- 用户友好的分割工具

**快速开始：**
```bash
# 启动Web演示
cd demo
docker-compose up
# 或按照demo/README.md进行详细设置
```

### 4. 训练和微调

**使用场景：**
- 自定义数据集训练
- 特定领域的模型微调
- 性能优化

**快速开始：**
```bash
# 按照训练指南
cd training
# 查看training/README.md了解详细说明
```

### 5. 完整工作流示例

以下是结合多个功能的典型工作流：

```python
# 1. 使用SAM 2在图像中分割对象
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
masks, _, _ = predictor.predict(prompts)

# 2. 使用OCR从同一图像中提取文本
from OCR.ocr_processor import OCRProcessor
ocr = OCRProcessor()
text_result = ocr.process_image("image.jpg")

# 3. 结合结果进行综合分析
print(f"找到{len(masks)}个对象和{len(text_result.text)}个文本元素")
```

## 模型描述

### SAM 2.1 检查点

下表显示了2024年9月29日发布的改进SAM 2.1检查点。
|      **模型**       | **大小 (M)** |    **速度 (FPS)**     | **SA-V测试 (J&F)** | **MOSE验证 (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2.1_hiera_tiny <br /> ([配置](sam2/configs/sam2.1/sam2.1_hiera_t.yaml), [检查点](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt))    |     38.9     |          91.2          |        76.5         |        71.8        |       77.3        |
|   sam2.1_hiera_small <br /> ([配置](sam2/configs/sam2.1/sam2.1_hiera_s.yaml), [检查点](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt))   |      46      |          84.8          |        76.6         |        73.5        |       78.3        |
| sam2.1_hiera_base_plus <br /> ([配置](sam2/configs/sam2.1/sam2.1_hiera_b+.yaml), [检查点](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)) |     80.8     |        64.1          |        78.2         |        73.7        |       78.2        |
|   sam2.1_hiera_large <br /> ([配置](sam2/configs/sam2.1/sam2.1_hiera_l.yaml), [检查点](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt))   |    224.4     |          39.5          |        79.5         |        74.6        |       80.6        |

### SAM 2 检查点

2024年7月29日发布的先前SAM 2检查点如下：

|      **模型**       | **大小 (M)** |    **速度 (FPS)**     | **SA-V测试 (J&F)** | **MOSE验证 (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2_hiera_tiny <br /> ([配置](sam2/configs/sam2/sam2_hiera_t.yaml), [检查点](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt))   |     38.9     |          91.5          |        75.0         |        70.9        |       75.3        |
|   sam2_hiera_small <br /> ([配置](sam2/configs/sam2/sam2_hiera_s.yaml), [检查点](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt))   |      46      |          85.6          |        74.9         |        71.5        |       76.4        |
| sam2_hiera_base_plus <br /> ([配置](sam2/configs/sam2/sam2_hiera_b+.yaml), [检查点](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)) |     80.8     |     64.8    |        74.7         |        72.8        |       75.8        |
|   sam2_hiera_large <br /> ([配置](sam2/configs/sam2/sam2_hiera_l.yaml), [检查点](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt))   |    224.4     | 39.7 |        76.0         |        74.6        |       79.8        |

速度在A100上测量，使用`torch 2.5.1, cuda 12.4`。有关基准测试的示例，请参见`benchmark.py`（编译所有模型组件）。仅编译图像编码器可以更灵活，也可以提供（较小的）加速（在配置中设置`compile_image_encoder: True`）。

## 项目结构

```
sam2/
├── sam2/                    # 核心SAM 2实现
├── OCR/                     # OCR功能
├── demo/                    # Web演示界面
├── training/                # 训练和微调代码
├── notebooks/               # 示例notebook
├── checkpoints/             # 模型检查点
└── tools/                   # 实用脚本
```

## 系统要求

- **Python**: 3.10+
- **PyTorch**: 2.5.1+
- **CUDA**: 11.8+（用于GPU加速）
- **内存**: 8GB+ RAM（推荐16GB+）
- **存储**: 10GB+用于模型和数据集

## 性能优化建议

1. **GPU加速**: 使用CUDA兼容GPU获得最佳性能
2. **模型选择**: 根据您的需求选择合适的模型大小：
   - Tiny: 快速推理，较低精度
   - Large: 更高精度，较慢推理
3. **批处理**: 批量处理多个图像/视频
4. **内存管理**: 推理时使用`torch.inference_mode()`
5. **图像优化**: 对大型图像使用OCR大小调整脚本

## 通用分割视频数据集

详情请参见[sav_dataset/README.md](sav_dataset/README.md)。

## 训练SAM 2

您可以在图像、视频或两者的自定义数据集上训练或微调SAM 2。请查看训练[README](training/README.md)了解如何开始。

## SAM 2的Web演示

我们已发布SAM 2 Web演示的前端+后端代码（类似于https://sam2.metademolab.com/demo的本地可部署版本）。详情请参见Web演示[README](demo/README.md)。

## 许可证

SAM 2模型检查点、SAM 2演示代码（前端和后端）和SAM 2训练代码根据[Apache 2.0](./LICENSE)许可，但SAM 2演示代码中使用的[Inter Font](https://github.com/rsms/inter?tab=OFL-1.1-1-ov-file)和[Noto Color Emoji](https://github.com/googlefonts/noto-emoji)根据[SIL开放字体许可证，版本1.1](https://openfontlicense.org/open-font-license-official-text/)提供。

## 贡献

请参见[贡献指南](CONTRIBUTING.md)和[行为准则](CODE_OF_CONDUCT.md)。

## 贡献者

SAM 2项目在众多贡献者的帮助下得以实现（按字母顺序）：

Karen Bergan, Daniel Bolya, Alex Bosenberg, Kai Brown, Vispi Cassod, Christopher Chedeau, Ida Cheng, Luc Dahlin, Shoubhik Debnath, Rene Martinez Doehner, Grant Gardner, Sahir Gomez, Rishi Godugu, Baishan Guo, Caleb Ho, Andrew Huang, Somya Jain, Bob Kamma, Amanda Kallet, Jake Kinney, Alexander Kirillov, Shiva Koduvayur, Devansh Kukreja, Robert Kuo, Aohan Lin, Parth Malani, Jitendra Malik, Mallika Malhotra, Miguel Martin, Alexander Miller, Sasha Mitts, William Ngan, George Orlin, Joelle Pineau, Kate Saenko, Rodrick Shepard, Azita Shokrpour, David Soofian, Jonathan Torres, Jenny Truong, Sagar Vaze, Meng Wang, Claudette Ward, Pengchuan Zhang.

第三方代码：我们使用从[`cc_torch`](https://github.com/zsef123/Connected_components_PyTorch)改编的基于GPU的连通分量算法（其许可证在[`LICENSE_cctorch`](./LICENSE_cctorch)中）作为掩码预测的可选后处理步骤。

## 引用SAM 2

如果您在研究中使用SAM 2或SA-V数据集，请使用以下BibTeX条目。

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
