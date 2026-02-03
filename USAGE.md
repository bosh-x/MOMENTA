# EMNLP_MOMENTA_All_DemoCode 使用说明

## 修改说明

代码已经重构，支持：
1. ✅ 安全导入（不会自动执行训练代码）
2. ✅ 实时特征提取（无需下载预提取的 .pt 文件）
3. ✅ 预提取特征支持（如果你有特征文件）

## 使用方式

### 方式 1：直接运行主脚本（推荐用于训练）

```bash
python EMNLP_MOMENTA_All_DemoCode.py
```

这会：
- 加载所有模型（CLIP、VGG16、SentenceTransformer）
- 创建数据集（使用实时特征提取）
- 开始训练
- 保存模型和结果

### 方式 2：作为模块导入（推荐用于自定义使用）

```python
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision import models
from PIL import Image
import EMNLP_MOMENTA_All_DemoCode as demo

# 1. 初始化模型
demo.clip_model = torch.jit.load("clip_model.pt").cuda().eval()
input_resolution = demo.clip_model.input_resolution.item()

demo.preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
])
demo.tokenizer = demo.SimpleTokenizer()

model_vgg_pretrained = models.vgg16(pretrained=True)
demo.model_vgg = demo.FeatureExtractor(model_vgg_pretrained)
demo.model_vgg = demo.model_vgg.to(device)

from sentence_transformers import SentenceTransformer
demo.model_sent_trans = SentenceTransformer('paraphrase-distilroberta-base-v1')

# 2. 创建数据集（实时提取特征）
dataset = demo.HarmemeMemesDatasetAug2(
    data_path="HarMeme_V1/Annotations/train.jsonl",
    img_dir="HarMeme_V1/images_flat",
    split_flag='train',
    use_preextracted=False  # 使用实时提取
)

# 3. 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 方式 3：使用预提取特征（如果你有 .pt 文件）

```python
import torch
import EMNLP_MOMENTA_All_DemoCode as demo

# 1. 加载预提取的特征文件
demo.train_ROI = torch.load("path_to_features/harmeme_pol_train_ROI.pt")
demo.train_ENT = torch.load("path_to_features/harmeme_pol_train_ent.pt")

# 2. 创建数据集（使用预提取特征）
dataset = demo.HarmemeMemesDatasetAug2(
    data_path="HarMeme_V1/Annotations/train.jsonl",
    img_dir="HarMeme_V1/images_flat",
    split_flag='train',
    use_preextracted=True  # 使用预提取特征
)

# 3. 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

## 参数说明

### HarmemeMemesDatasetAug2 参数

- `data_path` (str): JSONL 数据文件路径
- `img_dir` (str): 图片文件夹路径
- `split_flag` (str): 'train', 'val', 或 'test'
- `use_preextracted` (bool, default=False):
  - `False`: 实时提取特征（需要先初始化模型）
  - `True`: 使用预加载的特征（需要先加载 train_ROI/val_ROI/test_ROI 和 train_ENT/val_ENT/test_ENT）

## 数据要求

### 实时提取模式所需的数据列（JSONL）

必需列：
- `id`: 样本ID
- `image`: 图片文件名
- `text`: 文本内容
- `labels`: 标签列表（可选，用于训练）

可选列（用于更好的特征提取）：
- `bbdict`: 边界框信息（用于 ROI 特征）
- `ent`: 实体列表（用于实体特征）

### 预提取模式所需文件

需要从 Google Drive 下载：
- ROI features: https://drive.google.com/file/d/1KRAJcTme4tmbuNXLQ72NTfnQf3x2YQT_/view
- Entity features: https://drive.google.com/file/d/1KBltp_97CJIOcrxln9VbDfoKxbVQKcVN/view

## 依赖项

```bash
# PyTorch
pip install torch torchvision

# CLIP 依赖
pip install ftfy regex

# 其他依赖
pip install sentence-transformers opencv-python pandas numpy scikit-learn tqdm
```

## 文件下载

- CLIP BPE 词表: https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz
- CLIP 模型: https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
  - 下载后重命名为 `clip_model.pt`

## 常见问题

### Q: NameError: name 'train_ROI' is not defined
A: 使用 `use_preextracted=False` 来实时提取特征，或者先加载预提取的特征文件。

### Q: 实时提取太慢怎么办？
A: 可以：
1. 减小 batch_size
2. 使用预提取的特征文件
3. 使用 GPU 加速

### Q: 没有 bbdict 或 ent 列怎么办？
A: 代码会自动处理：
- 没有 `bbdict`: 使用整张图片的中心裁剪提取 VGG 特征
- 没有 `ent`: 使用整段文本提取 SentenceTransformer 特征

## 性能建议

**实时提取模式：**
- 优点：无需下载大型特征文件，灵活性高
- 缺点：第一次训练较慢
- 适用：数据集较小、硬件资源充足、需要调整特征提取逻辑

**预提取模式：**
- 优点：训练速度快
- 缺点：需要下载特征文件，占用磁盘空间
- 适用：数据集较大、重复训练多次、硬件资源有限
