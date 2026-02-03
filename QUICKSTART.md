# MOMENTA 快速开始指南

## 📊 你的数据集信息

- **数据集**: Harm-C (COVID-19 memes)
- **位置**:
  - JSONL: `HarMeme_V1/Annotations/Harm-C/`
  - Images: `HarMeme_V1/images_flat/`
- **样本数量**:
  - Train: 3,013 样本
  - Val: 177 样本
  - Test: 354 样本
  - **Total: 3,544 样本**

## ⏱️ 时间估算

| 操作 | GPU | CPU |
|------|-----|-----|
| 特征提取（一次） | ~6-12 分钟 | ~1-2 小时 |
| 训练（使用预提取特征） | ~5-10 分钟 | ~30-60 分钟 |
| 训练（实时提取） | ~15-25 分钟 | ~2-3 小时 |

## 🚀 推荐流程（三步走）

### 步骤 1️⃣: 快速测试（确保一切正常）

```bash
# 测试数据集加载和模型
python test_dataset_quick.py
```

**预期输出**：
```
✓ Using device: cuda
✓ All models loaded successfully
✓ Dataset created: 3013 samples
✓ Sample loaded successfully!
✓ All tests passed!
```

**如果测试失败**，检查是否缺少文件：
```bash
# 下载 CLIP 模型
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -O clip_model.pt

# 下载 BPE 词表
wget https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz
```

### 步骤 2️⃣: 提取并保存特征（约 6-12 分钟）

```bash
# 提取特征并保存
python extract_and_save_features.py
```

**预期输出**：
```
Dataset Configuration: Harm-C (COVID-19)
...
Processing TRAIN set
  Extracting train: 100%|██████████| 3013/3013
  ✓ Extracted ROI features: torch.Size([3013, 4096])
  ✓ Extracted ENT features: torch.Size([3013, 768])
  ✓ Saved to: ./extracted_features/Harm-C/train_ROI.pt

Processing VAL set
  ...

Processing TEST set
  ...

✓ All done! Total time: X.XX minutes
```

**保存的文件**：
```
./extracted_features/Harm-C/
├── train_ROI.pt  (~47 MB)
├── train_ENT.pt  (~9 MB)
├── val_ROI.pt    (~3 MB)
├── val_ENT.pt    (~0.5 MB)
├── test_ROI.pt   (~5 MB)
└── test_ENT.pt   (~1 MB)

Total: ~66 MB
```

### 步骤 3️⃣: 使用预提取特征训练（快！）

创建训练脚本 `train.py`：

```python
import torch
from torch.utils.data import DataLoader
import EMNLP_MOMENTA_All_DemoCode as demo

# 加载预提取的特征
print("Loading pre-extracted features...")
demo.train_ROI = torch.load('./extracted_features/Harm-C/train_ROI.pt')
demo.train_ENT = torch.load('./extracted_features/Harm-C/train_ENT.pt')
demo.val_ROI = torch.load('./extracted_features/Harm-C/val_ROI.pt')
demo.val_ENT = torch.load('./extracted_features/Harm-C/val_ENT.pt')
demo.test_ROI = torch.load('./extracted_features/Harm-C/test_ROI.pt')
demo.test_ENT = torch.load('./extracted_features/Harm-C/test_ENT.pt')

# 还需要初始化 CLIP（用于编码）
demo.clip_model = torch.jit.load("clip_model.pt").cuda().eval()
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from PIL import Image
demo.preprocess = Compose([
    Resize(demo.clip_model.input_resolution.item(), interpolation=Image.BICUBIC),
    CenterCrop(demo.clip_model.input_resolution.item()),
    ToTensor()
])
demo.tokenizer = demo.SimpleTokenizer()

# 创建数据集（使用预提取特征，快！）
print("Creating datasets...")
train_dataset = demo.HarmemeMemesDatasetAug2(
    data_path='HarMeme_V1/Annotations/Harm-C/train.jsonl',
    img_dir='HarMeme_V1/images_flat',
    split_flag='train',
    use_preextracted=True  # 关键！使用预提取特征
)

val_dataset = demo.HarmemeMemesDatasetAug2(
    data_path='HarMeme_V1/Annotations/Harm-C/val.jsonl',
    img_dir='HarMeme_V1/images_flat',
    split_flag='val',
    use_preextracted=True
)

test_dataset = demo.HarmemeMemesDatasetAug2(
    data_path='HarMeme_V1/Annotations/Harm-C/test.jsonl',
    img_dir='HarMeme_V1/images_flat',
    split_flag='test',
    use_preextracted=True
)

# 创建 DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"✓ Train: {len(train_dataset)} samples")
print(f"✓ Val: {len(val_dataset)} samples")
print(f"✓ Test: {len(test_dataset)} samples")

# 现在可以训练模型了
# ... 你的训练代码 ...
```

然后运行：
```bash
python train.py
```

## 🔄 切换到 Harm-P 数据集

如果想使用 Politics 数据集：

1. 编辑 `extract_features_config.py`：
```python
DATASET_TO_USE = 'Harm-P'  # 改成 Harm-P
```

2. 重新提取特征：
```bash
python extract_and_save_features.py
```

3. 更新训练脚本路径：
```python
# 修改为 Harm-P 路径
demo.train_ROI = torch.load('./extracted_features/Harm-P/train_ROI.pt')
# ...
data_path='HarMeme_V1/Annotations/Harm-P/train_v1.jsonl'
# ...
```

## 📝 常见问题

### Q1: 没有 GPU 怎么办？
A: 可以使用 CPU，但会慢很多。修改代码：
```python
# 在所有 .cuda() 的地方改成 .to(device)
device = torch.device('cpu')
```

### Q2: 内存不足怎么办？
A: 减小 batch_size：
```python
DataLoader(dataset, batch_size=16)  # 从 64 改成 16
```

### Q3: 想跳过特征保存，直接训练？
A: 可以，但每次训练都要重新提取（慢）：
```python
dataset = demo.HarmemeMemesDatasetAug2(
    ...,
    use_preextracted=False  # 实时提取
)
```

### Q4: 特征文件太大，能删除吗？
A: 可以随时删除，需要时重新提取：
```bash
rm -rf ./extracted_features/
```

### Q5: 数据集没有 bbdict 和 ent 字段？
A: 没关系！代码会自动处理：
- 没有 bbdict → 使用整张图片中心裁剪
- 没有 ent → 使用整段文本编码

## 📚 文件说明

| 文件 | 用途 |
|------|------|
| `test_dataset_quick.py` | 快速测试配置 |
| `extract_features_config.py` | 数据集配置 |
| `extract_and_save_features.py` | 特征提取脚本 |
| `EMNLP_MOMENTA_All_DemoCode.py` | 主代码（已修改） |
| `example_usage.py` | 使用示例 |
| `USAGE.md` | 详细文档 |
| `QUICKSTART.md` | 本文件 |

## 🎯 建议工作流

1. **首次使用**：
   ```
   test_dataset_quick.py → extract_and_save_features.py → 训练
   ```

2. **后续实验**：
   ```
   直接训练（使用保存的特征）
   ```

3. **切换数据集**：
   ```
   修改 config → extract_and_save_features.py → 训练
   ```

---

**准备好了吗？开始第一步：**
```bash
python test_dataset_quick.py
```
