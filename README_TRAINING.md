# MOMENTA 训练指南

## 修改说明

已修改 `EMNLP_MOMENTA_All_DemoCode.py`，使其能够使用预提取的特征进行快速训练。

## 主要改动

### 1. 自动配置管理
- 从 `extract_features_config.py` 自动读取数据集配置
- 不再需要手动修改硬编码的路径

### 2. 预提取特征开关
```python
USE_PREEXTRACTED_FEATURES = True  # True: 使用预提取特征, False: 实时提取
```

### 3. 自动数据集适配
- 自动识别 Harm-C (3分类) 或 Harm-P (2分类)
- 自动设置正确的类别数和损失函数
- 自动生成实验名称和保存路径

## 使用方法

### 步骤 1: 配置数据集

编辑 `extract_features_config.py`:
```python
DATASET_TO_USE = 'Harm-C'  # 或 'Harm-P'
```

### 步骤 2: 提取特征（如果还没提取）

```bash
python extract_and_save_features.py
```

这会在 `./extracted_features/Harm-C/` 生成：
- `train_ROI.pt`, `train_ENT.pt`
- `val_ROI.pt`, `val_ENT.pt`
- `test_ROI.pt`, `test_ENT.pt`

### 步骤 3: 训练模型

```bash
python EMNLP_MOMENTA_All_DemoCode.py
```

## 训练配置

在 `EMNLP_MOMENTA_All_DemoCode.py` 的 `__main__` 部分：

```python
# 使用预提取特征（推荐，速度快）
USE_PREEXTRACTED_FEATURES = True

# 或者实时提取特征（慢，但不需要预先提取）
USE_PREEXTRACTED_FEATURES = False
```

## 模型输出

### 保存位置
```
./checkpoints/MOMENTA_Harm-C_MultiClass/
├── checkpoint_MOMENTA_Harm-C_MultiClass.pt  # 最佳模型
└── final.pt                                  # 最终模型
```

### 评估指标
训练完成后会显示：
- Accuracy (准确率)
- F1-Score (F1分数)
- Recall (召回率)
- Precision (精确率)
- MAE (平均绝对误差)
- MMAE (宏平均绝对误差)

## 训练参数

在代码中可以调整：
```python
n_epochs = 25      # 最大训练轮数
patience = 25      # Early stopping 耐心值
lr = 0.001        # 学习率
batch_size = 64   # 批次大小
```

## 切换数据集

### 方法 1: 修改配置文件（推荐）
编辑 `extract_features_config.py`:
```python
DATASET_TO_USE = 'Harm-P'  # 切换到 Politics 数据集
```

### 方法 2: 直接修改主文件
如果不使用配置文件，可以注释掉配置导入，使用硬编码路径（不推荐）。

## 性能对比

| 模式 | 特征提取时间 | 训练速度 | 总时间 |
|------|------------|---------|--------|
| 实时提取 | 每个epoch都要提取 | 慢 | 很长 |
| 预提取 | 一次性（约5-10分钟） | 快 | 短很多 |

**建议**: 使用预提取模式，特别是需要多次实验时。

## 常见问题

### Q1: FileNotFoundError: No such file or directory
确保：
1. `extract_features_config.py` 中的路径正确
2. 已经提取了特征文件
3. HarMeme_V1 数据集在正确位置

### Q2: ImportError: No module named 'pytorchtools'
项目已包含 `pytorchtools.py`，不需要额外下载。
如果还是报错，确保该文件在项目根目录。

### Q3: RuntimeError: Expected all tensors to be on the same device
这个问题已经修复。预提取的特征会自动加载到正确的设备（GPU/CPU）。
如果仍有问题，检查：
```python
print(train_ROI.device)  # 应该显示 cuda:0 或 cpu
```

### Q4: CUDA out of memory
降低 batch_size:
```python
batch_size = 32  # 或 16
```

或使用 CPU 训练（慢但不需要 GPU 内存）：
- 在代码开头会自动检测并使用 CPU 如果没有 GPU

## 加载已训练模型

```python
import torch
from EMNLP_MOMENTA_All_DemoCode import MM

# 加载模型
output_size = 3  # Harm-C
model = MM(output_size)
model.load_state_dict(torch.load('./checkpoints/MOMENTA_Harm-C_MultiClass/final.pt'))
model.to(device)
model.eval()

# 进行预测
# ...
```

## 更多信息

参考原始论文和代码说明了解 MOMENTA 模型的详细架构。
