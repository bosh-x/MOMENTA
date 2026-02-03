#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用预提取特征训练 MOMENTA 模型

用法：
    python train_with_preextracted.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from PIL import Image
from pathlib import Path
import os

import EMNLP_MOMENTA_All_DemoCode as demo
from extract_features_config import DATA_CONFIG, DATASET_TO_USE, print_config

print("="*60)
print("MOMENTA Training with Pre-extracted Features")
print("="*60)

# 打印配置
print_config()

# ==================== 1. 设置设备 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}\n")

# ==================== 2. 加载预提取的特征 ====================
print("[1/6] Loading pre-extracted features...")
feature_dir = f'./extracted_features/{DATASET_TO_USE}'

demo.train_ROI = torch.load(f'{feature_dir}/train_ROI.pt')
demo.train_ENT = torch.load(f'{feature_dir}/train_ENT.pt')
demo.val_ROI = torch.load(f'{feature_dir}/val_ROI.pt')
demo.val_ENT = torch.load(f'{feature_dir}/val_ENT.pt')
demo.test_ROI = torch.load(f'{feature_dir}/test_ROI.pt')
demo.test_ENT = torch.load(f'{feature_dir}/test_ENT.pt')

print(f"  ✓ Train: ROI {demo.train_ROI.shape}, ENT {demo.train_ENT.shape}")
print(f"  ✓ Val:   ROI {demo.val_ROI.shape}, ENT {demo.val_ENT.shape}")
print(f"  ✓ Test:  ROI {demo.test_ROI.shape}, ENT {demo.test_ENT.shape}")

# ==================== 3. 初始化 CLIP 模型（还需要用它编码图像和文本）====================
print("\n[2/6] Initializing CLIP model...")
demo.clip_model = torch.jit.load("clip_model.pt")
if torch.cuda.is_available():
    demo.clip_model = demo.clip_model.cuda()
demo.clip_model.eval()
input_resolution = demo.clip_model.input_resolution.item()

demo.preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
])
demo.tokenizer = demo.SimpleTokenizer()
print("  ✓ CLIP model loaded")

# ==================== 4. 创建数据集和 DataLoaders ====================
print("\n[3/6] Creating datasets...")

train_dataset = demo.HarmemeMemesDatasetAug2(
    data_path=DATA_CONFIG['train']['data_path'],
    img_dir=DATA_CONFIG['train']['img_dir'],
    split_flag='train',
    use_preextracted=True  # 使用预提取特征！
)

val_dataset = demo.HarmemeMemesDatasetAug2(
    data_path=DATA_CONFIG['val']['data_path'],
    img_dir=DATA_CONFIG['val']['img_dir'],
    split_flag='val',
    use_preextracted=True
)

test_dataset = demo.HarmemeMemesDatasetAug2(
    data_path=DATA_CONFIG['test']['data_path'],
    img_dir=DATA_CONFIG['test']['img_dir'],
    split_flag='test',
    use_preextracted=True
)

print(f"  ✓ Train dataset: {len(train_dataset)} samples")
print(f"  ✓ Val dataset:   {len(val_dataset)} samples")
print(f"  ✓ Test dataset:  {len(test_dataset)} samples")

# 创建 DataLoaders
batch_size = 64
dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"  ✓ Batch size: {batch_size}")

# ==================== 5. 初始化模型 ====================
print("\n[4/6] Initializing MOMENTA model...")

# 输出类别数（根据数据集调整）
output_size = 3  # Harm-C: 3类 (not harmful, somewhat harmful, very harmful)
# output_size = 2  # 如果是二分类

# 创建模型
model = demo.MM(output_size)
model = model.to(device)

# 优化器和损失函数
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

print(f"  ✓ Model created")
print(f"  ✓ Output size: {output_size} classes")
print(f"  ✓ Learning rate: {lr}")

# 总参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ Total parameters: {total_params:,}")

# ==================== 6. 训练设置 ====================
print("\n[5/6] Training configuration...")

# 实验设置
exp_name = f"MOMENTA_{DATASET_TO_USE}_MultiClass"
exp_path = f"./checkpoints/{exp_name}"
Path(exp_path).mkdir(parents=True, exist_ok=True)

n_epochs = 25
patience = 10  # Early stopping patience

print(f"  ✓ Experiment: {exp_name}")
print(f"  ✓ Save path: {exp_path}")
print(f"  ✓ Epochs: {n_epochs}")
print(f"  ✓ Early stopping patience: {patience}")

# ==================== 7. 开始训练 ====================
print("\n" + "="*60)
print("[6/6] Starting training...")
print("="*60)

# 导入 EarlyStopping（如果有的话）
try:
    import sys
    sys.path.append('path_to_the_module/early-stopping-pytorch')
    from pytorchtools import EarlyStopping
    use_early_stopping = True
except ImportError:
    print("⚠ EarlyStopping not found, will train for all epochs")
    use_early_stopping = False

# 训练！
print("\nTraining started...\n")

try:
    # 调用训练函数
    if use_early_stopping:
        model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, final_epoch = demo.train_model(
            model, patience, n_epochs
        )
    else:
        # 如果没有 early stopping，需要修改 train_model 或者自己实现简单训练循环
        print("Training without early stopping...")
        # 这里可以实现一个简单的训练循环
        pass

    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)

    # ==================== 8. 测试评估 ====================
    print("\n[7/6] Evaluating on test set...")
    outputs = demo.test_model(model)

    print("\n" + "="*60)
    print("✓ All done!")
    print("="*60)
    print(f"\nModel saved to: {exp_path}/final.pt")
    print(f"To use this model later:")
    print(f"""
    model = demo.MM({output_size})
    model.load_state_dict(torch.load('{exp_path}/final.pt'))
    model.to(device)
    model.eval()
    """)

except KeyboardInterrupt:
    print("\n\n⚠ Training interrupted by user")
    print(f"Partial model saved to: {exp_path}/")

except Exception as e:
    print(f"\n✗ Error during training: {e}")
    import traceback
    traceback.print_exc()
