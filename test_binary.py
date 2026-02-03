#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将三分类模型的结果转换为二分类进行评估

用法：
    python test_binary.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import EMNLP_MOMENTA_All_DemoCode as demo
from extract_features_config import DATA_CONFIG, DATASET_TO_USE

print("="*60)
print("Binary Classification Testing")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 加载模型 ====================
print("\n[1/4] Loading trained model...")
output_size = 3  # 原始模型是3分类
model = demo.MM(output_size)
model_path = f'./checkpoints/MOMENTA_{DATASET_TO_USE}_MultiClass/final.pt'
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
print(f"  ✓ Model loaded from: {model_path}")

# ==================== 加载预提取特征 ====================
print("\n[2/4] Loading pre-extracted features...")
FEATURE_DIR = f'./extracted_features/{DATASET_TO_USE}'
demo.test_ROI = torch.load(f'{FEATURE_DIR}/test_ROI.pt').to(device)
demo.test_ENT = torch.load(f'{FEATURE_DIR}/test_ENT.pt').to(device)
print(f"  ✓ Test features loaded")

# ==================== 初始化 CLIP ====================
print("\n[3/4] Initializing CLIP model...")
demo.clip_model = torch.jit.load("clip_model.pt")
if torch.cuda.is_available():
    demo.clip_model = demo.clip_model.cuda()
demo.clip_model.eval()

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from PIL import Image
input_resolution = demo.clip_model.input_resolution.item()
demo.preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
])
demo.tokenizer = demo.SimpleTokenizer()
print("  ✓ CLIP initialized")

# ==================== 创建测试数据集 ====================
print("\n[4/4] Creating test dataset...")
test_dataset = demo.HarmemeMemesDatasetAug2(
    data_path=DATA_CONFIG['test']['data_path'],
    img_dir=DATA_CONFIG['test']['img_dir'],
    split_flag='test',
    use_preextracted=True
)
dataloader_test = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
print(f"  ✓ Test dataset: {len(test_dataset)} samples")

# ==================== 测试并收集预测结果 ====================
print("\n" + "="*60)
print("Running inference on test set...")
print("="*60)

outputs = []
with torch.no_grad():
    for data in dataloader_test:
        img_inp_clip = data['image_clip_input']
        txt_inp_clip = data['text_clip_input']

        img_feat_clip = demo.clip_model.encode_image(img_inp_clip).float().to(device)
        txt_feat_clip = demo.clip_model.encode_text(txt_inp_clip).float().to(device)

        img_feat_vgg = data['image_vgg_feature']
        txt_feat_trans = data['text_drob_embedding']

        out = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
        outputs += list(out.cpu().data.numpy())

# ==================== 三分类预测 ====================
y_pred_3class = []
for i in outputs:
    y_pred_3class.append(np.argmax(i))

# ==================== 转换为二分类 ====================
print("\nConverting to binary classification...")
print("  Mapping: 0 (not harmful) -> 0")
print("           1 (somewhat harmful) -> 1 (harmful)")
print("           2 (very harmful) -> 1 (harmful)")

# 将三分类预测转换为二分类
y_pred_binary = []
for pred in y_pred_3class:
    if pred == 0:
        y_pred_binary.append(0)  # not harmful
    else:
        y_pred_binary.append(1)  # harmful (包括 somewhat 和 very)

# ==================== 获取二分类真实标签 ====================
test_samples_frame = pd.read_json(DATA_CONFIG['test']['data_path'], lines=True)
test_labels_binary = []
for index, row in test_samples_frame.iterrows():
    lab = row['labels'][0]
    if lab == "not harmful":
        test_labels_binary.append(0)
    else:
        test_labels_binary.append(1)  # 所有harmful类别

# ==================== 计算二分类指标 ====================
print("\n" + "="*60)
print("Binary Classification Results:")
print("="*60)

acc = np.round(accuracy_score(test_labels_binary, y_pred_binary), 4)
prec = np.round(precision_score(test_labels_binary, y_pred_binary, average="binary"), 4)
rec = np.round(recall_score(test_labels_binary, y_pred_binary, average="binary"), 4)
f1 = np.round(f1_score(test_labels_binary, y_pred_binary, average="binary"), 4)

print(classification_report(test_labels_binary, y_pred_binary,
                          target_names=['Not Harmful', 'Harmful']))

print("\n" + "="*60)
print("Summary Metrics (Binary):")
print("="*60)
print(f"Accuracy:  {acc}")
print(f"Precision: {prec}")
print(f"Recall:    {rec}")
print(f"F1-Score:  {f1}")
print("="*60)

# ==================== 对比三分类结果 ====================
print("\n" + "="*60)
print("Comparison: 3-Class vs Binary")
print("="*60)

# 三分类真实标签
test_labels_3class = []
for index, row in test_samples_frame.iterrows():
    lab = row['labels'][0]
    if lab == "not harmful":
        test_labels_3class.append(0)
    elif lab == "somewhat harmful":
        test_labels_3class.append(1)
    else:
        test_labels_3class.append(2)

acc_3class = np.round(accuracy_score(test_labels_3class, y_pred_3class), 4)
f1_3class = np.round(f1_score(test_labels_3class, y_pred_3class, average="macro"), 4)

print(f"3-Class Accuracy: {acc_3class}")
print(f"3-Class F1-Score: {f1_3class}")
print(f"\nBinary Accuracy:  {acc}")
print(f"Binary F1-Score:  {f1}")
print("="*60)

print("\n✓ Binary classification testing completed!")
