#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例：如何使用实时特征提取的 HarmemeMemesDatasetAug2

使用方法1：直接运行主脚本
    python EMNLP_MOMENTA_All_DemoCode.py

使用方法2：作为模块导入并使用
    见下方示例
"""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision import models
from PIL import Image

# 导入模块
import EMNLP_MOMENTA_All_DemoCode as demo

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 初始化必要的模型（这些会被数据集的处理函数使用）
print("Loading CLIP model...")
demo.clip_model = torch.jit.load("clip_model.pt").cuda().eval()
input_resolution = demo.clip_model.input_resolution.item()

print("Initializing preprocessor and tokenizer...")
demo.preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
])
demo.tokenizer = demo.SimpleTokenizer()

print("Loading VGG16 model...")
model_vgg_pretrained = models.vgg16(pretrained=True)
demo.model_vgg = demo.FeatureExtractor(model_vgg_pretrained)
demo.model_vgg = demo.model_vgg.to(device)

print("Loading SentenceTransformer...")
from sentence_transformers import SentenceTransformer
demo.model_sent_trans = SentenceTransformer('paraphrase-distilroberta-base-v1')

# 2. 创建数据集（使用实时特征提取）
print("\nCreating dataset with on-demand feature extraction...")
dataset = demo.HarmemeMemesDatasetAug2(
    data_path="path_to_jsonl/train.jsonl",
    img_dir="path_to_images/images",
    split_flag='train',
    use_preextracted=False  # 关键参数：使用实时提取
)

# 3. 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

# 4. 测试加载一个 batch
print("\nTesting dataloader...")
for batch in dataloader:
    print(f"Batch loaded successfully!")
    print(f"  - IDs: {batch['id']}")
    print(f"  - Image CLIP features shape: {batch['image_clip_input'].shape}")
    print(f"  - Image VGG features shape: {batch['image_vgg_feature'].shape}")
    print(f"  - Text CLIP features shape: {batch['text_clip_input'].shape}")
    print(f"  - Text embedding shape: {batch['text_drob_embedding'].shape}")
    if 'label' in batch:
        print(f"  - Labels: {batch['label']}")
    break  # 只测试第一个 batch

print("\nDone! Dataset is working with on-demand feature extraction.")
