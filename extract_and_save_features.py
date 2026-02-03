#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取并保存特征文件

这个脚本会：
1. 加载数据集
2. 提取所有样本的 ROI 和 ENT 特征
3. 保存为 .pt 文件供后续使用

用法：
    python extract_and_save_features.py

配置：
    在 extract_features_config.py 中修改 DATASET_TO_USE 选择数据集
"""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision import models
from PIL import Image
from tqdm import tqdm
import os

import EMNLP_MOMENTA_All_DemoCode as demo

# ==================== 加载配置 ====================
try:
    from extract_features_config import DATA_CONFIG, SAVE_DIR, print_config, DATASET_TO_USE
    print_config()
except ImportError:
    print("Warning: extract_features_config.py not found, using default Harm-C configuration")
    DATASET_TO_USE = 'Harm-C'
    DATA_CONFIG = {
        'name': 'Harm-C (COVID-19)',
        'train': {
            'data_path': 'HarMeme_V1/Annotations/Harm-C/train.jsonl',
            'img_dir': 'HarMeme_V1/images_flat',
        },
        'val': {
            'data_path': 'HarMeme_V1/Annotations/Harm-C/val.jsonl',
            'img_dir': 'HarMeme_V1/images_flat',
        },
        'test': {
            'data_path': 'HarMeme_V1/Annotations/Harm-C/test.jsonl',
            'img_dir': 'HarMeme_V1/images_flat',
        }
    }
    SAVE_DIR = './extracted_features/Harm-C'

os.makedirs(SAVE_DIR, exist_ok=True)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 初始化模型 ====================
print("\n[1/4] Initializing models...")

print("  - Loading CLIP model...")
demo.clip_model = torch.jit.load("clip_model.pt").cuda().eval()
input_resolution = demo.clip_model.input_resolution.item()

print("  - Initializing preprocessor and tokenizer...")
demo.preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
])
demo.tokenizer = demo.SimpleTokenizer()

print("  - Loading VGG16 model...")
model_vgg_pretrained = models.vgg16(pretrained=True)
demo.model_vgg = demo.FeatureExtractor(model_vgg_pretrained)
demo.model_vgg = demo.model_vgg.to(device)

print("  - Loading SentenceTransformer...")
from sentence_transformers import SentenceTransformer
demo.model_sent_trans = SentenceTransformer('paraphrase-distilroberta-base-v1')

print("✓ Models initialized\n")

# ==================== 特征提取函数 ====================
def extract_features(split_name, data_path, img_dir):
    """
    提取指定数据集的特征

    Args:
        split_name: 'train', 'val', 或 'test'
        data_path: JSONL 文件路径
        img_dir: 图片文件夹路径

    Returns:
        roi_features: ROI 特征张量
        ent_features: Entity 特征张量
    """
    print(f"[2/4] Extracting features for {split_name} set...")

    # 创建数据集
    dataset = demo.HarmemeMemesDatasetAug2(
        data_path=data_path,
        img_dir=img_dir,
        split_flag=split_name,
        use_preextracted=False  # 实时提取
    )

    # 不使用 DataLoader，逐个样本提取（避免批处理复杂性）
    num_samples = len(dataset)
    print(f"  Total samples: {num_samples}")

    roi_features_list = []
    ent_features_list = []

    # 使用 tqdm 显示进度
    for idx in tqdm(range(num_samples), desc=f"  Extracting {split_name}"):
        try:
            sample = dataset[idx]
            roi_features_list.append(sample['image_vgg_feature'].cpu())
            ent_features_list.append(sample['text_drob_embedding'].cpu())
        except Exception as e:
            print(f"\n  Warning: Failed to process sample {idx}: {e}")
            # 使用零向量填充失败的样本
            roi_features_list.append(torch.zeros(4096))
            ent_features_list.append(torch.zeros(768))

    # 转换为张量
    roi_features = torch.stack(roi_features_list)
    ent_features = torch.stack(ent_features_list)

    print(f"  ✓ Extracted ROI features: {roi_features.shape}")
    print(f"  ✓ Extracted ENT features: {ent_features.shape}")

    return roi_features, ent_features

# ==================== 主程序 ====================
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Feature Extraction Script")
    print("=" * 60)

    total_start_time = time.time()

    # 对每个数据集分割提取特征
    for split_name in ['train', 'val', 'test']:
        if split_name not in DATA_CONFIG:
            continue

        config = DATA_CONFIG[split_name]

        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} set")
        print(f"{'='*60}")

        start_time = time.time()

        # 检查文件是否存在
        if not os.path.exists(config['data_path']):
            print(f"  ⚠ Skipping {split_name}: data file not found at {config['data_path']}")
            continue

        try:
            # 提取特征
            roi_features, ent_features = extract_features(
                split_name=split_name,
                data_path=config['data_path'],
                img_dir=config['img_dir']
            )

            # 保存特征
            print(f"\n[3/4] Saving features for {split_name} set...")
            roi_save_path = os.path.join(SAVE_DIR, f'{split_name}_ROI.pt')
            ent_save_path = os.path.join(SAVE_DIR, f'{split_name}_ENT.pt')

            torch.save(roi_features, roi_save_path)
            torch.save(ent_features, ent_save_path)

            print(f"  ✓ Saved ROI features to: {roi_save_path}")
            print(f"  ✓ Saved ENT features to: {ent_save_path}")

            # 统计
            roi_size_mb = os.path.getsize(roi_save_path) / (1024 * 1024)
            ent_size_mb = os.path.getsize(ent_save_path) / (1024 * 1024)
            elapsed_time = time.time() - start_time

            print(f"\n[4/4] Statistics for {split_name}:")
            print(f"  - ROI file size: {roi_size_mb:.2f} MB")
            print(f"  - ENT file size: {ent_size_mb:.2f} MB")
            print(f"  - Extraction time: {elapsed_time:.2f} seconds")
            print(f"  - Time per sample: {elapsed_time/len(roi_features):.3f} seconds")

        except Exception as e:
            print(f"  ✗ Error processing {split_name}: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"✓ All done! Total time: {total_elapsed/60:.2f} minutes")
    print(f"{'='*60}")
    print(f"\nExtracted features saved to: {SAVE_DIR}/")
    print("\nTo use these features in training:")
    print(f"""
import torch
import EMNLP_MOMENTA_All_DemoCode as demo

# Load pre-extracted features
demo.train_ROI = torch.load('{SAVE_DIR}/train_ROI.pt')
demo.train_ENT = torch.load('{SAVE_DIR}/train_ENT.pt')
demo.val_ROI = torch.load('{SAVE_DIR}/val_ROI.pt')
demo.val_ENT = torch.load('{SAVE_DIR}/val_ENT.pt')
demo.test_ROI = torch.load('{SAVE_DIR}/test_ROI.pt')
demo.test_ENT = torch.load('{SAVE_DIR}/test_ENT.pt')

# Create dataset with pre-extracted features
dataset = demo.HarmemeMemesDatasetAug2(
    data_path="your_data.jsonl",
    img_dir="your_images/",
    split_flag='train',
    use_preextracted=True  # Use pre-extracted features!
)
""")
