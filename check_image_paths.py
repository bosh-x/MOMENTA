#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查图片路径配置

这个脚本会：
1. 读取 JSONL 文件
2. 检查实际的图片路径
3. 建议正确的 img_dir 配置

用法：
    python check_image_paths.py
"""

import json
import os
from pathlib import Path

print("="*60)
print("Image Path Configuration Checker")
print("="*60)

# 读取第一个样本
jsonl_path = 'HarMeme_V1/Annotations/Harm-C/train.jsonl'
print(f"\nReading: {jsonl_path}")

with open(jsonl_path, 'r') as f:
    first_line = f.readline()
    sample = json.loads(first_line)

print(f"\nFirst sample:")
print(f"  ID: {sample['id']}")
print(f"  Image field: '{sample['image']}'")

# 可能的图片目录
possible_dirs = [
    'HarMeme_V1/images_flat',
    'HarMeme_V1/images/HarMeme_Images/harmeme_images_covid_19',
    'HarMeme_V1/images',
]

print(f"\nChecking possible image directories...")
found = False

for img_dir in possible_dirs:
    # 方式1: 直接拼接
    path1 = os.path.join(img_dir, sample['image'])
    # 方式2: 只用文件名（去掉子路径）
    filename = os.path.basename(sample['image'])
    path2 = os.path.join(img_dir, filename)

    print(f"\n  Trying: {img_dir}")

    if os.path.exists(path1):
        print(f"    ✓ FOUND (full path): {path1}")
        print(f"\n{'='*60}")
        print(f"SUCCESS! Use this configuration:")
        print(f"{'='*60}")
        print(f"img_dir = '{img_dir}'")
        print(f"\nThe image field contains: '{sample['image']}'")
        print(f"Full path will be: {path1}")
        found = True
        break
    elif os.path.exists(path2):
        print(f"    ✓ FOUND (basename only): {path2}")
        print(f"\n{'='*60}")
        print(f"SUCCESS! Use this configuration:")
        print(f"{'='*60}")
        print(f"img_dir = '{img_dir}'")
        print(f"\nNote: Need to use basename of image field!")
        print(f"The code needs to be modified to use: os.path.basename(row.image)")
        found = True
        break
    else:
        print(f"    ✗ Not found at: {path1}")
        print(f"    ✗ Not found at: {path2}")

if not found:
    print(f"\n{'='*60}")
    print("Image not found in common locations!")
    print(f"{'='*60}")
    print("\nPlease check manually:")
    print("1. Run: find HarMeme_V1 -name 'covid_memes_18.png'")
    print("2. Check where the images are actually located")
    print("\nOr list the directory structure:")
    print("   ls -R HarMeme_V1/images*/")

print("\n" + "="*60)
print("Listing available image directories:")
print("="*60)

for root, dirs, files in os.walk('HarMeme_V1'):
    # 只显示包含图片文件的目录
    if any(f.endswith(('.png', '.jpg', '.jpeg')) for f in files):
        num_images = sum(1 for f in files if f.endswith(('.png', '.jpg', '.jpeg')))
        print(f"  {root}: {num_images} images")
