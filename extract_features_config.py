#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征提取配置文件

使用方法：
1. 修改下面的 DATASET_TO_USE 选择要提取的数据集
2. 运行: python extract_and_save_features.py
"""

# ==================== 选择数据集 ====================
# 可选值: 'Harm-C' (COVID-19) 或 'Harm-P' (Politics)
DATASET_TO_USE = 'Harm-C'  # 修改这里来切换数据集

# ==================== 数据集配置 ====================
DATASETS = {
    'Harm-C': {
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
    },
    'Harm-P': {
        'name': 'Harm-P (Politics)',
        'train': {
            'data_path': 'HarMeme_V1/Annotations/Harm-P/train_v1.jsonl',
            'img_dir': 'HarMeme_V1/images_flat',
        },
        'val': {
            'data_path': 'HarMeme_V1/Annotations/Harm-P/val_v1.jsonl',
            'img_dir': 'HarMeme_V1/images_flat',
        },
        'test': {
            'data_path': 'HarMeme_V1/Annotations/Harm-P/test_v1.jsonl',
            'img_dir': 'HarMeme_V1/images_flat',
        }
    }
}

# 获取当前选择的数据集配置
DATA_CONFIG = DATASETS[DATASET_TO_USE]
SAVE_DIR = f'./extracted_features/{DATASET_TO_USE}'

# 显示配置信息
def print_config():
    print(f"\n{'='*60}")
    print(f"Dataset Configuration: {DATA_CONFIG['name']}")
    print(f"{'='*60}")
    print(f"Train data: {DATA_CONFIG['train']['data_path']}")
    print(f"Val data:   {DATA_CONFIG['val']['data_path']}")
    print(f"Test data:  {DATA_CONFIG['test']['data_path']}")
    print(f"Images dir: {DATA_CONFIG['train']['img_dir']}")
    print(f"Save to:    {SAVE_DIR}")
    print(f"{'='*60}\n")
