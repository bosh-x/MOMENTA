#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试数据集加载和特征提取

这个脚本会：
1. 测试加载一个样本
2. 验证特征提取是否正常工作
3. 不保存任何东西

用法：
    python test_dataset_quick.py
"""

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision import models
from PIL import Image
import os

import EMNLP_MOMENTA_All_DemoCode as demo

print("="*60)
print("Quick Dataset Test")
print("="*60)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {device}")

# 检查必要文件
print("\n[1/5] Checking required files...")
required_files = [
    'HarMeme_V1/Annotations/Harm-C/train.jsonl',
    'HarMeme_V1/images/HarMeme_Images/harmeme_images_covid_19',
    'clip_model.pt',
    'bpe_simple_vocab_16e6.txt.gz'
]

all_exist = True
for f in required_files:
    exists = os.path.exists(f)
    status = "✓" if exists else "✗"
    print(f"  {status} {f}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n✗ Some required files are missing!")
    print("\nTo download missing files:")
    print("  - CLIP model: wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -O clip_model.pt")
    print("  - BPE vocab: wget https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz")
    exit(1)

# 初始化模型
print("\n[2/5] Initializing models...")
try:
    print("  - Loading CLIP model...")
    demo.clip_model = torch.jit.load("clip_model.pt")
    if torch.cuda.is_available():
        demo.clip_model = demo.clip_model.cuda()
    demo.clip_model.eval()
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
    demo.model_vgg.eval()

    print("  - Loading SentenceTransformer...")
    from sentence_transformers import SentenceTransformer
    demo.model_sent_trans = SentenceTransformer('paraphrase-distilroberta-base-v1')

    print("  ✓ All models loaded successfully")
except Exception as e:
    print(f"\n✗ Error loading models: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 创建数据集
print("\n[3/5] Creating dataset...")
try:
    dataset = demo.HarmemeMemesDatasetAug2(
        data_path='HarMeme_V1/Annotations/Harm-C/train.jsonl',
        img_dir='HarMeme_V1/images/HarMeme_Images/harmeme_images_covid_19',
        split_flag='train',
        use_preextracted=False  # 实时提取
    )
    print(f"  ✓ Dataset created: {len(dataset)} samples")
except Exception as e:
    print(f"\n✗ Error creating dataset: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试加载一个样本
print("\n[4/5] Testing sample loading...")
try:
    print("  Loading sample 0...")
    sample = dataset[0]

    print(f"  ✓ Sample loaded successfully!")
    print(f"    - ID: {sample['id']}")
    print(f"    - Image CLIP shape: {sample['image_clip_input'].shape}")
    print(f"    - Image VGG shape: {sample['image_vgg_feature'].shape}")
    print(f"    - Text CLIP shape: {sample['text_clip_input'].shape}")
    print(f"    - Text embedding shape: {sample['text_drob_embedding'].shape}")
    if 'label' in sample:
        print(f"    - Label: {sample['label']}")
except Exception as e:
    print(f"\n✗ Error loading sample: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试特征形状
print("\n[5/5] Validating feature shapes...")
expected_shapes = {
    'image_clip_input': (3, 224, 224),  # CLIP image input
    'image_vgg_feature': (4096,),        # VGG features
    'text_clip_input': (77,),            # CLIP text tokens
    'text_drob_embedding': (768,)        # Sentence-BERT embeddings
}

all_correct = True
for key, expected in expected_shapes.items():
    actual = tuple(sample[key].shape)
    match = actual == expected
    status = "✓" if match else "✗"
    print(f"  {status} {key}: {actual} {'(expected ' + str(expected) + ')' if not match else ''}")
    if not match:
        all_correct = False

# 总结
print("\n" + "="*60)
if all_correct:
    print("✓ All tests passed! Ready to extract features.")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: python extract_and_save_features.py")
    print("  2. Wait for feature extraction to complete (~6-12 minutes)")
    print("  3. Use extracted features for fast training!")
else:
    print("✗ Some tests failed. Please check the errors above.")
    print("="*60)
