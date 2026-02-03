# test.py
from EMNLP_MOMENTA_All_DemoCode import HarmemeMemesDatasetAug2
from torch.utils.data import DataLoader

dataset = HarmemeMemesDatasetAug2(
    data_path="./HarMeme_V1/Annotations/Harm-C/train.jsonl",
    img_dir="./HarMeme_V1/images_flat",
    split_flag="train"
)

print("Dataset size:", len(dataset))

sample = dataset[0]
print("Image clip shape:", sample["image_clip_input"].shape)
print("Label:", sample["label"])
