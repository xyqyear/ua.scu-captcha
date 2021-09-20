import os
from typing import Iterable
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

labels = "0123456789abcdefghijklmnopqrstuvwxyz"
labels_reverse = {v: k for k, v in enumerate(labels)}

image_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        lambda x: x > 0.5,
        lambda x: x.float(),
    ]
)


def label2string(label: Iterable):
    return "".join(map(lambda x: labels[x], label))


def string2label(string: str):
    return torch.tensor(tuple(map(lambda x: labels_reverse[x], string)))


class CaptchaDataset(Dataset):
    def __init__(self, image_dir: str):
        self.imgs_path = [
            os.path.join(image_dir, filename) for filename in os.listdir(image_dir)
        ]

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label = string2label(os.path.split(img_path)[-1].split(".")[0])
        data = Image.open(img_path)
        data = image_transform(data)
        return data, label

    def __len__(self):
        return len(self.imgs_path)
