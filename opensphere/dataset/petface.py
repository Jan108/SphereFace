import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PetFace(Dataset):
    def __init__(
            self, root_dir, img_list,
            test_mode=False,
            augment_params=None,
            name='barfoo',
        ):
        super().__init__()

        self.root_dir = root_dir
        self.img_list = img_list
        self.test_mode = test_mode
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ])
        self.name = name

        self.image_list = []
        self.label_list = []

        with open(img_list, 'r') as f:
            for line in f:
                line = line.strip()
                img_path, img_label = line.split(',')
                if img_path == 'label':
                    continue

                img_path = os.path.join(root_dir, img_path)
                if not os.path.exists(img_path):
                    continue

                self.image_list.append(img_path)
                self.label_list.append(int(img_label))
        self.num_classes = len(set(self.label_list))
        print(f'Loaded Dataset from {img_list}: found {len(self.image_list)} images with {self.num_classes} classes')

    def __getitem__(self, index):
        path_img = self.image_list[index]
        label = self.label_list[index]
        img = Image.open(path_img).convert('RGB').crop((28, 28, 196, 169)).resize((112, 112), Image.LANCZOS)

        sample = self.transform(img)
        label = torch.tensor(label, dtype=torch.int32)
        return sample, label

    def __len__(self):
        return len(self.image_list)
