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
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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

        sample = self.transform(img).to(torch.float32)
        label = torch.tensor(label, dtype=torch.int64)
        return sample, label

    def __len__(self):
        return len(self.image_list)


class PetFaceIdentification(Dataset):
    def __init__(self, root_dir, img_list, pool_only=False):
        super(PetFaceIdentification, self).__init__()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.image_list = []
        self.label_list = []

        with open(img_list, 'r') as f:
            for line in f:
                line = line.strip()
                img_label, img_path, pool = line.split(',')
                if img_label == 'individual':
                    continue

                if (pool_only and pool == '0') or (not pool_only and pool == '1'):
                    continue

                img_path = os.path.join(root_dir, img_path)
                if not os.path.exists(img_path):
                    continue

                self.image_list.append(img_path)
                self.label_list.append(int(img_label))
        self.num_classes = len(set(self.label_list))
        print(f'Loaded identification Dataset from {img_list}: found {len(self.image_list)} images')


    def __getitem__(self, index):
        path_img = self.image_list[index]
        label = self.label_list[index]
        img = Image.open(path_img).convert('RGB').crop((28, 28, 196, 169)).resize((112, 112), Image.LANCZOS)

        sample = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
        return sample, label

    def __len__(self):
        return len(self.image_list)


class PetFaceVerification(Dataset):
    def __init__(self, root_dir, img_list):
        super(PetFaceVerification, self).__init__()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.image1_list = []
        self.image2_list = []
        self.label_list = []

        with open(img_list, 'r') as f:
            for line in f:
                line = line.strip()
                img1_path, img2_path, img_label = line.split(',')
                if img_label == 'label':
                    continue

                img1_path = os.path.join(root_dir, img1_path)
                img2_path = os.path.join(root_dir, img2_path)
                if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                    continue

                self.image1_list.append(img1_path)
                self.image2_list.append(img2_path)
                self.label_list.append(int(img_label))
        self.num_classes = len(set(self.label_list))
        print(f'Loaded Dataset from {img_list}: found {len(self.image1_list)} images pairs')


    def __getitem__(self, index):
        path_img1 = self.image1_list[index]
        path_img2 = self.image2_list[index]
        label = self.label_list[index]
        img1 = Image.open(path_img1).convert('RGB').crop((28, 28, 196, 169)).resize((112, 112), Image.LANCZOS)
        img2 = Image.open(path_img2).convert('RGB').crop((28, 28, 196, 169)).resize((112, 112), Image.LANCZOS)

        sample1 = self.transform(img1)
        sample2 = self.transform(img2)
        label = torch.tensor(label, dtype=torch.long)
        return sample1, sample2, label

    def __len__(self):
        return len(self.image1_list)