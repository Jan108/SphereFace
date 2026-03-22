import argparse
import glob
import os
import yaml
import os.path as osp
from datetime import datetime, timedelta
from itertools import islice

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from opensphere.dataset import PetFaceVerification, PetFaceIdentification
from opensphere.model import Model


@torch.no_grad()
def verification(params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get config of the trained model
    cfg_path = osp.join(params.proj_dir, 'config.yml')
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    # get model
    model = Model(config['model'])
    model_path = glob.glob(f'{params.proj_dir}/checkpoint/model_*.pth')[0]
    print(f'Loading model from {model_path}')
    model.load_state_dict(torch.load(model_path))
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    model.eval_mode()

    def net(img):
        feat = model.get_feature(img)
        feat = model.head.f_wrapping(feat)
        return model.head.f_fusing(feat)

    # Load data
    dataset = PetFaceVerification(params.img_path, params.img_verification)
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    # Predict similarity
    sim_list = []
    label_list = []
    for img1, img2, label in tqdm(test_loader, desc='Test image pairs'):
        vec1 = F.normalize(net(img1.to(device)))
        vec2 = F.normalize(net(img2.to(device)))

        sim = nn.CosineSimilarity()(vec1, vec2).cpu().data.numpy().tolist()
        sim_list += sim
        label_list += label.cpu().data.numpy().tolist()

    pd.DataFrame(
        {'file1': dataset.image1_list, 'file2': dataset.image2_list,
         'sim': sim_list, 'label': label_list}).to_csv(
        os.path.join(params.proj_dir, 'verification.csv'), index=False)

    # Test latency
    test_loader_latency = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    inf_times = []
    for img1, img2, label in tqdm(islice(test_loader_latency, params.latency_test),
                                  desc='Test image pairs latency', total=params.latency_test):
        start_time = datetime.now()
        F.normalize(net(img1.to(device)))
        F.normalize(net(img2.to(device)))
        end_time = datetime.now()
        inf_times.append(end_time - start_time)
    avg_time = sum(inf_times, timedelta()) / len(inf_times)
    print(f'Inference for {params.proj_dir} took {avg_time}')
    with open(os.path.join(params.proj_dir, 'timing.txt'), 'w') as file:
        file.write(str(avg_time))


def identification(params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pool_loader = DataLoader(PetFaceIdentification(params.img_path, params.img_identification, pool_only=True), batch_size=64, shuffle=False)
    test_loader = DataLoader(PetFaceIdentification(params.img_path, params.img_identification, pool_only=False), batch_size=16, shuffle=False)

    # get config of the trained model
    cfg_path = osp.join(params.proj_dir, 'config.yml')
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    # get model
    model = Model(config['model'])
    model_path = glob.glob(f'{params.proj_dir}/checkpoint/model_*.pth')[0]
    print(f'Loading model from {model_path}')
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    model.eval_mode()

    def net(img):
        feat = model.get_feature(img)
        feat = model.head.f_wrapping(feat)
        return model.head.f_fusing(feat)

    features, labels = [], []
    with torch.no_grad():
        for images, batch_labels in tqdm(pool_loader, desc="Extracting features"):
            images = images.to(device)
            batch_features = net(images)
            batch_features = F.normalize(batch_features, p=2, dim=1)
            features.append(batch_features)
            labels.extend(batch_labels.tolist())

    pool_features, pool_labels = torch.cat(features, dim=0), labels
    pool_features = pool_features.to(device)

    # Open output file
    with open(os.path.join(params.proj_dir, 'identification.csv'), 'w') as f:
        f.write("test_label,predicted_label\n")

        # Process test images in batches
        for test_images, test_labels in tqdm(test_loader, desc="Processing test images"):
            test_images = test_images.to(device)
            test_features = net(test_images)
            test_features = F.normalize(test_features, p=2, dim=1)

            # Compute cosine similarity (dot product of normalized vectors)
            sim = torch.matmul(test_features, pool_features.T)  # (batch_size, num_pool)
            max_sim, indices = torch.max(sim, dim=1)

            # Determine predicted labels
            predicted_labels = [
                pool_labels[i] if sim > 0 else -1
                for sim, i in zip(max_sim.tolist(), indices.tolist())
            ]

            # Write results
            for test_label, pred_label in zip(test_labels.tolist(), predicted_labels):
                f.write(f"{test_label},{pred_label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SphereFace Evaluation')
    parser.add_argument("--proj_dir", type=str, required=True, help="Project directory")
    parser.add_argument("--img_path", type=str, default='./data/images', help="Image file rott directory")
    parser.add_argument("--img_verification", type=str, default='./data/split/verification.csv',
                        help="File containing a list of images files with corresponding label for verification")
    parser.add_argument("--img_identification", type=str, default='./data/split/identification.csv',
                        help="File containing a list of images files with corresponding label for identification")
    parser.add_argument("--latency_test", type=int, default=1000, help="Amount of images for the latency test")
    args = parser.parse_args()
    verification(args)
    identification(args)
