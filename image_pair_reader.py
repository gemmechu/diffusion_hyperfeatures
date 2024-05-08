import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torchvision
import argparse
import pandas as pd

from sklearn.cluster import KMeans
import torch.nn.functional as F
from typing import Tuple
from PIL import Image

# from train_hyperfeatures import compute_clip_loss


def process_image(image_pil, res=None, range=(-1, 1)):
    if res:
        image_pil = image_pil.resize(res, Image.BILINEAR)
    image = torchvision.transforms.ToTensor()(image_pil) # range [0, 1]
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min # range [r_min, r_max]
    return image[None, ...], image_pil

def rescale_points(points, old_shape, new_shape):
    # Assumes old_shape and new_shape are in the format (w, h)
    # and points are in (y, x) order
    x_scale = new_shape[0] / old_shape[0]
    y_scale = new_shape[1] / old_shape[1]
    rescaled_points = np.multiply(points, np.array([y_scale, x_scale]))
    return rescaled_points

def load_image_pair(ann, load_size, device, image_path="", output_size=None):
    img1_pil = Image.open(f"{image_path}/{ann['source_path']}").convert("RGB")
    img2_pil = Image.open(f"{image_path}/{ann['target_path']}").convert("RGB")
    source_size = img1_pil.size
    target_size = img2_pil.size
    ann["source_size"] = source_size
    ann["target_size"] = target_size

    # swap from (x, y) to (y, x)
    if "source_points" in ann:
        source_points, target_points = ann["source_points"], ann["target_points"]
        source_points = np.flip(source_points, 1)
        target_points = np.flip(target_points, 1)
        if output_size is not None:
            source_points = rescale_points(source_points, source_size, output_size)
            target_points = rescale_points(target_points, target_size, output_size)
        else:
            source_points = rescale_points(source_points, source_size, load_size)
            target_points = rescale_points(target_points, target_size, load_size)
    else:
        source_points, target_points = None, None

    img1, img1_pil = process_image(img1_pil, res=load_size)
    img2, img2_pil = process_image(img2_pil, res=load_size)
    img1, img2 = img1.to(device), img2.to(device)
    imgs = torch.cat([img1, img2])

def load_image_pair_modified(img_pair_idx, pair_info, load_size, device, output_size=None):
    
    scene_name = pair_info.scene_name[img_pair_idx]
    img_id0 = pair_info.img_id0[img_pair_idx] #321
    img_id1 = pair_info.img_id1[img_pair_idx] #176
    
    img0_md = np.load("data/MegaDepth/00" + str(scene_name) +"/dense/frames/" + str(img_id0) + "_0.npy.npz")
    img1_md = np.load("data/MegaDepth/00" + str(scene_name) +"/dense/frames/" + str(img_id1) + "_0.npy.npz")

    img0_path = "data/MegaDepth_images/00" + str(scene_name) +"/dense/images/" + str(img0_md["image_fn"])
    img1_path = "data/MegaDepth_images/00" + str(scene_name) +"/dense/images/" + str(img1_md["image_fn"])

    img0_pil = Image.open(img0_path)
    img1_pil = Image.open(img1_path)

    source_size = img0_pil.size
    target_size = img1_pil.size
    
    total_pts = img0_md['xys'].shape[0]
    vis0, vis1 = np.zeros(total_pts, dtype=bool), np.zeros(total_pts,dtype=bool)

    vis0[img0_md["vis"]] = True 
    vis1[img1_md["vis"]] = True
    vis = np.multiply(vis0, vis1)

    corr0 = img0_md['xys'][vis]
    corr1 = img1_md['xys'][vis]

    # swap from (x, y) to (y, x)
    if corr0.shape[0] > 0:
        source_points, target_points = corr0, corr1
        source_points = np.flip(source_points, 1)
        target_points = np.flip(target_points, 1)
        if output_size is not None:
            source_points = rescale_points(source_points, source_size, output_size)
            target_points = rescale_points(target_points, target_size, output_size)
        else:
            source_points = rescale_points(source_points, source_size, load_size)
            target_points = rescale_points(target_points, target_size, load_size)
    else:
        source_points, target_points = None, None

    img0, img0_pil = process_image(img0_pil, res=load_size)
    img1, img1_pil = process_image(img1_pil, res=load_size)
    img0, img1 = img0.to(device), img1.to(device)
    imgs = torch.cat([img0, img1])

    return source_points, target_points, img0_pil, img1_pil, imgs

def image_pair_reader(args):
    load_size = (46, 46)
    output_size = (64, 64)

    pair_info = pd.read_csv("/data/diffusion_hyperfeatures/datasets/data/MegaDepth/0020/dense/pairs_metadata.csv")
    I = np.argmax(pair_info.overlap_score)


    source_points, target_points, _, _, imgs = load_image_pair_modified(I, pair_info, load_size, "cuda:0", output_size=output_size)
    img1_hyperfeats, img2_hyperfeats = torch.ones((1280,46,46)), torch.ones((1280,46,46))
    # compute_clip_loss()
    p = 1

    # Image.fromarray(np.hstack((np.array(img0),np.array(img1)))).show()

    p = 1


    print(pair_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/train.yaml")
    args = parser.parse_args()    
    image_pair_reader(args)
