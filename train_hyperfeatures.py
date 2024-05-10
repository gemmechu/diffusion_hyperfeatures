import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from omegaconf import OmegaConf
import pandas as pd
import torch
from tqdm import tqdm
import wandb
import pandas as pd
import glob 

from archs.correspondence_utils import (
    load_image_pair,
    batch_cosine_sim,
    points_to_idxs,
    find_nn_source_correspondences,
    draw_correspondences,
    compute_pck,
    rescale_points
)
from archs.stable_diffusion.resnet import collect_dims
from archs.diffusion_extractor import DiffusionExtractor
from archs.aggregation_network import AggregationNetwork
from image_pair_reader import load_image_pair_modified
from metrics import px_th_noquery

def get_rescale_size(config):
    output_size = (config["output_resolution"], config["output_resolution"])
    if "load_resolution" in config:
        load_size = (config["load_resolution"], config["load_resolution"])
    else:
        load_size = output_size
    return output_size, load_size

def log_aggregation_network(aggregation_network, config):
    mixing_weights = torch.nn.functional.softmax(aggregation_network.mixing_weights)
    num_layers = len(aggregation_network.feature_dims)
    num_timesteps = len(aggregation_network.save_timestep)
    save_timestep = aggregation_network.save_timestep
    if config["diffusion_mode"] == "inversion":
        save_timestep = save_timestep[::-1]
    fig, ax = plt.subplots()
    ax.imshow(mixing_weights.view((num_timesteps, num_layers)).T.detach().cpu().numpy())
    ax.set_ylabel("Layer")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(1, num_layers+1))
    ax.set_xlabel("Timestep")
    ax.set_xticklabels(save_timestep)
    ax.set_xticks(range(num_timesteps))
    wandb.log({f"mixing_weights": plt})

def get_hyperfeats(diffusion_extractor, aggregation_network, imgs):
    with torch.inference_mode():
        with torch.autocast("cuda"):
            feats, _ = diffusion_extractor.forward(imgs)
            b, s, l, w, h = feats.shape
    diffusion_hyperfeats = aggregation_network(feats.float().view((b, -1, w, h)))
    img1_hyperfeats = diffusion_hyperfeats[0][None, ...]
    img2_hyperfeats = diffusion_hyperfeats[1][None, ...]
    return img1_hyperfeats, img2_hyperfeats

def get_hyperfeats_modified(img_pair_idx, pair_info, aggregation_network,device, split="train"):

    scene_name = pair_info.scene_name[img_pair_idx]

    img_id0 = pair_info.img_id0[img_pair_idx] 
    img_id1 = pair_info.img_id1[img_pair_idx] 

    tk_id = [0, 128, 261]
    img0_feat = torch.empty([len(tk_id),1280,16,16])
    img1_feat = torch.empty([len(tk_id),1280,16,16])
    for i in range(len(tk_id)):
        img0_feat[i] = (torch.load(f"/share/hariharan/ac2538/proj_md_traindata2{'_val' if split == 'val' else ''}/scene_" + str(scene_name.replace("/", "-"))+"_imgid_" + str(img_id0) +".pt")['dift'][tk_id[i]]).squeeze()
        img1_feat[i] = (torch.load(f"/share/hariharan/ac2538/proj_md_traindata2{'_val' if split == 'val' else ''}/scene_" + str(scene_name.replace("/", "-"))+"_imgid_" + str(img_id1) +".pt")['dift'][tk_id[i]]).squeeze()

    input = torch.stack((img0_feat.view((-1, 16, 16)), img1_feat.view((-1, 16, 16))), dim = 0)
    input = input.to(device)

    diffusion_hyperfeats = aggregation_network(input)
    img1_hyperfeats = diffusion_hyperfeats[0][None, ...]
    img2_hyperfeats = diffusion_hyperfeats[1][None, ...]
    return img1_hyperfeats, img2_hyperfeats

def compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size):
    # Assumes hyperfeats are batch_size=1 to avoid complex indexing
    # Compute in both directions for cycle consistency
    source_logits = aggregation_network.logit_scale.exp() * batch_cosine_sim(img1_hyperfeats, img2_hyperfeats)
    target_logits = aggregation_network.logit_scale.exp() * batch_cosine_sim(img2_hyperfeats, img1_hyperfeats)
    source_idx = torch.from_numpy(points_to_idxs(source_points, output_size)).long().to(source_logits.device)
    target_idx = torch.from_numpy(points_to_idxs(target_points, output_size)).long().to(target_logits.device)
    loss_source = torch.nn.functional.cross_entropy(source_logits[0, source_idx], target_idx)
    loss_target = torch.nn.functional.cross_entropy(target_logits[0, target_idx], source_idx)
    loss = (loss_source + loss_target) / 2
    return loss

def save_model(config, aggregation_network, optimizer, step):
    dict_to_save = {
        "step": step,
        "config": config,
        "aggregation_network": aggregation_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    results_folder = f"{config['results_folder']}/{wandb.run.name}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    torch.save(dict_to_save, f"{results_folder}/checkpoint_step_{step}.pt")

# def validate(config, diffusion_extractor, aggregation_network, val_anns):
#     device = config.get("device", "cuda")
#     output_size, load_size = get_rescale_size(config)
#     plot_every_n_steps = config.get("plot_every_n_steps", -1)
#     pck_threshold = config["pck_threshold"]
#     ids, val_dist, val_pck_img, val_pck_bbox = [], [], [], []
#     for j, ann in tqdm(enumerate(val_anns)):
#         with torch.no_grad():
#             source_points, target_points, img1_pil, img2_pil, imgs = load_image_pair(ann, load_size, device, image_path=config["image_path"])
#             img1_hyperfeats, img2_hyperfeats = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
#             loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
#             wandb.log({"val/loss": loss.item()}, step=j)
#             # Log NN correspondences
#             _, predicted_points = find_nn_source_correspondences(img1_hyperfeats, img2_hyperfeats, source_points, output_size, load_size)
#             predicted_points = predicted_points.detach().cpu().numpy()
#             # Rescale to the original image dimensions
#             target_size = ann["target_size"]
#             predicted_points = rescale_points(predicted_points, load_size, target_size)
#             target_points = rescale_points(target_points, load_size, target_size)
#             dist, pck_img, sample_pck_img = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold)
#             _, pck_bbox, sample_pck_bbox = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold, target_bounding_box=ann["target_bounding_box"])
#             wandb.log({"val/sample_pck_img": sample_pck_img}, step=j)
#             wandb.log({"val/sample_pck_bbox": sample_pck_bbox}, step=j)
#             val_dist.append(dist)
#             val_pck_img.append(pck_img)
#             val_pck_bbox.append(pck_bbox)
#             ids.append([j] * len(dist))
#             if plot_every_n_steps > 0 and j % plot_every_n_steps == 0:
#                 title = f"pck@{pck_threshold}_img: {sample_pck_img.round(decimals=2)}"
#                 title += f"\npck@{pck_threshold}_bbox: {sample_pck_bbox.round(decimals=2)}"
#                 draw_correspondences(source_points, predicted_points, img1_pil, img2_pil, title=title, radius1=1)
#                 wandb.log({"val/correspondences": plt}, step=j)
#     ids = np.concatenate(ids)
#     val_dist = np.concatenate(val_dist)
#     val_pck_img = np.concatenate(val_pck_img)
#     val_pck_bbox = np.concatenate(val_pck_bbox)
#     df = pd.DataFrame({
#         "id": ids,
#         "distances": val_dist,
#         "pck_img": val_pck_img,
#         "pck_bbox": val_pck_bbox,
#     })
#     wandb.log({"val/pck_img": val_pck_img.sum() / len(val_pck_img)})
#     wandb.log({"val/pck_bbox": val_pck_bbox.sum() / len(val_pck_bbox)})
#     wandb.log({f"val/distances_csv": wandb.Table(dataframe=df)})

def validate_modified(config, aggregation_network, val_anns, train_step, dump_pairs=False):
    
    pt_files = glob.glob(os.path.join(config["proj_md_val_path"], '**', '*.pt'), recursive=True)
    existing_data = {int(os.path.basename(file).split('_')[-1].replace('.pt', '')) for file in pt_files}
    condition = val_anns['img_id0'].isin(existing_data) & val_anns['img_id1'].isin(existing_data)
    val_anns = val_anns[condition]
    
    device = config.get("device", "cuda")
    output_size, load_size = get_rescale_size(config)
    plot_every_n_steps = config.get("plot_every_n_steps", -1)
    pck_threshold = config["pck_threshold"]
    ids, val_dist, val_pck_img, val_pck_bbox = [], [], [], []
    # import ipdb; ipdb.set_trace()
    val_loss = 0
    j = 0
    
    all_px_th_dict = None
    for j, ann in tqdm(enumerate(list(val_anns.index))):
        with torch.no_grad():
            # load_image_pair_modified(i, train_anns, load_size, device, output_size=output_size)
            source_points, target_points, img1_pil, img2_pil, imgs = load_image_pair_modified(ann, val_anns, load_size, device, output_size=output_size)
            img1_hyperfeats, img2_hyperfeats = get_hyperfeats_modified(ann, val_anns, aggregation_network, device, split="val")
            loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            # wandb.log({"val/loss": loss.item()}, step=j)
            val_loss += loss.detach().cpu().item()
            # Log NN correspondences
            _, predicted_points = find_nn_source_correspondences(img1_hyperfeats, img2_hyperfeats, source_points, output_size, load_size)
            predicted_points = predicted_points.detach().cpu().numpy()
            # Rescale to the original image dimensions
            target_size = output_size
            
            # import ipdb; ipdb.set_trace()
            source_points = rescale_points(source_points, output_size, load_size)
            predicted_points = rescale_points(predicted_points, output_size, load_size)
            target_points = rescale_points(target_points, output_size, load_size)
            # import ipdb; ipdb.set_trace()
            dist, pck_img, sample_pck_img = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold)
            # _, pck_bbox, sample_pck_bbox = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold, target_bounding_box=ann["target_bounding_box"])
            
            px_th_dict = px_th_noquery(
                torch.from_numpy(target_points).reshape(1, -1, 1, 2),
                torch.from_numpy(predicted_points).reshape(1, -1, 1, 2)
            )
            if all_px_th_dict is None:
                all_px_th_dict = px_th_dict
            else:
                for key, value in px_th_dict.items():
                    all_px_th_dict[key] += value
            # wandb.log({"val/sample_pck_img": sample_pck_img}, step=j)
            # wandb.log({"val/sample_pck_bbox": sample_pck_bbox}, step=j)
            val_dist.append(dist)
            val_pck_img.append(pck_img)
            # val_pck_bbox.append(pck_bbox)
            ids.append([j] * len(dist))
            
            if dump_pairs:
                
                dump_dir = "./dumps"
                os.makedirs(dump_dir, exist_ok=True)
                dump_fn = os.path.join(dump_dir, f"pair_{j}.npz")
                np.savez(dump_fn, 
                         source_points=source_points, 
                         target_points=target_points, 
                         predicted_points=predicted_points,
                         img1=np.array(img1_pil),
                         img2=np.array(img2_pil)
                )
            
            
            if plot_every_n_steps > 0 and j % plot_every_n_steps == 0:
                title = f"pck@{pck_threshold}_img: {sample_pck_img.round(decimals=2)}"
                # title += f"\npck@{pck_threshold}_bbox: {sample_pck_bbox.round(decimals=2)}"
                # import ipdb; ipdb.set_trace()
                draw_correspondences(source_points, predicted_points, img1_pil, img2_pil, title=title)
                wandb.log({"val/correspondences": plt}, step=train_step)
    
    for key, value in all_px_th_dict.items():
        all_px_th_dict[key] = value / (j+1)
        wandb.log({f"val/{key}": all_px_th_dict[key]}, step=train_step)
    val_loss /= (j+1)
    wandb.log({"val/loss": val_loss}, step=train_step)
    
    ids = np.concatenate(ids)
    val_dist = np.concatenate(val_dist)
    val_pck_img = np.concatenate(val_pck_img)
    # val_pck_bbox = np.concatenate(val_pck_bbox)
    # df = pd.DataFrame({
    #     "id": ids,
    #     "distances": val_dist,
    #     "pck_img": val_pck_img,
    #     # "pck_bbox": val_pck_bbox,
    # })
    # wandb.log({"val/pck_img": val_pck_img.sum() / len(val_pck_img)})
    # wandb.log({"val/pck_bbox": val_pck_bbox.sum() / len(val_pck_bbox)})
    # wandb.log({f"val/distances_csv": wandb.Table(dataframe=df)})

def train(config, diffusion_extractor, aggregation_network, optimizer, train_anns, val_anns):
    device = config.get("device", "cuda")
    output_size, load_size = get_rescale_size(config)
    np.random.seed(0)
    for epoch in range(config["max_epochs"]):
        epoch_train_anns = np.random.permutation(train_anns)[:config["max_steps_per_epoch"]]
        for i, ann in tqdm(enumerate(epoch_train_anns)):
            step = epoch * config["max_steps_per_epoch"] + i
            optimizer.zero_grad()
            source_points, target_points, _, _, imgs = load_image_pair(ann, load_size, device, image_path=config["image_path"])
            if config.get("use_paper_size", False):
                # In the paper we set load_size = 64, output_size = 64 during training
                # and load_size = 224, output_size = 64 during testing to maintain a fair
                # comparison with DINO descriptors.
                # However, one could also set load_size = 512, output_size = 64 to use the
                # max possible resolution supported by Stable Diffusion, which is our
                # recommended setting when training for your use case.
                assert load_size == output_size, "Load and output resolution should be the same for use_paper_size."
                source_points, target_points, _, _, imgs = load_image_pair(ann, load_size, device, image_path=config["image_path"])
            else:
                # Resize input images to load_size and rescale points to output_size.
                source_points, target_points, _, _, imgs = load_image_pair(ann, load_size, device, image_path=config["image_path"], output_size=output_size)
            img1_hyperfeats, img2_hyperfeats = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            loss.backward()
            optimizer.step()
            wandb.log({"train/loss": loss.item()}, step=step)
            if step > 0 and config["val_every_n_steps"] > 0 and step % config["val_every_n_steps"] == 0:
                with torch.no_grad():
                    log_aggregation_network(aggregation_network, config)
                    save_model(config, aggregation_network, optimizer, step)
                    validate_modified(config, diffusion_extractor, aggregation_network, val_anns)

# 05/05/2024
def train_modified(config, aggregation_network, optimizer, pair_info, val_anns):
    # TODO : Pseudo pair_info from pair_info which is a dataset of img_pair_idx for which feat maps exist
    
    pt_files = glob.glob(os.path.join(config["proj_md_path"], '**', '*.pt'), recursive=True)
    existing_data = {int(os.path.basename(file).split('_')[-1].replace('.pt', '')) for file in pt_files}
    condition = pair_info['img_id0'].isin(existing_data) & pair_info['img_id1'].isin(existing_data)
    train_anns = pair_info[condition]
    ###############################
    device = config.get("device", "cuda")
    output_size, load_size = get_rescale_size(config)
    np.random.seed(0)
    for epoch in range(config["max_epochs"]):
        epoch_train_anns = train_anns.sample(config["max_steps_per_epoch"] , random_state=None)
        counter = 0
        for i in list(epoch_train_anns.index):
            step = epoch * config["max_steps_per_epoch"] + counter
            optimizer.zero_grad()
            source_points, target_points, _, _, imgs = load_image_pair_modified(i, train_anns, load_size, device, output_size=output_size)
            img1_hyperfeats, img2_hyperfeats = get_hyperfeats_modified(i, train_anns, aggregation_network,device)
            loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            loss.backward()
            optimizer.step()
            wandb.log({"train/loss": loss.item()}, step=step)
            if config["val_every_n_steps"] > 0 and step % config["val_every_n_steps"] == 0:
                with torch.no_grad():
                    log_aggregation_network(aggregation_network, config)
                    save_model(config, aggregation_network, optimizer, step)
                    validate_modified(config, aggregation_network, val_anns, train_step=step)
            counter += 1
def load_models(config_path):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    device = config.get("device", "cuda")
    diffusion_extractor = DiffusionExtractor(config, device)
    dims = config.get("dims")
    if dims is None:
        dims = collect_dims(diffusion_extractor.unet, idxs=diffusion_extractor.idxs)
    if config.get("flip_timesteps", False):
        config["save_timestep"] = config["save_timestep"][::-1]
    aggregation_network = AggregationNetwork(
            projection_dim=config["projection_dim"],
            feature_dims=dims,
            device=device,
            save_timestep=config["save_timestep"],
            num_timesteps=config["num_timesteps"]
    )
    return config, diffusion_extractor, aggregation_network

def main(args):
    config, diffusion_extractor, aggregation_network = load_models(args.config_path)
    wandb.init(project=config["wandb_project"], name=config["wandb_run"])
    # wandb.run.name = f"{str(wandb.run.id)}_{wandb.run.name}"
    parameter_groups = [
        {"params": aggregation_network.mixing_weights, "lr": config["lr"]},
        {"params": aggregation_network.bottleneck_layers.parameters(), "lr": config["lr"]}
    ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config["weight_decay"])
    
    # if config.get("train_path"):
    assert config["batch_size"] == 2, "The loss computation compute_clip_loss assumes batch_size=2."
    # train_anns = json.load(open(config["train_path"]))
    # train(config, diffusion_extractor, aggregation_network, optimizer, train_anns, val_anns)
    pair_info = pd.read_csv(config["train_path"])
    # import ipdb; ipdb.set_trace()
    val_anns = pd.read_csv(config["val_path"])
    train_modified(config, aggregation_network, optimizer, pair_info, val_anns)
    # else:
    
    # if config.get("weights_path"):
    #     aggregation_network.load_state_dict(torch.load(config["weights_path"], map_location="cpu")["aggregation_network"])
    validate_modified(config, aggregation_network, val_anns, train_step=config["max_steps_per_epoch"] + 1, dump_pairs=True)
    wandb.finish()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/custom.yaml")
    args = parser.parse_args()
    main(args)