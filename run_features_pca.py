import argparse, os
from tqdm import trange
import torch
from einops import rearrange
from pnp_utils import visualize_and_save_features_pca
from omegaconf import OmegaConf
import json
from run_features_extraction import load_model_from_config
import numpy as np
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler


def load_experiments_features(feature_maps_paths, block, feature_type, t):
    feature_maps = []
    for i, feature_maps_path in enumerate(feature_maps_paths):
        if "attn" in feature_type:
            feature_map = torch.load(os.path.join(feature_maps_path, f"{block}_{feature_type}_time_{t}.pt"))[8:]
            feature_map = rearrange(feature_map, 'h n d -> n (h d)')
        else:
            feature_map = \
                torch.load(os.path.join(feature_maps_path, f"{block}_{feature_type}_time_{t}.pt"))[1]
            feature_map = feature_map.reshape(feature_map.shape[0], -1).t()  # N X C
        feature_maps.append(feature_map)

    return feature_maps


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="configs/pnp/feature-pca-vis.yaml",
        help="path to the feature PCA visualization config file"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )

    opt = parser.parse_args()
    setup_config = OmegaConf.load("./configs/pnp/setup.yaml")
    exp_path_root = setup_config.config.exp_path_root
    exp_config = OmegaConf.load(f"{opt.config}")
    transform_experiments = exp_config.config.experiments_transform
    fit_experiments = exp_config.config.experiments_fit

    with open(os.path.join(exp_path_root, transform_experiments[0], "args.json"), "r") as f:
        args = json.load(f)
        ddim_steps = args["save_feature_timesteps"][-1]

    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0, verbose=False)
    time_range = np.flip(sampler.ddim_timesteps)
    total_steps = sampler.ddim_timesteps.shape[0]
    iterator = tqdm(time_range, desc="visualizing features", total=total_steps)

    print(f"visualizing features PCA experiments: block - {exp_config.config.block}; transform experiments - {exp_config.config.experiments_transform}; fit experiments - {exp_config.config.experiments_fit}")

    transform_feature_maps_paths = []
    for experiment in transform_experiments:
        transform_feature_maps_paths.append(os.path.join(exp_path_root, experiment, "feature_maps"))

    fit_feature_maps_paths = []
    for experiment in fit_experiments:
        fit_feature_maps_paths.append(os.path.join(exp_path_root, experiment, "feature_maps"))

    feature_types = [
        "in_layers_features",
        "out_layers_features",
        "self_attn_q",
        "self_attn_k"
    ]
    feature_pca_paths = {}

    pca_folder_path = os.path.join(exp_path_root, "PCA_features_vis", exp_config.config.experiment_name)
    os.makedirs(pca_folder_path, exist_ok=True)

    for feature_type in feature_types:
        feature_pca_path = os.path.join(pca_folder_path, f"{exp_config.config.block}_{feature_type}")
        feature_pca_paths[feature_type] = feature_pca_path
        os.makedirs(feature_pca_path, exist_ok=True)

    for t in iterator:
        for feature_type in feature_types:
            fit_features = load_experiments_features(fit_feature_maps_paths, exp_config.config.block, feature_type, t)  # N X C
            transform_features = load_experiments_features(transform_feature_maps_paths, exp_config.config.block, feature_type, t)
            visualize_and_save_features_pca(torch.cat(fit_features, dim=0),
                                            torch.cat(transform_features, dim=0),
                                            transform_experiments,
                                            t,
                                            feature_pca_paths[feature_type])


if __name__ == "__main__":
    main()
