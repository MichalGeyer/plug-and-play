import os
import subprocess
from typing import List, Optional
from argparse import Namespace
from tqdm import tqdm
from einops import rearrange

import numpy as np
import torch
from torch import autocast
from PIL import Image

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from contextlib import nullcontext

subprocess.run(["mkdir", "-p", "/root/.cache/torch/hub/checkpoints"])
subprocess.run(["cp", "-r", "huggingface", "/root/.cache"])
subprocess.run(["cp", "checkpoint_liberty_with_aug.pth", "/root/.cache/torch/hub/checkpoints"])
# https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth

# from pnp_utils import check_safety
from ldm.models.diffusion.ddim import DDIMSampler
from run_features_extraction import load_model_from_config, load_img

from cog import BasePredictor, Path, Input, BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["mkdir", "-p", "/root/.cache/torch/hub/checkpoints"])
        subprocess.run(["cp", "-r", "huggingface", "/root/.cache"])
        subprocess.run(["cp", "checkpoint_liberty_with_aug.pth", "/root/.cache/torch/hub/checkpoints"])

        common_config = Namespace()
        common_config.ddim_eta = 0.0
        common_config.H = common_config.W = 512
        common_config.C = 4  # Latent channels
        common_config.f = 8  # downsampling factor
        common_config.precision = "autocast"
        common_config.save_all_features = False
        common_config.check_safety = False

        model_config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
        self.model = load_model_from_config(model_config, "models/ldm/stable-diffusion-v1/model.ckpt")
        self.common_config = common_config

    def predict(
            self,
            input_image: Path = Input(description="Image to edit (instead of generation prompt"),
            # ddim_inversion_steps: int = Input(description="Number of forward steps in diffusion process for invering the image (if supplied)", default=999),
            generation_prompt: str = Input(description="Instead of input_image, generate an image from a text prompt"
                                                       " (Input image is ignored if this is supplied)",
                                           default=""),
            # num_ddim_steps: int = Input(description="Number of timesteps in the underlying diffusion process. default for generation from text is 50", ge=1, le=999, default=999),
            translation_prompts: str = Input(
                description="Text to Image prompts. A list of edit texts (separated by ';')"
                            " an image will be output for each edit txt",
                default="A photo of a robot horse"),
            scale: float = Input(
                description="Unconditional guidance scale. Note that a higher value encourages deviation from the source image "
                            "(10 is the default for tranlsation from image 7.5 for text", default=10.),
            feature_injection_threshold: float = Input(
                description="Control the level of structure preservation. What timestep to stop Injecting"
                            " the saved features into the translation diffusion process. "
                            "(0 is first and 1 is final timestep meaning more preservation) ", ge=0., le=1.,
                default=0.8),

            negative_prompt: str = Input(description="Control the level of deviation from the source image",
                                         default=""),
            negative_prompt_alpha: float = Input(description="Strength of the effect of the negative prompt "
                                                             "(lower is stronger)", ge=0., le=1., default=1.)

    ) -> List[Path]:
        self.common_config.generation_prompt = str(generation_prompt)

        extraction_config = Namespace()
        extraction_config.ddim_inversion_steps = 999

        pnp_config = Namespace()
        pnp_config.translation_prompts = str(translation_prompts).split(';')
        pnp_config.feature_injection_threshold = float(feature_injection_threshold)
        pnp_config.negative_prompt = str(negative_prompt)
        pnp_config.negative_prompt_alpha = float(negative_prompt_alpha)
        pnp_config.negative_prompt_schedule = "linear"  # ∈ {"linear", "constant", "exp"}, determines the attenuation schedule of negative-prompting
        # setting negative_prompt_alpha = 1.0, negative_prompt_schedule = "constant" is equivalent to not using negative prompting

        if generation_prompt == '':  # From Image
            self.common_config.seed = 50
            self.common_config.output_dir = "./outputs_real"

            # Extraction
            extraction_config.init_img = str(input_image)
            extraction_config.ddim_steps = 999
            extraction_config.save_feature_timesteps = 50
            extraction_config.scale = 1.0
            extract_features(self.model, self.common_config, extraction_config)

            # Translation
            pnp_config.scale = float(scale)
            pnp_config.num_ddim_sampling_steps = extraction_config.save_feature_timesteps
            image_paths = run_pnp(self.model, self.common_config, pnp_config)

            return [Path(x) for x in image_paths]

        else:  # From text
            self.common_config.seed = 50
            self.common_config.output_dir = "./outputs_gen"

            # Extraction
            extraction_config.init_img = ""
            extraction_config.save_feature_timesteps = extraction_config.ddim_steps = 50
            extraction_config.scale = 5.0
            gen_paths = extract_features(self.model, self.common_config, extraction_config)

            # Translation
            pnp_config.scale = float(scale)
            pnp_config.num_ddim_sampling_steps = extraction_config.save_feature_timesteps
            image_paths = run_pnp(self.model, self.common_config, pnp_config)

            return [Path(x) for x in gen_paths] + [Path(x) for x in image_paths]


def extract_features(model, opt, exp_config):
    seed_everything(opt.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)

    predicted_samples_path = os.path.join(opt.output_dir, "predicted_samples")
    feature_maps_path = os.path.join(opt.output_dir, "feature_maps")
    sample_path = os.path.join(opt.output_dir, "samples")
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(predicted_samples_path, exist_ok=True)
    os.makedirs(feature_maps_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)

    def save_sampled_img(x, i, save_path):
        x_samples_ddim = model.decode_first_stage(x)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
        x_sample = x_image_torch[0]
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(os.path.join(save_path, f"{i}.png"))

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_sampled_img(pred_x0, i, predicted_samples_path)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block in tqdm(blocks, desc="Saving input blocks feature maps"):
            if not opt.save_all_features and block_idx < 4:
                block_idx += 1
                continue
            if "ResBlock" in str(type(block[0])):
                if opt.save_all_features or block_idx == 4:
                    save_feature_map(block[0].in_layers_features,
                                     f"{feature_type}_{block_idx}_in_layers_features_time_{i}")
                    save_feature_map(block[0].out_layers_features,
                                     f"{feature_type}_{block_idx}_out_layers_features_time_{i}")
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                save_feature_map(block[1].transformer_blocks[0].attn1.k,
                                 f"{feature_type}_{block_idx}_self_attn_k_time_{i}")
                save_feature_map(block[1].transformer_blocks[0].attn1.q,
                                 f"{feature_type}_{block_idx}_self_attn_q_time_{i}")
            block_idx += 1

    def save_feature_maps_callback(i):
        if opt.save_all_features:
            save_feature_maps(unet_model.input_blocks, i, "input_block")
        save_feature_maps(unet_model.output_blocks, i, "output_block")

    def save_feature_map(feature_map, filename):
        save_path = os.path.join(feature_maps_path, f"{filename}.pt")
        torch.save(feature_map, save_path)

    prompts = [opt.generation_prompt]

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = model.get_learned_conditioning([""])
                c = model.get_learned_conditioning(prompts)
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                z_enc = None
                if exp_config.init_img != '':
                    assert os.path.isfile(exp_config.init_img)
                    init_image = load_img(exp_config.init_img).to(device)
                    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
                    z_enc, _ = sampler.encode_ddim(init_latent, num_steps=exp_config.ddim_inversion_steps,
                                                   conditioning=c, unconditional_conditioning=uc,
                                                   unconditional_guidance_scale=exp_config.scale)
                else:
                    z_enc = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
                torch.save(z_enc, f"{opt.output_dir}/z_enc.pt")

                samples_ddim, _ = sampler.sample(S=exp_config.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=1,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=exp_config.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 x_T=z_enc,
                                                 img_callback=ddim_sampler_callback,
                                                 callback_ddim_timesteps=exp_config.save_feature_timesteps,
                                                 outpath=opt.output_dir)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                # if opt.check_safety:
                #     x_samples_ddim = check_safety(x_samples_ddim)
                x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                sample_idx = 0
                png_paths = []
                for x_sample in x_image_torch:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    png_path = os.path.join(sample_path, f"{sample_idx}.png")
                    img.save(png_path)
                    png_paths.append(png_path)
                    sample_idx += 1

    print(f"Sampled images and extracted features saved in: {opt.output_dir}")
    return png_paths


def run_pnp(model, opt, exp_config):
    exp_config.feature_injection_threshold = int(
        exp_config.feature_injection_threshold * exp_config.num_ddim_sampling_steps)

    seed_everything(opt.seed)

    negative_prompt = opt.generation_prompt if exp_config.negative_prompt is None else exp_config.negative_prompt

    ddim_steps = exp_config.num_ddim_sampling_steps  # TODO in generated scenario this shoud ddim_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    seed = torch.initial_seed()
    opt.seed = seed

    translation_folders = [p.replace(' ', '_') for p in exp_config.translation_prompts]
    outpaths = [os.path.join(f"{opt.output_dir}/translations",
                             f"{exp_config.scale}_{translation_folder}") for translation_folder in translation_folders]
    out_label = f"INJECTION_T_{exp_config.feature_injection_threshold}_STEPS_{ddim_steps}"
    out_label += f"_NP-ALPHA_{exp_config.negative_prompt_alpha}_SCHEDULE_{exp_config.negative_prompt_schedule}_NP_{negative_prompt.replace(' ', '_')}"

    predicted_samples_paths = [os.path.join(outpath, f"predicted_samples_{out_label}") for outpath in outpaths]
    for i in range(len(outpaths)):
        os.makedirs(outpaths[i], exist_ok=True)
        os.makedirs(predicted_samples_paths[i], exist_ok=True)

    def save_sampled_img(x, i, save_paths):
        for im in range(x.shape[0]):
            x_samples_ddim = model.decode_first_stage(x[im].unsqueeze(0))
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
            x_sample = x_image_torch[0]

            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(save_paths[im], f"{i}.png"))

    def ddim_sampler_callback(pred_x0, xt, i):
        save_sampled_img(pred_x0, i, predicted_samples_paths)

    def load_target_features():
        self_attn_output_block_indices = [4, 5, 6, 7, 8, 9, 10, 11]
        out_layers_output_block_indices = [4]
        output_block_self_attn_map_injection_thresholds = [ddim_steps // 2] * len(self_attn_output_block_indices)
        feature_injection_thresholds = [exp_config.feature_injection_threshold]
        target_features = []

        source_experiment_out_layers_path = os.path.join(opt.output_dir, "feature_maps")
        source_experiment_qkv_path = os.path.join(opt.output_dir, "feature_maps")

        time_range = np.flip(sampler.ddim_timesteps)
        total_steps = sampler.ddim_timesteps.shape[0]

        iterator = tqdm(time_range, desc="loading source experiment features", total=total_steps)

        for i, t in enumerate(iterator):
            current_features = {}
            for (output_block_idx, output_block_self_attn_map_injection_threshold) in zip(
                    self_attn_output_block_indices, output_block_self_attn_map_injection_thresholds):
                if i <= int(output_block_self_attn_map_injection_threshold):
                    output_q = torch.load(os.path.join(source_experiment_qkv_path,
                                                       f"output_block_{output_block_idx}_self_attn_q_time_{t}.pt"))
                    output_k = torch.load(os.path.join(source_experiment_qkv_path,
                                                       f"output_block_{output_block_idx}_self_attn_k_time_{t}.pt"))
                    current_features[f'output_block_{output_block_idx}_self_attn_q'] = output_q
                    current_features[f'output_block_{output_block_idx}_self_attn_k'] = output_k

            for (output_block_idx, feature_injection_threshold) in zip(out_layers_output_block_indices,
                                                                       feature_injection_thresholds):
                if i <= int(feature_injection_threshold):
                    output = torch.load(os.path.join(source_experiment_out_layers_path,
                                                     f"output_block_{output_block_idx}_out_layers_features_time_{t}.pt"))
                    current_features[f'output_block_{output_block_idx}_out_layers'] = output

            target_features.append(current_features)

        return target_features

    batch_size = len(exp_config.translation_prompts)
    translation_prompts = exp_config.translation_prompts

    start_code_path = f"{opt.output_dir}/z_enc.pt"
    start_code = torch.load(start_code_path).cuda() if os.path.exists(start_code_path) else None
    if start_code is not None:
        start_code = start_code.repeat(batch_size, 1, 1, 1)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    injected_features = load_target_features()
    unconditional_prompt = ""
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = None
                nc = None
                if exp_config.scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [unconditional_prompt])
                    nc = model.get_learned_conditioning(batch_size * [negative_prompt])

                c = model.get_learned_conditioning(translation_prompts)
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 negative_conditioning=nc,
                                                 batch_size=len(translation_prompts),
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=exp_config.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 x_T=start_code,
                                                 img_callback=ddim_sampler_callback,
                                                 injected_features=injected_features,
                                                 negative_prompt_alpha=exp_config.negative_prompt_alpha,
                                                 negative_prompt_schedule=exp_config.negative_prompt_schedule,
                                                 )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                # if opt.check_safety:
                #     x_samples_ddim = check_safety(x_samples_ddim)
                x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                png_paths = []
                sample_idx = 0
                for k, x_sample in enumerate(x_image_torch):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    png_path = os.path.join(outpaths[k], f"{out_label}_sample_{sample_idx}.png")
                    png_paths.append(png_path)
                    img.save(png_path)
                    sample_idx += 1
    print(f"PnP results saved in: {'; '.join(outpaths)}")
    return png_paths
