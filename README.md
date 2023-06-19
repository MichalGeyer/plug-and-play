# Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation (CVPR 2023)

## [<a href="https://pnp-diffusion.github.io/" target="_blank">Project Page</a>] [<a href="https://github.com/MichalGeyer/pnp-diffusers" target="_blank">Diffusers Implementation</a>]

[![arXiv](https://img.shields.io/badge/arXiv-PnP-b31b1b.svg)](https://arxiv.org/abs/2211.12572) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hysts/PnP-diffusion-features) <a href="https://replicate.com/arielreplicate/plug_and_play_image_translation"><img src="https://replicate.com/arielreplicate/plug_and_play_image_translation/badge"></a> [![TI2I](https://img.shields.io/badge/benchmarks-TI2I-blue)](https://www.dropbox.com/sh/8giw0uhfekft47h/AAAF1frwakVsQocKczZZSX6La?dl=0)

![teaser](assets/teaser.png)

# Updates:

**19/06/23** 🧨 Diffusers implementation of Plug-and-Play is available [here](https://github.com/MichalGeyer/pnp-diffusers).

## TODO:
- [ ] Diffusers support and pipeline integration
- [ ] Gradio demo
- [x] Release TI2I Benchmarks


## Usage

**To plug-and-play diffusion features, please follow these steps:**

1. [Setup](#setup)
2. [Feature extraction](#feature-extraction)
3. [Running PnP](#running-pnp)
4. [TI2I Benchmarks](#ti2i-benchmarks)


## Setup

Our codebase is built on [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
and has shared dependencies and model architecture.

### Creating a Conda Environment

```
conda env create -f environment.yaml
conda activate pnp-diffusion
```

### Downloading StableDiffusion Weights

Download the StableDiffusion weights from the [CompVis organization at Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
(download the `sd-v1-4.ckpt` file), and link them:
```
mkdir -p models/ldm/stable-diffusion-v1/
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 
```


### Setting Experiment Root Path

The data of all the experiments is stored in a root directory.
The path of this directory is specified in `configs/pnp/setup.yaml`, under the `config.exp_path_root` key.


## Feature Extraction

For generating and extracting the features of an image, first set the parameters for the translation in a yaml config file.
An example of extraction configs can be found in `configs/pnp/feature-extraction-generated.yaml` for generated images
and in `configs/pnp/feature-extraction-real.yaml` for real images. Once the arguments are set, run:

```
python run_features_extraction.py --config <extraction_config_path>
```

For real images, the timesteps at which features are saved are determined by the `save_feature_timesteps` argument.
Note that for running PnP with `T` sampling steps for real images, you need to run the extraction with `save_feature_timesteps` = `T`
(since we're sampling with 999 steps for reconstructing the real image, we need to specify the timesteps at which features are saved).


After running the extraction script, an experiment folder is created in `<exp_path_root>/<source_experiment_name>`,
where `source_experiment_name` is specified by the config file. The experiment directory contains the following structure:
```
- <source_experiment_name>
    - feature_maps         # contains the extracted features
    - predicted_samples    # predicted clean images for each sampling timestep
    - samples              # contains the generated/inverted image
    - translations         # PnP translation results
    - z_enc.pt             # the initial noisy latent code
    - args.json            # the config arguments of the experiment
```

For visualizing the extracted features, see the [Feature Visualization](#feature-visualization) section.


## Running PnP

For running PnP, first set the parameters for the translation in a yaml config file.
An example of PnP config can be found in `configs/pnp/pnp-generated.yaml` for generated images
and in `configs/pnp/pnp-real.yaml` for real images. Once the arguments are set, run:

```
python run_pnp.py --config <pnp_config_path>
```

In the config parameters, you can control the following aspects in the translation:

- **Structure preservation** can be controlled by the `feature_injection_threshold` parameter
  (a higher value allows better structure preservation but can also leak details from the source image, ~80% of the total sampling steps generally gives a good tradeoff).
- **Deviation from the guidance image** can be controlled through the `scale`, `negative_prompt_alpha` and `negative_prompt_schedule` parameters (see the sample config files for details).
The effect of negative prompting is minor in case of realistic guidance images, but it can significantly help in case of minimalistic and abstract guidance images (e.g. segmentations).

Note that you can run a batch of translations by providing multiple target prompts in the `prompts`  parameter.

## Feature Visualization

### ResBlock Features Visualization
For running PCA visualizations on the extracted ResBlock features (Figure 3 in the paper),
first set the parameters for the visualization in a yaml config file.
An example of visualization config can be found in `configs/pnp/feature-pca-vis.yaml`.
Once the arguments are set, run:

```
python run_features_pca.py --config "<pca_vis_config_path>"
```

The feature visualizations are saved under `<config.exp_path_root>/PCA_features_vis/<experiment_name>` directory,
where `<experiment_name>` is specified in the visualization config file.


### Self-Attention Visualization


To visualize the self-attention maps of a generated/inverted image (Figure 6 in the paper), run: 
```
python run_self_attn_pca.py --block "<visualization_module_name>" --experiment "<experiment_name>"
```

The self-attention visualizations are saved under `<config.exp_path_root>/PCA_self_attention_vis/<experiment_name>` directory.


## TI2I Benchmarks

You can find the **Wild-TI2I**, **ImageNetR-TI2I** and **ImageNetR-Fake-TI2I** benchmarks in [this dropbox folder](https://www.dropbox.com/sh/8giw0uhfekft47h/AAAF1frwakVsQocKczZZSX6La?dl=0). The translation prompts and all the necessary configs (e.g. seed, generation prompt, guidance image path) are provided in a yaml file in each benchmark folder.


## Citation
```
@InProceedings{Tumanyan_2023_CVPR,
    author    = {Tumanyan, Narek and Geyer, Michal and Bagon, Shai and Dekel, Tali},
    title     = {Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {1921-1930}
}
```
