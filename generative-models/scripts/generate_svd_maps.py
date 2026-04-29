import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional
import time

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch, torchvision
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor

# Import datasets lazily inside extract() so unused dataset dependencies do not
# block runs for the selected dataset on Windows.
from torchvision.io.video import read_video
import torch.nn.functional as F
from sgm.modules.encoders.modules import FrozenOpenCLIPEmbedder2

from tqdm import tqdm

from sgm.modules.encoders.modules import FrozenOpenCLIPEmbedder


def load_model(
        config: str,
        device: str,
        num_frames: int,
        num_steps: int,
        verbose: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(torch.float16).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def preprocess(frames):
    frames = frames / 255.0
    frames = 2 * frames - 1
    frames = torch.einsum("t h w c -> t c h w", frames)

    T, C, H, W = frames.shape

    frames = F.interpolate(frames, size=(320, 576), mode='bilinear', align_corners=True)

    assert C == 3, f"expected 3 channels, got {C}"

    return frames.float()


def _to_spatial_grid(
    latents: list,
    mid_idxs: List[int],
    deep_idxs: List[int],
    grid_size: int,
) -> torch.Tensor:
    g = grid_size
    pooled = []
    for li in (mid_idxs + deep_idxs):
        lat = latents[li - 1].float()          # (T, C, H, W)
        lat_p = F.adaptive_avg_pool2d(lat, (g, g))   # (T, C, g, g)
        pooled.append(lat_p)

    feat_spatial = torch.cat(pooled, dim=1)    # (T, D_total, g, g)
    T, D_total = feat_spatial.shape[:2]
    feat = feat_spatial.permute(0, 2, 3, 1).reshape(T, g * g, D_total)
    return feat.detach().cpu()                 # (T, P, D)


def extract(
        input_path: str = "assets/test_image.png",
        input_type: str = "video",
        num_frames: Optional[int] = 3,
        feature_stride: int = 4,
        offset_frames: int = 8,
        num_steps: Optional[int] = None,
        version: str = "svd",
        fps_id: int = 6,
        motion_bucket_id: int = 127,
        cond_aug: float = 0.02,
        seed: int = 23,
        decoding_t: int = 14,
        device: str = "cuda",
        output_folder: Optional[str] = './results',
        exp_name: str = "generate_svd_feats",
        elevations_deg: Optional[float | List[float]] = 10.0,
        azimuths_deg: Optional[List[float]] = None,
        image_frame_ratio: Optional[float] = None,
        verbose: Optional[bool] = False,
        dataset: str = "thumos14",
        split: str = None,
        layer_res: str = "16x16",
        timestep: int = 0,
        resize: int = None,
        reduction: str = None,
        layer_idxs: List[int] = None,
        conditioning: str = None,
        start_idx: int = None,
        end_idx: int = None,
        diffusion_step: int = 23,
        diffusion_step2: int = None,
        add_noise: bool = True,
        shuffle: bool = False,
        show_time=False,
        num_episodes: int = 600,
        mid_layer_idxs: List[int] = None,
        deep_layer_idxs: List[int] = None,
        grid_size: int = 3,
        mid_layer_idxs2: List[int] = None,
        deep_layer_idxs2: List[int] = None,
):
    if layer_idxs is None:
        layer_idxs = list(range(1, 14))
    elif isinstance(layer_idxs, int):
        layer_idxs = [layer_idxs]

    if mid_layer_idxs is None:
        mid_layer_idxs = [5, 7]
    elif isinstance(mid_layer_idxs, int):
        mid_layer_idxs = [mid_layer_idxs]

    if deep_layer_idxs is None:
        deep_layer_idxs = [12]
    elif isinstance(deep_layer_idxs, int):
        deep_layer_idxs = [deep_layer_idxs]

    # step2 层配置：默认与 step1 相同（保持向后兼容）
    if mid_layer_idxs2 is None:
        mid_layer_idxs2 = mid_layer_idxs
    elif isinstance(mid_layer_idxs2, int):
        mid_layer_idxs2 = [mid_layer_idxs2]

    if deep_layer_idxs2 is None:
        deep_layer_idxs2 = deep_layer_idxs
    elif isinstance(deep_layer_idxs2, int):
        deep_layer_idxs2 = [deep_layer_idxs2]

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "scripts/sampling/configs/svd.yaml"
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    elif version == "sd_v2":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt_sdv2.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    print(f"Conditioning: {conditioning}")
    print(f"Diffusion step: {diffusion_step}/{num_steps}" +
          (f"  +  step2={diffusion_step2}" if diffusion_step2 is not None else ""))
    print(f"Add noise: {add_noise}")
    print(f"Layer indices: {layer_idxs}")
    print(f"Version: {version}")
    print(f"Dataset: {dataset}")

    is_episodic = False

    build_dataset = None
    if dataset == 'thumos14':
        from datasets.thumos import ThumosDataset
        build_dataset = ThumosDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'animal_kingdom':
        from datasets.animal_kingdom import AnimalKingdomActionRecognitionDataset
        build_dataset = AnimalKingdomActionRecognitionDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'charades':
        from datasets.charades import CharadesDataset
        build_dataset = CharadesDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'phav':
        from datasets.phav import PhavDataset
        build_dataset = PhavDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'minisports':
        from datasets.minisports import MiniSportsDataset
        build_dataset = MiniSportsDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'ucfcrime':
        from datasets.ucfcrime import UCFCrimeDataset
        build_dataset = UCFCrimeDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'xdviolence':
        from datasets.xdviolence import XDViolenceDataset
        build_dataset = XDViolenceDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'ucf':
        from datasets.ucf import UCFDataset
        build_dataset = UCFDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'hmdb':
        from datasets.hmdb import HMDBDataset
        build_dataset = HMDBDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'hmdb51':
        from datasets.hmdb51 import HMDB51Dataset
        build_dataset = HMDB51Dataset
        dataset_kwargs = {"data_root": "../hmdb51/videos"}
    elif dataset == 'ucf101':
        from datasets.ucf101 import UCF101Dataset
        build_dataset = UCF101Dataset
        dataset_kwargs = {"data_root": "../ucf101/videos"}
    elif dataset == 'kinetics400':
        from datasets.kinetics400 import Kinetics400Dataset
        build_dataset = Kinetics400Dataset
        dataset_kwargs = {"data_root": "../kinetics400/videos"}
    elif dataset == 'epic_kitchens':
        from datasets.epic_kitchens import EpicKitchensDataset
        build_dataset = EpicKitchensDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'ssv2':
        from datasets.ssv2 import SSV2Dataset
        build_dataset = SSV2Dataset
        dataset_kwargs = {"split": split, "data_root": "../ssv2/videos"}
    elif dataset == 'ssv2_episodic':
        from datasets.ssv2 import SSV2EpisodicVideoDataset
        build_dataset = SSV2EpisodicVideoDataset
        is_episodic = True
        dataset_kwargs = {
            "split":        split if split in ("meta_train", "meta_val", "meta_test") else "meta_train",
            "video_root":   "../../ssv2/videos",
            "label_root":   "../../ssv2/labels",
            "n_way":        5,
            "k_shot":       1,
            "n_query":      15,
            "n_frames":     8,
            "num_episodes": num_episodes,
            "seed":         seed,
        }
    else:
        raise ValueError(f"Dataset {dataset} does not exist.")

    dataset = build_dataset(**dataset_kwargs)
    exp_dir = os.path.join(output_folder, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # hmdb51 / ucf101: save spatial grid features (T, P, D) to feats/grid_{g}x{g}/
    _use_grid = (dataset.__class__.__name__ in ("HMDB51Dataset", "UCF101Dataset", "Kinetics400Dataset"))

    feats_dir = (os.path.join(exp_dir, f"feats_{reduction}") if reduction is not None
                 else os.path.join(exp_dir, "feats"))
    os.makedirs(feats_dir, exist_ok=True)

    if _use_grid:
        _grid_subdir = os.path.join(feats_dir, f"grid_{grid_size}x{grid_size}")
        os.makedirs(_grid_subdir, exist_ok=True)
    elif reduction is None:
        if diffusion_step2 is not None:
            os.makedirs(os.path.join(feats_dir, "combined"), exist_ok=True)
        else:
            for layer_idx in layer_idxs:
                os.makedirs(os.path.join(feats_dir, f"layer_{layer_idx}"), exist_ok=True)

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    del model.first_stage_model.decoder
    torch.cuda.empty_cache()
    torch.manual_seed(seed)

    if conditioning == "actions":
        cond_embedding = []
        cond_model = FrozenOpenCLIPEmbedder2(always_return_pooled=True, legacy=False).to('cuda')
        for action in dataset.actions:
            _, action_feats = cond_model(action)
            cond_embedding.append(action_feats)
        cond_embedding = torch.cat(cond_embedding)
        cond_embedding = cond_embedding.expand(num_frames, -1, -1)

        _, uncond_embedding = cond_model("")
        uncond_embedding = uncond_embedding.expand(num_frames, len(dataset.actions), -1)

        del cond_model

    elif conditioning == "empty_str":
        cond_model = FrozenOpenCLIPEmbedder2(always_return_pooled=True, legacy=False).to('cuda')
        _, uncond_embedding = cond_model("")
        uncond_embedding = uncond_embedding.expand(num_frames, len(dataset.actions), -1)
        cond_embedding = uncond_embedding

        del cond_model

    print(f"Dataset length: {len(dataset)} videos")

    if is_episodic:
        from torch.utils.data import DataLoader as _EpisodicLoader
        from torch.utils.data import Subset

        existing_eps = glob(os.path.join(feats_dir, "episode_*.pt"))
        start_ep_idx = 0
        if existing_eps:
            indices = [int(os.path.splitext(os.path.basename(f))[0].split('_')[1]) for f in existing_eps]
            start_ep_idx = max(indices) + 1

        print(f"[Episodic] 检测到已存在 {start_ep_idx} 个 episode。")

        if start_ep_idx >= len(dataset):
            print("[Episodic] 所有 Episode 均已提取完毕！")
            return
        else:
            print(f"[Episodic] 将从 episode_{start_ep_idx:04d} 直接继续提取...")

        remaining_dataset = Subset(dataset, range(start_ep_idx, len(dataset)))

        episodic_loader = _EpisodicLoader(
            remaining_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )

        def _to_fp16(obj):
            if isinstance(obj, torch.Tensor) and obj.is_floating_point():
                return obj.to(torch.float16)
            elif isinstance(obj, dict):
                return {k: _to_fp16(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_to_fp16(v) for v in obj)
            return obj

        def _extract_svd_feats_for_clip(clip_frames: torch.Tensor, diff_step_override: int = None) -> list:
            frames = clip_frames.to(device).to(torch.float16)
            T_in, _, H_in, W_in = frames.shape

            value_dict = {
                "cond_frames_without_noise": frames,
                "motion_bucket_id": motion_bucket_id,
                "fps_id": fps_id,
                "cond_aug": cond_aug,
                "cond_frames": frames + cond_aug * torch.randn_like(frames),
            }

            batch_ep, batch_uc_ep = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, T_in],
                T=T_in,
                device=device,
            )
            batch_ep = _to_fp16(batch_ep)
            batch_uc_ep = _to_fp16(batch_uc_ep)

            c_ep, uc_ep = model.conditioner.get_unconditional_conditioning(
                batch_ep,
                batch_uc=batch_uc_ep,
                force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
            )

            additional_ep = {
                "image_only_indicator": torch.zeros(2, T_in, device=device, dtype=torch.float16),
                "num_video_frames": batch_ep["num_video_frames"],
            }

            def _denoiser_ep(inp, sigma, c_in, return_latents=False):
                return model.denoiser(
                    model.model, inp, sigma, c_in,
                    return_latents=return_latents, **additional_ep
                )

            latents = get_svd_latents(
                frames=frames,
                model=model,
                denoiser=_denoiser_ep,
                cond=c_ep,
                uc=uc_ep,
                num_steps=num_steps,
                diff_step=diff_step_override if diff_step_override is not None else diffusion_step,
                add_noise=add_noise,
            )
            return latents

        with torch.no_grad():
            with torch.autocast(device):
                for i, (support_x, support_y, query_x, query_y) in enumerate(
                        tqdm(episodic_loader, desc="Episodes", initial=start_ep_idx, total=len(dataset))
                ):
                    episode_idx = start_ep_idx + i
                    ep_save_path = os.path.join(feats_dir, f"episode_{episode_idx:04d}.pt")

                    if os.path.exists(ep_save_path):
                        continue

                    support_x = support_x.squeeze(0)
                    support_y = support_y.squeeze(0)
                    query_x = query_x.squeeze(0)
                    query_y = query_y.squeeze(0)

                    def _extract_feat(clip_frames):
                        lat1 = _extract_svd_feats_for_clip(clip_frames, diff_step_override=diffusion_step)
                        feat1 = _to_spatial_grid(lat1, mid_layer_idxs, deep_layer_idxs, grid_size)
                        if diffusion_step2 is not None:
                            lat2 = _extract_svd_feats_for_clip(clip_frames, diff_step_override=diffusion_step2)
                            feat2 = _to_spatial_grid(lat2, mid_layer_idxs2, deep_layer_idxs2, grid_size)
                            return torch.cat([feat1, feat2], dim=-1)
                        return feat1

                    support_feats = []
                    for j in range(support_x.shape[0]):
                        support_feats.append(_extract_feat(support_x[j]))
                    support_feats = torch.stack(support_feats)

                    query_feats = []
                    for j in range(query_x.shape[0]):
                        query_feats.append(_extract_feat(query_x[j]))
                    query_feats = torch.stack(query_feats)

                    torch.save({
                        "support_feats": support_feats,
                        "support_y": support_y,
                        "query_feats": query_feats,
                        "query_y": query_y,
                        "grid_size": grid_size,
                    }, ep_save_path)

        print(f"[Episodic] 完成，特征已保存至 {feats_dir}")
        return

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(dataset)

    dataset_start_time = time.time()
    dataset_n_frmaes = 0

    with torch.no_grad():
        with torch.autocast(device):
            idx_range = list(range(start_idx, end_idx))
            if shuffle:
                np.random.shuffle(idx_range)
            for idx in tqdm(idx_range):

                video_start_time = time.time()

                if idx < 0:
                    continue

                video_name = dataset.get_meta(idx)['name']
                video_path = dataset.get_meta(idx)['path']
                video_kwargs = None
                if "kwargs" in dataset.get_meta(idx):
                    video_kwargs = dataset.get_meta(idx)["kwargs"]

                if _use_grid:
                    all_features_exist = os.path.exists(
                        os.path.join(_grid_subdir, f"{video_name}_grid.pt")
                    )
                elif diffusion_step2 is not None:
                    all_features_exist = os.path.exists(
                        os.path.join(feats_dir, "combined", f"{video_name}_mean.npy")
                    )
                else:
                    all_features_exist = all(
                        os.path.exists(os.path.join(feats_dir, f"layer_{layer_idx}", f"{video_name}_mean.npy"))
                        for layer_idx in layer_idxs
                    )

                if all_features_exist:
                    continue

                if input_type == "video":
                    if video_kwargs is not None:
                        video, _, _ = read_video(video_path, **video_kwargs)
                    else:
                        video, _, _ = read_video(video_path, start_pts=0, end_pts=180, pts_unit='sec')
                elif input_type == "frames":
                    frame_files = sorted(glob(os.path.join(video_path, "*.jpg")))
                    frames = []
                    for frame_file in frame_files:
                        frame = torchvision.io.read_image(frame_file)
                        frame = frame.permute(1, 2, 0)
                        frames.append(frame)
                    video = torch.stack(frames, dim=0)
                else:
                    raise ValueError(f"Invalid input_type: {input_type}. Must be either 'video' or 'frames'")

                video_feats = {layer_idx: [] for layer_idx in layer_idxs}
                video_feats_mean = {layer_idx: [] for layer_idx in layer_idxs}
                video_combined_feats = []
                video_grid_feats = []

                T, H, W, C = video.shape
                video = video[offset_frames:T:feature_stride]

                if len(video) == 0:
                    print(f"[Warning] Video {video_name} is empty or too short (T={T}). Skipping...")
                    continue

                pad_size = (num_frames - (len(video) % num_frames)) % num_frames
                video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_size), mode='constant', value=0)

                for frame_idx in range(0, len(video), num_frames):
                    frames = video[frame_idx:frame_idx + num_frames].to('cuda')
                    frames = preprocess(frames)

                    value_dict = {
                        "cond_frames_without_noise": frames,
                        "motion_bucket_id": motion_bucket_id,
                        "fps_id": fps_id,
                        "cond_aug": cond_aug,
                        "cond_frames": frames + cond_aug * torch.randn_like(frames)
                    }

                    batch, batch_uc = get_batch(
                        get_unique_embedder_keys_from_conditioner(model.conditioner),
                        value_dict,
                        [1, num_frames],
                        T=num_frames,
                        device=device,
                    )

                    def to_half_precision(obj):
                        if isinstance(obj, torch.Tensor) and obj.is_floating_point():
                            return obj.to(torch.float16)
                        elif isinstance(obj, dict):
                            return {k: to_half_precision(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [to_half_precision(v) for v in obj]
                        elif isinstance(obj, tuple):
                            return tuple(to_half_precision(v) for v in obj)
                        return obj

                    batch = to_half_precision(batch)
                    batch_uc = to_half_precision(batch_uc)
                    frames = to_half_precision(frames)
                    if 'cond_embedding' in locals(): cond_embedding = to_half_precision(cond_embedding)
                    if 'uncond_embedding' in locals(): uncond_embedding = to_half_precision(uncond_embedding)

                    c, uc = model.conditioner.get_unconditional_conditioning(
                        batch,
                        batch_uc=batch_uc,
                        force_uc_zero_embeddings=[
                            "cond_frames",
                            "cond_frames_without_noise",
                        ],
                    )

                    if conditioning == 'no_cond':
                        c = uc
                    elif conditioning in ['actions', 'empty_str']:
                        c['crossattn'] = cond_embedding
                        uc['crossattn'] = uncond_embedding
                    elif conditioning is not None:
                        raise ValueError(f"Conditioning {conditioning} does not exist.")

                    T, _, H, W = frames.shape
                    F_ = 8
                    C = 4
                    shape = (T, C, H // F_, W // F_)

                    randn = torch.randn(shape, device=device).to(torch.float16)

                    additional_model_inputs = {}
                    additional_model_inputs["image_only_indicator"] = torch.zeros(
                        2, num_frames
                    ).to(device).to(torch.float16)
                    additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                    def denoiser(input, sigma, c, return_latents=False):
                        return model.denoiser(
                            model.model, input, sigma, c,
                            return_latents=return_latents,
                            **additional_model_inputs
                        )

                    latents = get_svd_latents(frames=frames,
                                              model=model,
                                              denoiser=denoiser,
                                              cond=c,
                                              uc=uc,
                                              num_steps=num_steps,
                                              diff_step=diffusion_step,
                                              add_noise=add_noise)

                    if diffusion_step2 is not None:
                        latents2 = get_svd_latents(frames=frames,
                                                   model=model,
                                                   denoiser=denoiser,
                                                   cond=c,
                                                   uc=uc,
                                                   num_steps=num_steps,
                                                   diff_step=diffusion_step2,
                                                   add_noise=add_noise)

                    if pad_size > 0 and (frame_idx + num_frames > len(video) - pad_size):
                        latents = [latent[:-pad_size] for latent in latents]
                        if diffusion_step2 is not None:
                            latents2 = [latent[:-pad_size] for latent in latents2]

                    if _use_grid:
                        grid_feat = _to_spatial_grid(
                            latents, mid_layer_idxs, deep_layer_idxs, grid_size
                        )
                        video_grid_feats.append(grid_feat)
                    elif reduction is None:
                        if diffusion_step2 is not None:
                            parts = []
                            for li in (mid_layer_idxs + deep_layer_idxs):
                                parts.append(latents[li - 1][num_frames:].float().mean(dim=(-1, -2)))
                            for li in (mid_layer_idxs2 + deep_layer_idxs2):
                                parts.append(latents2[li - 1][num_frames:].float().mean(dim=(-1, -2)))
                            video_combined_feats.append(
                                torch.cat(parts, dim=-1).detach().cpu().numpy()
                            )
                        else:
                            for layer_idx in layer_idxs:
                                frame_feats = latents[layer_idx - 1][num_frames:]
                                frame_feats_mean = torch.mean(frame_feats, dim=(-1, -2))
                                video_feats_mean[layer_idx].append(frame_feats_mean.detach().cpu().numpy())

                if _use_grid:
                    if len(video_grid_feats) == 0:
                        print(f"[Warning] grid feats empty for {video_name}. Skipping.")
                        continue
                    grid_tensor = torch.cat(video_grid_feats, dim=0)
                    torch.save(
                        {"feats": grid_tensor, "grid_size": grid_size},
                        os.path.join(_grid_subdir, f"{video_name}_grid.pt"),
                    )
                elif diffusion_step2 is not None:
                    if len(video_combined_feats) == 0:
                        print(f"[Warning] video_combined_feats is empty for {video_name}. Skipping save.")
                        continue
                    combined_feat = np.concatenate(video_combined_feats, axis=0)
                    with open(os.path.join(feats_dir, "combined", f"{video_name}_mean.npy"), 'wb') as f:
                        np.save(f, combined_feat)
                else:
                    if len(video_feats_mean[layer_idxs[0]]) == 0:
                        print(f"[Warning] video_feats_mean is empty for {video_name}. Skipping save.")
                        continue
                    for layer_idx in layer_idxs:
                        video_feats_mean[layer_idx] = np.concatenate(video_feats_mean[layer_idx], axis=0)
                        with open(os.path.join(feats_dir, f"layer_{layer_idx}", f"{video_name}_mean.npy"), 'wb') as f:
                            np.save(f, video_feats_mean[layer_idx])

                video_end_time = time.time()
                video_n_frames = len(video)
                print(f"Video frames/sec = {video_n_frames / (video_end_time - video_start_time):.2f}")

                dataset_n_frmaes += video_n_frames
                print(f"Dataset frames/sec = {dataset_n_frmaes / (video_end_time - dataset_start_time):.2f}")


def to_str(layer_idxs):
    return '_'.join(map(str, layer_idxs))


def get_svd_latents(frames, model, denoiser, cond, uc, num_steps, diff_step, add_noise):
    z_chunks = []
    chunk_size = 2
    for i in range(0, frames.shape[0], chunk_size):
        chunk = frames[i:i + chunk_size]
        z_chunks.append(model.encode_first_stage(chunk))
    z = torch.cat(z_chunks, dim=0)

    if hasattr(model, 'sampler') and hasattr(model.sampler, 'guider'):
        if hasattr(model.sampler.guider, 'num_frames'):
            model.sampler.guider.num_frames = frames.shape[0]

    _, s_in, sigmas, num_sigmas, cond, uc = prepare_extraction_loop(
        model.sampler, z, cond, uc, num_steps
    )

    t = diff_step

    gamma = (
        min(model.sampler.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
        if model.sampler.s_tmin <= sigmas[t] <= model.sampler.s_tmax
        else 0.0
    )

    if not add_noise:
        gamma = 0.0
        sigmas[t] = 0.0

    _, latents = model.sampler.sampler_step(
        s_in * sigmas[t],
        None,
        denoiser,
        z,
        cond,
        uc,
        gamma,
        return_latents=True,
    )

    return latents


def prepare_extraction_loop(self, x, cond, uc=None, num_steps=None):
    sigmas = self.discretization(
        self.num_steps if num_steps is None else num_steps, device=self.device
    )
    uc = default(uc, cond)

    num_sigmas = len(sigmas)

    s_in = x.new_ones([x.shape[0]])

    return x, s_in, sigmas, num_sigmas, cond, uc


if __name__ == "__main__":
    Fire(extract)

