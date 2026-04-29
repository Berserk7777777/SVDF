"""
visualize_heatmap.py – SVD latent heatmap overlay visualizer.

For a given video file, extracts intermediate SVD U-Net latents at a specified
layer, collapses channels via |abs|-mean, Min-Max normalises, applies JET
pseudo-color, and blends 0.4 : 0.6 with the original BGR frame.  Output is
written as an MP4 video.

Usage example (run from generative-models/):

    python scripts/visualize_heatmap.py \\
        --input_path  /path/to/video.webm \\
        --output_path results/heatmaps/video_layer7.mp4 \\
        --layer_idx   7 \\
        --diffusion_step 23 \\
        --version svd_xt \\
        --device  cuda

Algorithm
---------
For each temporal chunk of num_frames video frames:
  1. Encode chunk through SVD VAE + partial denoising (get_svd_latents).
  2. Take the conditional half of the latent at ``layer_idx``:
         latent : (T, C, H_l, W_l)      C ≈ 320-1280 depending on layer
  3. Heatmap per frame:
         hmap = latent.abs().mean(dim=1)   → (T, H_l, W_l)
  4. Min-Max normalise each frame to [0, 255] uint8.
  5. Resize with OpenCV to original (W_frame, H_frame).
  6. Apply cv2.COLORMAP_JET → BGR uint8.
  7. Blend:  out = 0.4 * orig_bgr + 0.6 * jet_hmap   (uint8 clip).
  8. Append blended frame to output VideoWriter.
"""

import math
import os
import sys
from pathlib import Path
from typing import Optional, List

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from fire import Fire
from omegaconf import OmegaConf
from torchvision.io.video import read_video
from tqdm import tqdm

from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.util import default, instantiate_from_config


# ---------------------------------------------------------------------------
# Re-used helpers (identical to generate_svd_maps.py)
# ---------------------------------------------------------------------------

def _load_model(config_path: str, device: str, num_frames: int, num_steps: int):
    cfg = OmegaConf.load(config_path)
    if device == "cuda":
        cfg.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device
    cfg.model.params.sampler_config.params.verbose = False
    cfg.model.params.sampler_config.params.num_steps = num_steps
    cfg.model.params.sampler_config.params.guider_config.params.num_frames = num_frames
    if device == "cuda":
        with torch.device(device):
            model = (
                instantiate_from_config(cfg.model)
                .to(torch.float16)
                .to(device)
                .eval()
            )
    else:
        model = instantiate_from_config(cfg.model).to(device).eval()
    filt = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filt


def _get_unique_embedder_keys(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def _get_batch(keys, value_dict, N, T, device):
    batch, batch_uc = {}, {}
    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]]).to(device).repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device), "1 -> b", b=math.prod(N)
            )
        else:
            batch[key] = value_dict[key]
    if T is not None:
        batch["num_video_frames"] = T
    for k in batch:
        if k not in batch_uc and isinstance(batch[k], torch.Tensor):
            batch_uc[k] = torch.clone(batch[k])
    return batch, batch_uc


def _to_fp16(obj):
    if isinstance(obj, torch.Tensor) and obj.is_floating_point():
        return obj.to(torch.float16)
    if isinstance(obj, dict):
        return {k: _to_fp16(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_fp16(v) for v in obj)
    return obj


def _preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """uint8 (T, H, W, C) → float16 (T, C, 576, 1024)  in [-1, 1]."""
    frames = frames.float() / 255.0
    frames = 2.0 * frames - 1.0
    frames = frames.permute(0, 3, 1, 2)          # (T, C, H, W)
    frames = F.interpolate(frames, size=(576, 1024), mode="bilinear", align_corners=True)
    return frames.to(torch.float16)


def _prepare_extraction_loop(sampler, x, cond, uc, num_steps):
    sigmas = sampler.discretization(
        sampler.num_steps if num_steps is None else num_steps, device=sampler.device
    )
    uc = uc if uc is not None else cond
    num_sigmas = len(sigmas)
    s_in = x.new_ones([x.shape[0]])
    return x, s_in, sigmas, num_sigmas, cond, uc


def _get_svd_latents(frames, model, denoiser, cond, uc, num_steps, diff_step, add_noise):
    """Extract per-layer latents at a single diffusion timestep."""
    chunk_size = 2
    z_chunks = []
    for i in range(0, frames.shape[0], chunk_size):
        z_chunks.append(model.encode_first_stage(frames[i: i + chunk_size]))
    z = torch.cat(z_chunks, dim=0)

    if hasattr(model, "sampler") and hasattr(model.sampler, "guider"):
        if hasattr(model.sampler.guider, "num_frames"):
            model.sampler.guider.num_frames = frames.shape[0]

    _, s_in, sigmas, num_sigmas, cond, uc = _prepare_extraction_loop(
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
    return latents   # list[Tensor(2*T, C, H_l, W_l)], conditional = second half


# ---------------------------------------------------------------------------
# Core heatmap computation
# ---------------------------------------------------------------------------

def _latent_to_heatmap(
    latent: torch.Tensor,   # (T, C, H_l, W_l)  – conditional half
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """
    Convert a single-chunk latent to a (T, out_h, out_w, 3) uint8 BGR heatmap.

    Steps:
      1. abs().mean(channel)  → (T, H_l, W_l)
      2. Per-frame Min-Max normalise to [0, 255]
      3. Resize to (out_w, out_h) with INTER_LINEAR
      4. Apply JET colormap (returns BGR)
    """
    T = latent.shape[0]
    act = latent.abs().mean(dim=1)              # (T, H_l, W_l)
    act = act.float().cpu().numpy()             # float32

    frames_bgr = []
    for t in range(T):
        a = act[t]                              # (H_l, W_l)
        a_min, a_max = a.min(), a.max()
        denom = a_max - a_min
        if denom < 1e-8:
            a_norm = np.zeros_like(a, dtype=np.uint8)
        else:
            a_norm = ((a - a_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        a_resized = cv2.resize(a_norm, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        a_jet = cv2.applyColorMap(a_resized, cv2.COLORMAP_JET)   # (H, W, 3) BGR uint8
        frames_bgr.append(a_jet)

    return np.stack(frames_bgr, axis=0)    # (T, out_h, out_w, 3)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def visualize(
    input_path: str,
    output_path: str = "results/heatmap_output.mp4",
    layer_idx: int = 7,
    version: str = "svd_xt",
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    diffusion_step: int = 23,
    add_noise: bool = True,
    alpha_orig: float = 0.4,     # weight of original frame in blend
    alpha_heat: float = 0.6,     # weight of heatmap in blend
    device: str = "cuda",
    output_fps: int = 8,
):
    """
    Generate an SVD latent heatmap overlay video.

    Args:
        input_path    : Path to the input video file.
        output_path   : Path for the output MP4.
        layer_idx     : SVD U-Net layer index to visualise (1-13 for svd_xt).
        version       : SVD model version: 'svd' or 'svd_xt'.
        diffusion_step: Which noise timestep to extract features from (0-24).
        add_noise     : Whether to apply stochastic noise during extraction.
        alpha_orig    : Blend weight of the original frame (default 0.4).
        alpha_heat    : Blend weight of the heatmap overlay (default 0.6).
        device        : 'cuda' or 'cpu'.
        output_fps    : Frame rate of the output MP4.
    """
    assert abs(alpha_orig + alpha_heat - 1.0) < 1e-6, \
        "alpha_orig + alpha_heat must equal 1.0"

    # ── Model config ────────────────────────────────────────────────────────
    if version == "svd":
        num_frames   = default(num_frames, 14)
        num_steps    = default(num_steps, 25)
        model_config = "scripts/sampling/configs/svd.yaml"
    elif version == "svd_xt":
        num_frames   = default(num_frames, 25)
        num_steps    = default(num_steps, 30)
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    else:
        raise ValueError(f"Unsupported version: {version}")

    print(f"[visualize_heatmap] layer={layer_idx}  step={diffusion_step}  "
          f"version={version}  device={device}")

    torch.manual_seed(seed)

    # ── Load SVD model ───────────────────────────────────────────────────────
    print("Loading SVD model …")
    model, _ = _load_model(model_config, device, num_frames, num_steps)
    del model.first_stage_model.decoder
    torch.cuda.empty_cache()

    # ── Read video ───────────────────────────────────────────────────────────
    print(f"Reading video: {input_path}")
    video, _, info = read_video(input_path, start_pts=0, end_pts=180, pts_unit="sec")
    # video: (T, H, W, C)  uint8
    T_total, H_orig, W_orig, _ = video.shape
    in_fps = info.get("video_fps", output_fps)
    print(f"  {T_total} frames  {W_orig}×{H_orig}  @ {in_fps:.1f} fps")

    # ── Prepare output writer ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(output_fps), (W_orig, H_orig))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

    # Pad video length to multiple of num_frames
    pad_size = (num_frames - (T_total % num_frames)) % num_frames
    if pad_size > 0:
        pad = torch.zeros(pad_size, H_orig, W_orig, 3, dtype=video.dtype)
        video_padded = torch.cat([video, pad], dim=0)
    else:
        video_padded = video
    T_padded = video_padded.shape[0]

    # ── Process chunk by chunk ───────────────────────────────────────────────
    all_orig_bgr: List[np.ndarray] = []

    with torch.no_grad(), torch.autocast(device):
        for chunk_start in tqdm(range(0, T_padded, num_frames), desc="Chunks"):
            chunk = video_padded[chunk_start: chunk_start + num_frames]   # (T, H, W, C)
            frames_gpu = _preprocess_frames(chunk).to(device)             # (T, C, 576, 1024)
            T_in = frames_gpu.shape[0]

            # ── Build conditioning ──────────────────────────────────────
            value_dict = {
                "cond_frames_without_noise": frames_gpu,
                "motion_bucket_id":          motion_bucket_id,
                "fps_id":                    fps_id,
                "cond_aug":                  cond_aug,
                "cond_frames": frames_gpu + cond_aug * torch.randn_like(frames_gpu),
            }
            batch, batch_uc = _get_batch(
                _get_unique_embedder_keys(model.conditioner),
                value_dict,
                [1, T_in],
                T=T_in,
                device=device,
            )
            batch    = _to_fp16(batch)
            batch_uc = _to_fp16(batch_uc)

            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
            )

            additional = {
                "image_only_indicator": torch.zeros(2, T_in, device=device,
                                                    dtype=torch.float16),
                "num_video_frames":     batch["num_video_frames"],
            }

            def denoiser(inp, sigma, c_in, return_latents=False):
                return model.denoiser(
                    model.model, inp, sigma, c_in,
                    return_latents=return_latents, **additional
                )

            # ── Extract latents ──────────────────────────────────────────
            latents = _get_svd_latents(
                frames=frames_gpu,
                model=model,
                denoiser=denoiser,
                cond=c,
                uc=uc,
                num_steps=num_steps,
                diff_step=diffusion_step,
                add_noise=add_noise,
            )

            # latents[layer_idx-1] : (2*T_in, C, H_l, W_l)
            # conditional half = second T_in frames
            layer_latent = latents[layer_idx - 1][T_in:]    # (T_in, C, H_l, W_l)

            # Trim padding from last chunk
            usable = min(T_in, T_total - chunk_start)
            if usable <= 0:
                break
            layer_latent = layer_latent[:usable]

            # ── Heatmap ──────────────────────────────────────────────────
            heatmaps = _latent_to_heatmap(layer_latent, H_orig, W_orig)
            # heatmaps : (usable, H_orig, W_orig, 3) uint8 BGR

            # ── Blend with original frames ───────────────────────────────
            for t in range(usable):
                orig_rgb = chunk[t].numpy()                                   # (H, W, 3) uint8
                orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)         # uint8

                # Resize original to match SVD output resolution then back
                # (already native resolution, no resize needed)
                blended = (
                    alpha_orig * orig_bgr.astype(np.float32)
                    + alpha_heat * heatmaps[t].astype(np.float32)
                ).clip(0, 255).astype(np.uint8)

                writer.write(blended)

    writer.release()
    print(f"\nHeatmap video saved to: {output_path}")
    print(f"  Layer {layer_idx}  |  {T_total} frames  |  {W_orig}×{H_orig}")


# ---------------------------------------------------------------------------
# Multi-layer × multi-timestep grid visualiser
# ---------------------------------------------------------------------------

def visualize_grid(
    input_path: str,
    output_path: str = "results/heatmaps/grid_output.mp4",
    layer_idxs: str = "5,7,12",          # comma-separated, e.g. "5,7,12"
    diffusion_steps: str = "15,23",       # comma-separated, e.g. "15,23"
    version: str = "svd_xt",
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    add_noise: bool = True,
    alpha_orig: float = 0.4,
    alpha_heat: float = 0.6,
    device: str = "cuda",
    output_fps: int = 8,
    cell_w: int = 320,
    cell_h: int = 180,
):
    """
    Generate a grid heatmap video: rows = diffusion timesteps, cols = UNet layers.

    Each cell shows the SVD latent heatmap blended with the original frame.
    A label is burned into each cell header.

    Default grid (rows=timesteps × cols=layers):

        t=15 | L5 | L7 | L12
        t=23 | L5 | L7 | L12

    Args:
        layer_idxs      : Comma-separated layer indices, e.g. "5,7,12"
        diffusion_steps : Comma-separated timestep indices, e.g. "15,23"
        cell_w / cell_h : Pixel size of each grid cell in the output video.
    """
    layers = [int(x) for x in str(layer_idxs).split(",")]
    steps  = [int(x) for x in str(diffusion_steps).split(",")]
    n_rows = len(steps)
    n_cols = len(layers)

    print(f"[visualize_grid] layers={layers}  steps={steps}  "
          f"grid={n_rows}×{n_cols}  version={version}  device={device}")

    # ── Model config ─────────────────────────────────────────────────────────
    if version == "svd":
        num_frames   = default(num_frames, 14)
        num_steps    = default(num_steps, 25)
        model_config = "scripts/sampling/configs/svd.yaml"
    elif version == "svd_xt":
        num_frames   = default(num_frames, 25)
        num_steps    = default(num_steps, 30)
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    else:
        raise ValueError(f"Unsupported version: {version}")

    torch.manual_seed(seed)

    print("Loading SVD model …")
    model, _ = _load_model(model_config, device, num_frames, num_steps)
    del model.first_stage_model.decoder
    torch.cuda.empty_cache()

    print(f"Reading video: {input_path}")
    video, _, info = read_video(input_path, start_pts=0, end_pts=180, pts_unit="sec")
    T_total, H_orig, W_orig, _ = video.shape
    print(f"  {T_total} frames  {W_orig}×{H_orig}")

    # ── Output writer ─────────────────────────────────────────────────────────
    grid_w = n_cols * cell_w
    grid_h = n_rows * cell_h
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(output_fps), (grid_w, grid_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

    # ── Pad video ─────────────────────────────────────────────────────────────
    pad_size = (num_frames - (T_total % num_frames)) % num_frames
    if pad_size > 0:
        pad = torch.zeros(pad_size, H_orig, W_orig, 3, dtype=video.dtype)
        video_padded = torch.cat([video, pad], dim=0)
    else:
        video_padded = video

    # ── Font for cell labels ──────────────────────────────────────────────────
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thick = 1

    with torch.no_grad(), torch.autocast(device):
        for chunk_start in tqdm(range(0, video_padded.shape[0], num_frames), desc="Chunks"):
            chunk      = video_padded[chunk_start: chunk_start + num_frames]
            frames_gpu = _preprocess_frames(chunk).to(device)
            T_in       = frames_gpu.shape[0]
            usable     = min(T_in, T_total - chunk_start)
            if usable <= 0:
                break

            # ── Build conditioning (shared across timesteps) ──────────────
            value_dict = {
                "cond_frames_without_noise": frames_gpu,
                "motion_bucket_id":          motion_bucket_id,
                "fps_id":                    fps_id,
                "cond_aug":                  cond_aug,
                "cond_frames": frames_gpu + cond_aug * torch.randn_like(frames_gpu),
            }
            batch, batch_uc = _get_batch(
                _get_unique_embedder_keys(model.conditioner),
                value_dict, [1, T_in], T=T_in, device=device,
            )
            batch    = _to_fp16(batch)
            batch_uc = _to_fp16(batch_uc)
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch, batch_uc=batch_uc,
                force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
            )
            additional = {
                "image_only_indicator": torch.zeros(2, T_in, device=device,
                                                    dtype=torch.float16),
                "num_video_frames":     batch["num_video_frames"],
            }

            def denoiser(inp, sigma, c_in, return_latents=False):
                return model.denoiser(
                    model.model, inp, sigma, c_in,
                    return_latents=return_latents, **additional
                )

            # ── One forward pass per timestep → all layers ────────────────
            # step_heatmaps[row][col] = (usable, cell_h, cell_w, 3) uint8
            step_heatmaps = []
            for t_step in steps:
                latents = _get_svd_latents(
                    frames=frames_gpu, model=model, denoiser=denoiser,
                    cond=c, uc=uc, num_steps=num_steps,
                    diff_step=t_step, add_noise=add_noise,
                )
                row_heatmaps = []
                for l_idx in layers:
                    layer_lat = latents[l_idx - 1][T_in:][:usable]  # (usable, C, H_l, W_l)
                    hm = _latent_to_heatmap(layer_lat, cell_h, cell_w)  # (usable, cell_h, cell_w, 3)
                    row_heatmaps.append(hm)
                step_heatmaps.append(row_heatmaps)

            # ── Assemble grid frame by frame ──────────────────────────────
            for t in range(usable):
                orig_rgb = chunk[t].numpy()
                orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
                orig_cell = cv2.resize(orig_bgr, (cell_w, cell_h))

                grid_rows = []
                for r, t_step in enumerate(steps):
                    row_cells = []
                    for c_idx, l_idx in enumerate(layers):
                        hm_cell = step_heatmaps[r][c_idx][t]   # (cell_h, cell_w, 3)
                        blended = (
                            alpha_orig * orig_cell.astype(np.float32)
                            + alpha_heat * hm_cell.astype(np.float32)
                        ).clip(0, 255).astype(np.uint8)

                        # Burn label: "t=XX L=YY"
                        label = f"t={t_step} L{l_idx}"
                        cv2.putText(blended, label, (4, 16),
                                    font, font_scale, (255, 255, 255), font_thick,
                                    cv2.LINE_AA)
                        cv2.putText(blended, label, (4, 16),
                                    font, font_scale, (0, 0, 0), font_thick - 1,
                                    cv2.LINE_AA)
                        row_cells.append(blended)
                    grid_rows.append(np.concatenate(row_cells, axis=1))  # (cell_h, n_cols*cell_w, 3)

                grid_frame = np.concatenate(grid_rows, axis=0)  # (n_rows*cell_h, n_cols*cell_w, 3)
                writer.write(grid_frame)

    writer.release()
    print(f"\nGrid heatmap saved to: {output_path}")
    print(f"  Grid: {n_rows} timesteps × {n_cols} layers  |  {grid_w}×{grid_h}px  |  {T_total} frames")


# ---------------------------------------------------------------------------
# Dataset-aware entry point
# ---------------------------------------------------------------------------

def _resolve_video_path(dataset: str, data_root: str, class_name: str, video_name: str) -> str:
    """Return the full path to a video given dataset layout conventions."""
    if dataset in ("ssv2-small", "ssv2-full"):
        # SSV2: flat layout, video_name is the numeric ID (no class subfolder)
        for ext in (".webm", ".mp4"):
            p = os.path.join(data_root, f"{video_name}{ext}")
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"SSV2 video not found: {video_name} in {data_root}")
    elif dataset == "hmdb51":
        for ext in (".avi", ".mp4"):
            p = os.path.join(data_root, class_name, f"{video_name}{ext}")
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"HMDB51 video not found: {class_name}/{video_name}")
    elif dataset == "ucf101":
        for ext in (".avi", ".mp4"):
            p = os.path.join(data_root, class_name, f"{video_name}{ext}")
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"UCF101 video not found: {class_name}/{video_name}")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def visualize_dataset(
    dataset: str = "ssv2-small",
    # ── Data roots ──────────────────────────────────────────────────────────
    features_root: str = None,          # path to *_mean.npy or *_grid.pt files
    data_root: str = None,              # path to raw videos
    splits_root: str = None,            # path to splits/ subfolder
    # ── Which video to visualise ─────────────────────────────────────────────
    split: str = "test",                # base / val / test
    class_name: str = None,             # None = pick randomly
    video_name: str = None,             # None = pick randomly from class
    seed: int = 42,
    # ── SVD heatmap settings ─────────────────────────────────────────────────
    layer_idx: int = 7,
    version: str = "svd_xt",
    diffusion_step: int = 23,
    add_noise: bool = True,
    alpha_orig: float = 0.4,
    alpha_heat: float = 0.6,
    output_path: str = None,
    output_fps: int = 8,
    device: str = "cuda",
):
    """
    Dataset-aware heatmap visualiser.

    Picks a video from the specified dataset split, finds its raw video file,
    and runs the SVD latent heatmap overlay.

    Supported datasets: ssv2-small, ssv2-full, hmdb51, ucf101

    Example:
        python scripts/visualize_heatmap.py visualize_dataset \\
            --dataset hmdb51 \\
            --features_root results/hmdb51/extract_hmdb51/feats/combined \\
            --data_root ../hmdb51/videos \\
            --split test \\
            --device cuda
    """
    import random as _random
    import glob as _glob

    _random.seed(seed)
    _scripts_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Resolve defaults ─────────────────────────────────────────────────────
    _dataset_defaults = {
        "ssv2-small": {
            "features_root": "results/ssv2/extract_ssv2_svdxt_t20/feats/combined",
            "data_root":     "../ssv2/videos",
            "splits_root":   os.path.normpath(os.path.join(_scripts_dir, "../../splits/ssv2_small")),
            "split_file_map": {"base": "train_few_shot.txt", "val": "val_few_shot.txt",
                               "test": "test_few_shot.txt"},
            "feat_suffix":   "_mean.npy",
        },
        "ssv2-full": {
            "features_root": "results/ssv2/extract_ssv2_svdxt_t20/feats/combined",
            "data_root":     "../ssv2/videos",
            "splits_root":   os.path.normpath(os.path.join(_scripts_dir, "../../splits/ssv2_full")),
            "split_file_map": {"base": "train_few_shot.txt", "val": "val_few_shot.txt",
                               "test": "test_few_shot.txt"},
            "feat_suffix":   "_mean.npy",
        },
        "hmdb51": {
            "features_root": "results/hmdb51/extract_hmdb51/feats/combined",
            "data_root":     "../hmdb51/videos",
            "splits_root":   os.path.normpath(os.path.join(_scripts_dir, "../../splits/hmdb_ARN")),
            "split_file_map": {"base": "trainlist03.txt", "val": "vallist03.txt", "test": "testlist03.txt"},
            "feat_suffix":   "_mean.npy",
            "line_format":   "class/video",
        },
        "ucf101": {
            "features_root": "results/ucf101/extract_ucf101/feats/combined",
            "data_root":     "../UCF101/videos",
            "splits_root":   os.path.normpath(os.path.join(_scripts_dir, "../../splits/ucf_ARN")),
            "split_file_map": {"base": "trainlist03.txt", "val": "vallist03.txt", "test": "testlist03.txt"},
            "feat_suffix":   "_mean.npy",
            "line_format":   "class/video",
        },
    }

    if dataset not in _dataset_defaults:
        raise ValueError(f"dataset must be one of {list(_dataset_defaults.keys())}")

    cfg = _dataset_defaults[dataset]
    features_root = features_root or cfg["features_root"]
    data_root     = data_root     or cfg["data_root"]
    splits_root   = splits_root   or cfg["splits_root"]

    # ── Load class list from splits file ─────────────────────────────────────
    split_file = os.path.join(splits_root, cfg["split_file_map"][split])
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Splits file not found: {split_file}")

    with open(split_file, encoding="utf-8") as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    # Deduplicate class names for "class/video" format files (hmdb51, ucf101)
    if cfg.get("line_format") == "class/video":
        seen, classes = set(), []
        for line in raw_lines:
            cls = line.split("/", 1)[0]
            if cls not in seen:
                seen.add(cls)
                classes.append(cls)
    else:
        classes = raw_lines

    if dataset in ("ssv2-small", "ssv2-full"):
        # SSV2 split files list video IDs (one per line), not class names
        # Build class→videos from feature files
        all_feats = _glob.glob(os.path.join(features_root, f"*{cfg['feat_suffix']}"))
        if not all_feats:
            raise FileNotFoundError(f"No features found in {features_root}")
        # For SSV2 we just pick any feature file
        chosen_feat = _random.choice(all_feats)
        stem = os.path.basename(chosen_feat).replace(cfg["feat_suffix"], "")
        chosen_class = None
        chosen_video = stem
    else:
        # HMDB51 / UCF101: split files list class names
        if class_name is None:
            chosen_class = _random.choice(classes)
        else:
            chosen_class = class_name

        # Find feature files for this class
        class_feats = _glob.glob(
            os.path.join(features_root, f"{chosen_class}__*{cfg['feat_suffix']}")
        )
        if not class_feats:
            raise FileNotFoundError(
                f"No features for class '{chosen_class}' in {features_root}"
            )
        if video_name is None:
            chosen_feat = _random.choice(class_feats)
            stem = os.path.basename(chosen_feat).replace(cfg["feat_suffix"], "")
            # stem = "{class_name}__{video_stem}"
            chosen_video = stem.split("__", 1)[1] if "__" in stem else stem
        else:
            chosen_video = video_name

    print(f"[visualize_dataset] dataset={dataset}  split={split}")
    print(f"  class={chosen_class}  video={chosen_video}")

    # ── Resolve raw video path ────────────────────────────────────────────────
    video_path = _resolve_video_path(dataset, data_root, chosen_class, chosen_video)
    print(f"  video_path={video_path}")

    # ── Output path ───────────────────────────────────────────────────────────
    if output_path is None:
        tag = chosen_video if chosen_class is None else f"{chosen_class}__{chosen_video}"
        output_path = f"results/heatmaps/{dataset}/{tag}_layer{layer_idx}.mp4"

    visualize(
        input_path=video_path,
        output_path=output_path,
        layer_idx=layer_idx,
        version=version,
        diffusion_step=diffusion_step,
        add_noise=add_noise,
        alpha_orig=alpha_orig,
        alpha_heat=alpha_heat,
        output_fps=output_fps,
        device=device,
    )


if __name__ == "__main__":
    Fire({
        "visualize":         visualize,
        "visualize_dataset": visualize_dataset,
        "visualize_grid":    visualize_grid,
    })
