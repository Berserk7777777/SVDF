"""
visualize_unet_layers.py
========================
可视化 SVD UNet 所有层在指定 timestep 下对视频的"关注区域"热力图。

输出：
  - 每帧一张 PNG：行 = timestep，列 = UNet 层（1-13）
  - 合成 MP4 视频（所有帧拼接）

用法（从 generative-models/ 目录运行）：

    python scripts/visualize_unet_layers.py \\
        --input_path "../hmdb51/videos/swing_baseball/BaseballSwingAnalysis_swing_baseball_u_nm_np1_fr_med_19.avi" \\
        --output_dir results/heatmaps/unet_layers \\
        --diffusion_steps 15,23 \\
        --device cuda

    # 只看特定层（节省时间）
    python scripts/visualize_unet_layers.py \\
        --input_path "../hmdb51/videos/swing_baseball/BaseballSwingAnalysis_swing_baseball_u_nm_np1_fr_med_19.avi" \\
        --output_dir results/heatmaps/unet_layers \\
        --diffusion_steps 15,23 \\
        --layer_idxs 1,3,5,7,9,11,13 \\
        --device cuda
"""

import math
import os
import sys
from pathlib import Path

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from omegaconf import OmegaConf
from torchvision.io.video import read_video
from tqdm import tqdm

from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.util import default, instantiate_from_config


# ---------------------------------------------------------------------------
# SVD model helpers (reused from visualize_heatmap.py)
# ---------------------------------------------------------------------------

def _load_model(config_path, device, num_frames, num_steps):
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
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "motion_bucket_id":
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "cond_aug":
            batch[key] = repeat(torch.tensor([value_dict["cond_aug"]]).to(device), "1 -> b", b=math.prod(N))
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


def _preprocess_frames(frames):
    """uint8 (T,H,W,C) → float16 (T,C,576,1024) in [-1,1]."""
    frames = frames.float() / 255.0
    frames = 2.0 * frames - 1.0
    frames = frames.permute(0, 3, 1, 2)
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


def _get_svd_latents(frames, model, denoiser, cond, uc, num_steps, diff_step, add_noise=True):
    """Extract per-layer latents at a single diffusion timestep.
    Returns list of tensors, one per UNet layer."""
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
        s_in * sigmas[t], None, denoiser, z, cond, uc, gamma, return_latents=True,
    )
    return latents  # list[Tensor(2*T, C, H_l, W_l)]


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

def _latent_to_heatmap(latent, out_h, out_w):
    """(T, C, H_l, W_l) → (T, out_h, out_w, 3) uint8 BGR JET heatmap."""
    with torch.no_grad():
        act = latent.float().abs().mean(dim=1)  # (T, H_l, W_l)
    T = act.shape[0]
    out = np.zeros((T, out_h, out_w, 3), dtype=np.uint8)
    for t in range(T):
        frame = act[t].cpu().numpy()
        mn, mx = frame.min(), frame.max()
        if mx > mn:
            frame = (frame - mn) / (mx - mn)
        else:
            frame = np.zeros_like(frame)
        frame_u8 = (frame * 255).astype(np.uint8)
        frame_resized = cv2.resize(frame_u8, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        out[t] = cv2.applyColorMap(frame_resized, cv2.COLORMAP_JET)
    return out


def _blend(orig_bgr, heatmap_bgr, alpha_orig=0.45, alpha_heat=0.55):
    """Blend original frame with heatmap."""
    return np.clip(
        alpha_orig * orig_bgr.astype(np.float32) +
        alpha_heat * heatmap_bgr.astype(np.float32),
        0, 255
    ).astype(np.uint8)


def _burn_label(img, text, font_scale=0.38, thickness=1):
    """Burn white text with black shadow onto top-left of image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (3, 13)
    cv2.putText(img, text, pos, font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SVD UNet all-layer heatmap visualiser.")
    p.add_argument("--input_path",      type=str, required=True,
                   help="Path to input video file")
    p.add_argument("--output_dir",      type=str, default="results/heatmaps/unet_layers",
                   help="Directory to save output PNG frames and MP4")
    p.add_argument("--diffusion_steps", type=str, default="15,23",
                   help="Comma-separated timestep indices, e.g. '15,23'")
    p.add_argument("--layer_idxs",      type=str, default=None,
                   help="Comma-separated layer indices 1-13. Default: all 13 layers.")
    p.add_argument("--version",         type=str, default="svd_xt",
                   choices=["svd", "svd_xt"])
    p.add_argument("--num_frames",      type=int, default=None)
    p.add_argument("--num_steps",       type=int, default=None)
    p.add_argument("--fps_id",          type=int, default=6)
    p.add_argument("--motion_bucket_id",type=int, default=127)
    p.add_argument("--cond_aug",        type=float, default=0.02)
    p.add_argument("--seed",            type=int, default=23)
    p.add_argument("--add_noise",       action="store_true", default=True)
    p.add_argument("--alpha_orig",      type=float, default=0.45)
    p.add_argument("--alpha_heat",      type=float, default=0.55)
    p.add_argument("--cell_w",          type=int, default=192,
                   help="Width of each grid cell in pixels")
    p.add_argument("--cell_h",          type=int, default=108,
                   help="Height of each grid cell in pixels")
    p.add_argument("--output_fps",      type=int, default=6)
    p.add_argument("--save_frames",     action="store_true", default=False,
                   help="Also save individual PNG frames")
    p.add_argument("--device",          type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    steps = [int(x) for x in args.diffusion_steps.split(",")]
    if args.layer_idxs is not None:
        layers = [int(x) for x in args.layer_idxs.split(",")]
    else:
        layers = list(range(1, 14))  # 1-13

    n_rows = len(steps)   # timesteps
    n_cols = len(layers)  # layers
    cell_w, cell_h = args.cell_w, args.cell_h

    # Add one extra column for the original frame
    grid_w = (n_cols + 1) * cell_w
    grid_h = n_rows * cell_h

    print(f"Grid: {n_rows} timesteps × {n_cols} layers  (+1 original col)")
    print(f"Cell: {cell_w}×{cell_h}  →  Output: {grid_w}×{grid_h}")
    print(f"Timesteps: {steps}")
    print(f"Layers: {layers}")

    # ── Model config ──────────────────────────────────────────────────────────
    _scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if args.version == "svd":
        num_frames   = args.num_frames or 14
        num_steps    = args.num_steps  or 25
        model_config = os.path.join(_scripts_dir, "sampling/configs/svd.yaml")
    else:
        num_frames   = args.num_frames or 25
        num_steps    = args.num_steps  or 30
        model_config = os.path.join(_scripts_dir, "sampling/configs/svd_xt.yaml")

    device = args.device
    print(f"\nLoading SVD model ({args.version}) …")
    model, _ = _load_model(model_config, device, num_frames, num_steps)
    del model.first_stage_model.decoder
    torch.cuda.empty_cache()

    # ── Read video ────────────────────────────────────────────────────────────
    print(f"Reading video: {args.input_path}")
    video, _, info = read_video(args.input_path, start_pts=0, end_pts=180, pts_unit="sec")
    T_total, H_orig, W_orig, _ = video.shape
    print(f"  {T_total} frames  {W_orig}×{H_orig}")

    # ── Output setup ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    video_stem = Path(args.input_path).stem
    out_video_path = os.path.join(args.output_dir, f"{video_stem}_t{'_'.join(map(str,steps))}_all_layers.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, float(args.output_fps), (grid_w, grid_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {out_video_path}")

    # ── Pad video to multiple of num_frames ───────────────────────────────────
    pad_size = (num_frames - (T_total % num_frames)) % num_frames
    if pad_size > 0:
        pad = torch.zeros(pad_size, H_orig, W_orig, 3, dtype=video.dtype)
        video_padded = torch.cat([video, pad], dim=0)
    else:
        video_padded = video

    frame_idx = 0
    with torch.no_grad(), torch.autocast(device):
        for chunk_start in tqdm(range(0, video_padded.shape[0], num_frames), desc="Chunks"):
            chunk      = video_padded[chunk_start: chunk_start + num_frames]
            frames_gpu = _preprocess_frames(chunk).to(device)
            T_in       = frames_gpu.shape[0]
            usable     = min(T_in, T_total - chunk_start)
            if usable <= 0:
                break

            # ── Build conditioning ────────────────────────────────────────────
            value_dict = {
                "cond_frames_without_noise": frames_gpu,
                "motion_bucket_id":          args.motion_bucket_id,
                "fps_id":                    args.fps_id,
                "cond_aug":                  args.cond_aug,
                "cond_frames": frames_gpu + args.cond_aug * torch.randn_like(frames_gpu),
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
                "image_only_indicator": torch.zeros(2, T_in, device=device, dtype=torch.float16),
                "num_video_frames":     batch["num_video_frames"],
            }

            def denoiser(inp, sigma, c_in, return_latents=False):
                return model.denoiser(
                    model.model, inp, sigma, c_in,
                    return_latents=return_latents, **additional
                )

            # ── Extract latents for each timestep ────────────────────────────
            # step_heatmaps[row][col] = (usable, cell_h, cell_w, 3) uint8
            step_heatmaps = []
            for t_step in steps:
                latents = _get_svd_latents(
                    frames=frames_gpu, model=model, denoiser=denoiser,
                    cond=c, uc=uc, num_steps=num_steps,
                    diff_step=t_step, add_noise=args.add_noise,
                )
                n_available = len(latents)
                row_heatmaps = []
                for l_idx in layers:
                    if l_idx - 1 < n_available:
                        layer_lat = latents[l_idx - 1][T_in:][:usable]  # conditional half
                        hm = _latent_to_heatmap(layer_lat, cell_h, cell_w)
                    else:
                        # Layer not available — fill with gray
                        hm = np.full((usable, cell_h, cell_w, 3), 128, dtype=np.uint8)
                    row_heatmaps.append(hm)
                step_heatmaps.append(row_heatmaps)

            # ── Assemble grid frame by frame ──────────────────────────────────
            for t in range(usable):
                orig_rgb = chunk[t].numpy()
                orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
                orig_cell = cv2.resize(orig_bgr, (cell_w, cell_h))

                grid_rows = []
                for r, t_step in enumerate(steps):
                    # First column: original frame with timestep label
                    orig_labeled = orig_cell.copy()
                    _burn_label(orig_labeled, f"Original  t={t_step}", font_scale=0.4)
                    row_cells = [orig_labeled]

                    for c_idx, l_idx in enumerate(layers):
                        hm_cell = step_heatmaps[r][c_idx][t]
                        blended = _blend(orig_cell, hm_cell, args.alpha_orig, args.alpha_heat)
                        _burn_label(blended, f"L{l_idx}", font_scale=0.4)
                        row_cells.append(blended)

                    grid_rows.append(np.concatenate(row_cells, axis=1))

                grid_frame = np.concatenate(grid_rows, axis=0)
                writer.write(grid_frame)

                if args.save_frames:
                    frame_path = os.path.join(
                        args.output_dir, f"{video_stem}_frame{frame_idx:04d}.png"
                    )
                    cv2.imwrite(frame_path, grid_frame)

                frame_idx += 1

    writer.release()
    print(f"\nSaved: {out_video_path}")
    print(f"  {frame_idx} frames  |  grid {grid_w}×{grid_h}  |  {n_rows}×{n_cols} cells")
    print(f"\nTo save individual PNG frames, add --save_frames")


if __name__ == "__main__":
    main()
