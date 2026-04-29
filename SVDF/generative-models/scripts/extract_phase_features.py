"""
extract_phase_features.py — 三段式时序相位特征提取

将时序建模前移到特征提取阶段，对每个视频均匀划分三段：
    begin  : 视频前 1/3 帧
    middle : 视频中 1/3 帧
    end    : 视频后 1/3 帧

每段独立经过 SVD UNet，在指定扩散步和层提取特征图，
空间维度（H×W）+ 时序维度（T_phase）双重平均池化 → 单一特征向量。

输出（每个视频 3 个文件）:
    {feats_dir}/{video_name}_begin_mean.npy   shape: (D,)  float32
    {feats_dir}/{video_name}_middle_mean.npy  shape: (D,)  float32
    {feats_dir}/{video_name}_end_mean.npy     shape: (D,)  float32

支持任意帧长视频：边界用 round(T * k/3) 保证整数切分不丢帧。
断点续跑：跳过已存在的 .npy 文件。

用法（从 generative-models/ 目录运行）:
    python scripts/extract_phase_features.py \\
        --dataset hmdb51 \\
        --data_root ../hmdb51/videos \\
        --output_folder results/hmdb51 \\
        --exp_name extract_hmdb51_phase3 \\
        --layer_idx 7 \\
        --diff_step 20 \\
        --num_frames 25 \\
        --device cuda

    # SSV2 示例:
    python scripts/extract_phase_features.py \\
        --dataset ssv2 \\
        --data_root ../ssv2/videos \\
        --output_folder results/ssv2 \\
        --exp_name extract_ssv2_phase3 \\
        --device cuda
"""

import math
import os
import sys
from typing import List, Optional

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from fire import Fire
from omegaconf import OmegaConf
from torchvision.io.video import read_video
from tqdm import tqdm

from sgm.util import default, instantiate_from_config
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering


# ---------------------------------------------------------------------------
# SVD 模型加载与辅助函数（与 generate_svd_maps.py 保持一致）
# ---------------------------------------------------------------------------

def _load_svd_model(config_path: str, device: str, num_frames: int, num_steps: int):
    cfg = OmegaConf.load(config_path)
    if device == "cuda":
        cfg.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device
    cfg.model.params.sampler_config.params.verbose = False
    cfg.model.params.sampler_config.params.num_steps = num_steps
    cfg.model.params.sampler_config.params.guider_config.params.num_frames = num_frames
    model = instantiate_from_config(cfg.model).to(torch.float16).to(device).eval()
    return model


def _get_embedder_keys(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def _build_batch(keys, value_dict, N, T, device):
    batch, batch_uc = {}, {}
    for key in keys:
        if key == "fps_id":
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "motion_bucket_id":
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device), "1 -> b", b=math.prod(N)
            )
        else:
            batch[key] = value_dict[key]
    if T is not None:
        batch["num_video_frames"] = T
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def _to_fp16(obj):
    if isinstance(obj, torch.Tensor) and obj.is_floating_point():
        return obj.to(torch.float16)
    elif isinstance(obj, dict):
        return {k: _to_fp16(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_fp16(v) for v in obj)
    return obj


def _preprocess(frames: torch.Tensor) -> torch.Tensor:
    """(T, H, W, C) uint8 → (T, C, 320, 576) float16 in [-1, 1]."""
    x = frames.float() / 255.0 * 2.0 - 1.0
    x = torch.einsum("thwc->tchw", x)
    x = F.interpolate(x, size=(320, 576), mode="bilinear", align_corners=True)
    return x.half()


# ---------------------------------------------------------------------------
# 三段式划分：适配任意帧长
# ---------------------------------------------------------------------------

def split_three_phases(video: torch.Tensor):
    """
    将视频帧均匀划分为 begin / middle / end 三段。

    策略：用 round(T * k/3) 计算边界，保证三段覆盖全部帧且无重叠。
    边界退化保护：极短视频（T<3）时各段至少含 1 帧（允许复用）。

    Args:
        video : (T, H, W, C) torch.Tensor，原始视频帧

    Returns:
        list of 3 tensors，每个形状 (T_k, H, W, C)
    """
    T = len(video)
    boundaries = [int(round(T * k / 3)) for k in range(4)]  # [0, T/3, 2T/3, T]

    phases = []
    for i in range(3):
        s, e = boundaries[i], boundaries[i + 1]
        # 保证每段至少 1 帧
        if s >= e:
            s = max(0, e - 1)
        phases.append(video[s:e])
    return phases   # [begin_frames, middle_frames, end_frames]


# ---------------------------------------------------------------------------
# 单个相位的特征提取
# ---------------------------------------------------------------------------

def _extract_one_phase(
    phase_frames: torch.Tensor,   # (T_phase, H, W, C) uint8，CPU
    model,
    num_frames: int,
    layer_idx: int,               # 1-based UNet 层号（推荐 7）
    diff_step: int,               # 扩散步（推荐 20，共 25 步）
    fps_id: int,
    motion_bucket_id: int,
    cond_aug: float,
    device: str,
) -> np.ndarray:
    """
    对单段视频帧提取 SVD 特征向量。

    流程：
        1. 预处理：resize → [-1,1] float16
        2. 均匀采样至多 num_frames 帧（节省显存）
        3. VAE 编码到潜在空间
        4. 扩散采样 1 步（diff_step），获取 UNet 各层激活
        5. 取 layer_idx 层特征图，空间均值 → (T_clip, D)，时序均值 → (D,)

    Returns:
        (D,) float32 numpy array
    """
    # 1. 预处理
    frames = _preprocess(phase_frames).to(device)   # (T_phase, C, H', W')
    T_phase = frames.shape[0]

    # 2. 均匀采样，避免超长相位耗尽显存
    # 2. 均匀采样/插值对齐：无论长短，都强制采样到 num_frames，避免 SVD 位置编码维度不匹配
    if T_phase != num_frames:
        idxs = torch.linspace(0, T_phase - 1, num_frames).long()
        frames = frames[idxs]
    T_clip = frames.shape[0]

    # 3. 构造条件字典
    value_dict = {
        "cond_frames_without_noise": frames,
        "motion_bucket_id":          motion_bucket_id,
        "fps_id":                    fps_id,
        "cond_aug":                  cond_aug,
        "cond_frames":               frames + cond_aug * torch.randn_like(frames),
    }
    keys = _get_embedder_keys(model.conditioner)
    batch, batch_uc = _build_batch(keys, value_dict, [1, T_clip], T=T_clip, device=device)
    batch    = _to_fp16(batch)
    batch_uc = _to_fp16(batch_uc)

    c, uc = model.conditioner.get_unconditional_conditioning(
        batch, batch_uc=batch_uc,
        force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
    )

    # 4. VAE 编码（分块降低显存）
    z_list = []
    for i in range(0, T_clip, 2):
        z_list.append(model.encode_first_stage(frames[i:i + 2]))
    z = torch.cat(z_list, dim=0)

    # 修正 Guider 帧数（dynamic frame length support）
    if hasattr(model, "sampler") and hasattr(model.sampler, "guider"):
        if hasattr(model.sampler.guider, "num_frames"):
            model.sampler.guider.num_frames = T_clip

    additional = {
        "image_only_indicator": torch.zeros(2, T_clip, device=device, dtype=torch.float16),
        "num_video_frames":     batch["num_video_frames"],
    }

    def _denoiser(inp, sigma, c_in, return_latents=False):
        return model.denoiser(
            model.model, inp, sigma, c_in,
            return_latents=return_latents, **additional
        )

    # 5. 单步扩散采样，获取 UNet 层激活
    sigmas    = model.sampler.discretization(model.sampler.num_steps, device=device)
    s_in      = z.new_ones([z.shape[0]])
    num_sig   = len(sigmas)
    gamma = (
        min(model.sampler.s_churn / (num_sig - 1), 2 ** 0.5 - 1)
        if model.sampler.s_tmin <= sigmas[diff_step] <= model.sampler.s_tmax
        else 0.0
    )

    _, latents = model.sampler.sampler_step(
        s_in * sigmas[diff_step], None, _denoiser, z, c, uc, gamma, return_latents=True,
    )

    # 6. 提取目标层：latents 按 CFG 在 batch 维度拼接，取后半（条件分支）
    #    latents[layer_idx-1] shape: (2*T_clip, C, H_l, W_l)
    #    [T_clip:] 为条件分支（与 generate_svd_maps.py 保持一致）
    layer_feat = latents[layer_idx - 1][T_clip:].float()  # (T_clip, C, H_l, W_l)
    spatial_mean  = layer_feat.mean(dim=(-1, -2))          # (T_clip, C)
    phase_vec     = spatial_mean.mean(dim=0)               # (C,)  — D = C_layer7

    return phase_vec.detach().cpu().float().numpy()


# ---------------------------------------------------------------------------
# 主提取函数
# ---------------------------------------------------------------------------

def extract(
    # 数据集
    dataset:           str  = "hmdb51",
    data_root:         str  = "../hmdb51/videos",
    # 输出
    output_folder:     str  = "results/hmdb51",
    exp_name:          str  = "extract_hmdb51_phase3",
    # SVD 模型
    version:           str  = "svd_xt",
    layer_idx:         int  = 7,    # UNet 层（1-based），推荐 7
    diff_step:         int  = 20,   # 扩散步（0-based，共 25 步），推荐 20
    num_frames:        int  = 25,   # 每相位最多采样帧数（节省显存）
    fps_id:            int  = 6,
    motion_bucket_id:  int  = 127,
    cond_aug:          float = 0.02,
    device:            str  = "cuda",
    seed:              int  = 42,
    # 断点续跑
    start_idx:         Optional[int] = None,
    end_idx:           Optional[int] = None,
):
    """
    为数据集中每个视频提取三段式相位特征。

    每个视频输出 3 个 .npy 文件：
        {feats_dir}/{video_name}_begin_mean.npy
        {feats_dir}/{video_name}_middle_mean.npy
        {feats_dir}/{video_name}_end_mean.npy
    每个文件形状为 (D,)，D 为所选 UNet 层的通道数（layer 7 → D=640）。
    """
    torch.manual_seed(seed)
    phase_names = ["begin", "middle", "end"]

    # ── SVD 配置 ──────────────────────────────────────────────────────────
    if version == "svd_xt":
        num_steps    = 25
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    elif version == "svd":
        num_steps    = 25
        model_config = "scripts/sampling/configs/svd.yaml"
    else:
        raise ValueError(f"Unsupported version: {version}. Use 'svd' or 'svd_xt'.")

    print(f"[Phase3] Loading SVD model ({version}) …")
    model = _load_svd_model(model_config, device, num_frames, num_steps)
    del model.first_stage_model.decoder   # 释放解码器，节省显存
    torch.cuda.empty_cache()

    # ── 数据集扫描 ────────────────────────────────────────────────────────
    if dataset == "hmdb51":
        from datasets.hmdb51 import HMDB51Dataset
        ds = HMDB51Dataset(data_root=data_root)
    elif dataset == "ssv2":
        from datasets.ssv2 import SSV2Dataset
        ds = SSV2Dataset(split=None, data_root=data_root)
    else:
        # 通用扩展接口：新增数据集在此添加 elif 分支
        raise ValueError(
            f"Dataset '{dataset}' not yet supported in extract_phase_features.py. "
            "Add an elif branch with the corresponding dataset class."
        )

    # ── 输出目录 ──────────────────────────────────────────────────────────
    feats_dir = os.path.join(output_folder, exp_name, "feats")
    os.makedirs(feats_dir, exist_ok=True)

    print(f"[Phase3] Dataset  : {dataset}  ({len(ds)} videos)")
    print(f"[Phase3] UNet     : layer {layer_idx}  |  diff_step {diff_step}/{num_steps}")
    print(f"[Phase3] Output   : {feats_dir}")
    print(f"[Phase3] Per-video: {phase_names[0]}, {phase_names[1]}, {phase_names[2]}")

    # ── 提取主循环 ────────────────────────────────────────────────────────
    idx_range = range(
        start_idx if start_idx is not None else 0,
        end_idx   if end_idx   is not None else len(ds),
    )

    with torch.no_grad(), torch.autocast(device):
        for idx in tqdm(idx_range, desc="Phase3 extract"):
            meta       = ds.get_meta(idx)
            video_name = meta["name"]
            video_path = meta["path"]

            # 断点续跑：全部 3 个文件存在则跳过
            save_paths = {
                ph: os.path.join(feats_dir, f"{video_name}_{ph}_mean.npy")
                for ph in phase_names
            }
            if all(os.path.exists(p) for p in save_paths.values()):
                continue

            # 读取视频
            try:
                video, _, _ = read_video(video_path, start_pts=0, end_pts=180, pts_unit="sec")
            except Exception as exc:
                print(f"  [WARN] read failed: {video_path}  ({exc})")
                continue

            if len(video) < 3:
                print(f"  [WARN] too few frames ({len(video)}) — skipping {video_name}")
                continue

            # 三段式划分（适配任意帧长）
            phases = split_three_phases(video)   # [begin, middle, end] 各 (T_k, H, W, C)

            for ph_name, ph_frames in zip(phase_names, phases):
                out_path = save_paths[ph_name]
                if os.path.exists(out_path):
                    continue
                try:
                    feat = _extract_one_phase(
                        ph_frames, model, num_frames, layer_idx, diff_step,
                        fps_id, motion_bucket_id, cond_aug, device,
                    )
                    np.save(out_path, feat)
                except Exception as exc:
                    print(f"  [WARN] phase '{ph_name}' failed for {video_name}: {exc}")

    print(f"\n[Phase3] Done. Features saved → {feats_dir}")


if __name__ == "__main__":
    Fire(extract)
