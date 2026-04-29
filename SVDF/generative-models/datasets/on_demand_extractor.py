"""
on_demand_extractor.py
======================
按需特征提取器：训练时发现某个类的特征文件不存在，立即提取后继续。

核心类：
    OnDemandFewShotDataset  – 独立 FewShotDataset，从 splits 文件读取类列表，
                               在 __getitem__ 前检查所需特征是否存在，缺失则调用 SVD 提取。

用法（在 train_fsar.py 中）：
    from datasets.on_demand_extractor import OnDemandFewShotDataset

    dataset = OnDemandFewShotDataset(
        split="base",
        splits_root="../../splits/hmdb_ARN",
        split_file_map={"base": "trainlist03.txt", "val": "vallist03.txt", "test": "testlist03.txt"},
        line_format="class/video",          # or "class_only"
        features_root="results/hmdb51/extract_hmdb51/feats/combined",
        video_root="../hmdb51/videos",
        n_way=5, k_shot=1, n_query=15,
        num_episodes=6000,
        extractor_cfg=dict(
            diffusion_step=23,
            mid_layer_idxs=[7],
            deep_layer_idxs=[12],
            version="svd_xt",
            device="cuda",
        ),
    )
"""

import glob
import os
import random
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# SVD model loader (lazy singleton per process)
# ---------------------------------------------------------------------------

_SVD_MODEL_CACHE: dict = {}
_SVD_MODEL_LOCK = threading.Lock()


def _get_svd_model(version: str, device: str, num_frames: int, num_steps: int):
    key = (version, device, num_frames, num_steps)
    if key in _SVD_MODEL_CACHE:
        return _SVD_MODEL_CACHE[key]
    with _SVD_MODEL_LOCK:
        if key in _SVD_MODEL_CACHE:
            return _SVD_MODEL_CACHE[key]
        import math
        import sys
        _here = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.normpath(os.path.join(_here, ".."))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from omegaconf import OmegaConf
        from sgm.util import instantiate_from_config

        _scripts = os.path.join(_root, "scripts")
        if version == "svd_xt":
            cfg_path = os.path.join(_scripts, "sampling/configs/svd_xt.yaml")
        else:
            cfg_path = os.path.join(_scripts, "sampling/configs/svd.yaml")

        cfg = OmegaConf.load(cfg_path)
        if device == "cuda":
            cfg.model.params.conditioner_config.params.emb_models[
                0
            ].params.open_clip_embedding_config.params.init_device = device
        cfg.model.params.sampler_config.params.verbose = False
        cfg.model.params.sampler_config.params.num_steps = num_steps
        cfg.model.params.sampler_config.params.guider_config.params.num_frames = num_frames

        print(f"  [OnDemand] Loading SVD model ({version}) …", flush=True)
        if device == "cuda":
            with torch.device(device):
                model = (
                    instantiate_from_config(cfg.model)
                    .to(torch.float16).to(device).eval()
                )
        else:
            model = instantiate_from_config(cfg.model).to(device).eval()

        # Drop decoder to save VRAM
        if hasattr(model, "first_stage_model") and hasattr(model.first_stage_model, "decoder"):
            del model.first_stage_model.decoder
        torch.cuda.empty_cache()
        print(f"  [OnDemand] SVD model loaded.", flush=True)

        _SVD_MODEL_CACHE[key] = model
        return model


# ---------------------------------------------------------------------------
# SVD single-class feature extraction
# ---------------------------------------------------------------------------

def _extract_class_features(
    class_name: str,
    video_paths: List[str],
    features_root: str,
    model,
    num_frames: int,
    num_steps: int,
    diffusion_step: int,
    diffusion_step2: Optional[int],
    mid_layer_idxs: List[int],
    deep_layer_idxs: List[int],
    mid_layer_idxs2: List[int],
    deep_layer_idxs2: List[int],
    fps_id: int,
    motion_bucket_id: int,
    cond_aug: float,
    add_noise: bool,
    device: str,
    feat_suffix: str = "_mean.npy",
):
    """提取一个类的所有视频特征，保存到 features_root。"""
    import math
    from einops import repeat
    from torchvision.io.video import read_video

    os.makedirs(features_root, exist_ok=True)

    def _to_fp16(obj):
        if isinstance(obj, torch.Tensor) and obj.is_floating_point():
            return obj.to(torch.float16)
        if isinstance(obj, dict):
            return {k: _to_fp16(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_fp16(v) for v in obj)
        return obj

    def _preprocess(frames):
        frames = frames.float() / 255.0
        frames = 2.0 * frames - 1.0
        frames = frames.permute(0, 3, 1, 2)
        frames = F.interpolate(frames, size=(320, 576), mode="bilinear", align_corners=True)
        return frames.to(torch.float16)

    def _get_batch(keys, value_dict, N, T):
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

    def _get_latents(frames, diff_step):
        chunk_size = 2
        z_chunks = []
        for i in range(0, frames.shape[0], chunk_size):
            z_chunks.append(model.encode_first_stage(frames[i: i + chunk_size]))
        z = torch.cat(z_chunks, dim=0)

        if hasattr(model, "sampler") and hasattr(model.sampler, "guider"):
            if hasattr(model.sampler.guider, "num_frames"):
                model.sampler.guider.num_frames = frames.shape[0]

        sigmas = model.sampler.discretization(num_steps, device=device)
        num_sigmas = len(sigmas)
        s_in = z.new_ones([z.shape[0]])

        t = diff_step
        gamma = (
            min(model.sampler.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
            if model.sampler.s_tmin <= sigmas[t] <= model.sampler.s_tmax
            else 0.0
        )
        if not add_noise:
            gamma = 0.0
            sigmas[t] = 0.0

        T_in = frames.shape[0]
        value_dict = {
            "cond_frames_without_noise": frames,
            "motion_bucket_id": motion_bucket_id,
            "fps_id": fps_id,
            "cond_aug": cond_aug,
            "cond_frames": frames + cond_aug * torch.randn_like(frames),
        }
        keys = list(set([x.input_key for x in model.conditioner.embedders]))
        batch, batch_uc = _get_batch(keys, value_dict, [1, T_in], T_in)
        batch = _to_fp16(batch)
        batch_uc = _to_fp16(batch_uc)
        c, uc = model.conditioner.get_unconditional_conditioning(
            batch, batch_uc=batch_uc,
            force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
        )
        additional = {
            "image_only_indicator": torch.zeros(2, T_in, device=device, dtype=torch.float16),
            "num_video_frames": batch["num_video_frames"],
        }

        def denoiser(inp, sigma, c_in, return_latents=False):
            return model.denoiser(model.model, inp, sigma, c_in,
                                  return_latents=return_latents, **additional)

        _, latents = model.sampler.sampler_step(
            s_in * sigmas[t], None, denoiser, z, c, uc, gamma, return_latents=True,
        )
        return latents

    def _to_gap(latents, idxs, T_in):
        parts = []
        for li in idxs:
            parts.append(latents[li - 1][T_in:].float().mean(dim=(-1, -2)))
        return torch.cat(parts, dim=-1)  # (T, D)

    with torch.no_grad(), torch.autocast(device):
        for vpath in video_paths:
            stem = os.path.splitext(os.path.basename(vpath))[0]
            video_name = f"{class_name}__{stem}"
            out_path = os.path.join(features_root, f"{video_name}{feat_suffix}")
            if os.path.exists(out_path):
                continue

            try:
                video, _, _ = read_video(vpath, start_pts=0, end_pts=180, pts_unit="sec")
            except Exception as e:
                print(f"  [warn] Cannot read {vpath}: {e}")
                continue

            T_total = video.shape[0]
            if T_total == 0:
                continue

            video_s = video[0:T_total:4]
            pad_size = (num_frames - (len(video_s) % num_frames)) % num_frames
            if pad_size > 0:
                pad = torch.zeros(pad_size, *video_s.shape[1:], dtype=video_s.dtype)
                video_s = torch.cat([video_s, pad], dim=0)

            all_parts = []
            for fi in range(0, len(video_s), num_frames):
                chunk = video_s[fi: fi + num_frames].to(device)
                frames = _preprocess(chunk)
                T_in = frames.shape[0]

                usable = min(T_in, T_total - fi * 4)
                if usable <= 0:
                    break

                lat1 = _get_latents(frames, diffusion_step)
                feat1 = _to_gap(lat1, mid_layer_idxs + deep_layer_idxs, T_in)[:usable]

                if diffusion_step2 is not None:
                    lat2 = _get_latents(frames, diffusion_step2)
                    feat2 = _to_gap(lat2, mid_layer_idxs2 + deep_layer_idxs2, T_in)[:usable]
                    all_parts.append(torch.cat([feat1, feat2], dim=-1).cpu().numpy())
                else:
                    all_parts.append(feat1.cpu().numpy())

            if not all_parts:
                continue

            feat_arr = np.concatenate(all_parts, axis=0)  # (T, D)
            np.save(out_path, feat_arr)

    print(f"  [OnDemand] Extracted {class_name}: {len(video_paths)} videos → {features_root}")


# ---------------------------------------------------------------------------
# OnDemandFewShotDataset  (standalone, no pre-built base dataset required)
# ---------------------------------------------------------------------------

class OnDemandFewShotDataset(Dataset):
    """
    独立 FewShotDataset，从 splits 文件读取类列表，按需提取 SVD 特征。

    不需要预先运行 generate_svd_maps.py。首次访问某个类时自动提取并缓存到磁盘。

    参数：
        split           : "base" / "val" / "test"
        splits_root     : splits 文件目录
        split_file_map  : {"base": "trainlist03.txt", ...}
        line_format     : "class/video"（hmdb/ucf）或 "class_only"（ssv2）
        features_root   : 特征输出目录（也是读取目录）
        video_root      : 原始视频根目录（video_root/{class}/{video}.avi）
        extractor_cfg   : SVD 提取配置 dict
        n_way/k_shot/n_query/num_episodes/max_len/seed : 标准 few-shot 参数
        video_exts      : 视频扩展名
        feat_suffix     : 特征文件后缀（默认 "_mean.npy"）
    """

    def __init__(
        self,
        split: str,
        splits_root: str,
        split_file_map: Dict[str, str],
        line_format: str,               # "class/video" or "class_only"
        features_root: str,
        video_root: str,
        extractor_cfg: dict,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        num_episodes: int = 600,
        max_len: int = 50,
        seed: int = 42,
        video_exts: Tuple[str, ...] = (".avi", ".mp4", ".webm"),
        feat_suffix: str = "_mean.npy",
    ):
        self.split = split
        self.features_root = features_root
        self.video_root = video_root
        self.cfg = extractor_cfg
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.num_episodes = num_episodes
        self.max_len = max_len
        self.video_exts = video_exts
        self.feat_suffix = feat_suffix
        self._lock = threading.Lock()

        os.makedirs(features_root, exist_ok=True)

        # ── Read all class names from splits file ──────────────────────────
        split_file = os.path.join(splits_root, split_file_map[split])
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Splits file not found: {split_file}")

        with open(split_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        if line_format == "class/video":
            all_classes = sorted({l.split("/")[0] for l in lines if "/" in l})
        else:
            all_classes = sorted(set(lines))

        self._all_classes = all_classes
        print(f"[OnDemandFewShotDataset] split={split}  {len(all_classes)} classes in splits file")

        # ── Scan already-extracted classes ────────────────────────────────
        self._extracted: set = set()
        existing = glob.glob(os.path.join(features_root, f"*{feat_suffix}"))
        for f in existing:
            stem = os.path.basename(f).replace(feat_suffix, "")
            cls = stem.split("__")[0] if "__" in stem else stem
            self._extracted.add(cls)
        print(f"  {len(self._extracted)} classes already extracted.")

        # ── Feature cache (populated lazily) ──────────────────────────────
        self._feat_cache: Dict[str, torch.Tensor] = {}
        self._class_to_videos: Dict[str, List[str]] = {}

        # Load already-extracted features into cache
        self._load_extracted_classes(self._extracted & set(all_classes))

        # ── Infer embed_dim from cache (or set placeholder) ───────────────
        if self._feat_cache:
            sample = next(iter(self._feat_cache.values()))
            self.embed_dim = sample.shape[-1]
            self.n_patches = None  # GAP only for on-demand
        else:
            # Will be set after first extraction
            self.embed_dim = None
            self.n_patches = None

        self._rng = random.Random(seed + 1)
        self._episodes: Optional[List] = None  # generated lazily after first extraction

    # ------------------------------------------------------------------
    def _load_extracted_classes(self, class_names):
        """Load feature files for given classes into cache."""
        for cls in class_names:
            files = glob.glob(os.path.join(self.features_root, f"{cls}__*{self.feat_suffix}"))
            for fpath in files:
                vname = os.path.basename(fpath).replace(self.feat_suffix, "")
                if vname not in self._feat_cache:
                    arr = np.load(fpath).astype(np.float32)
                    feat = torch.from_numpy(arr).half()
                    if feat.ndim == 1:
                        feat = feat.unsqueeze(0)
                    self._feat_cache[vname] = feat
                    if cls not in self._class_to_videos:
                        self._class_to_videos[cls] = []
                    if vname not in self._class_to_videos[cls]:
                        self._class_to_videos[cls].append(vname)

    def _get_video_paths(self, class_name: str) -> List[str]:
        paths = []
        for ext in self.video_exts:
            paths.extend(glob.glob(os.path.join(self.video_root, class_name, f"*{ext}")))
        return sorted(paths)

    def _ensure_extracted(self, class_names: List[str]):
        """Extract features for any missing classes (thread-safe)."""
        missing = [c for c in class_names if c not in self._extracted]
        if not missing:
            return

        cfg = self.cfg
        model = _get_svd_model(
            version    = cfg.get("version", "svd_xt"),
            device     = cfg.get("device", "cuda"),
            num_frames = cfg.get("num_frames", 25),
            num_steps  = cfg.get("num_steps", 30),
        )

        for cls in missing:
            vpaths = self._get_video_paths(cls)
            if not vpaths:
                print(f"  [OnDemand] No videos for '{cls}' in {self.video_root}")
                with self._lock:
                    self._extracted.add(cls)  # mark to avoid retrying
                continue

            _extract_class_features(
                class_name       = cls,
                video_paths      = vpaths,
                features_root    = self.features_root,
                model            = model,
                num_frames       = cfg.get("num_frames", 25),
                num_steps        = cfg.get("num_steps", 30),
                diffusion_step   = cfg.get("diffusion_step", 23),
                diffusion_step2  = cfg.get("diffusion_step2", None),
                mid_layer_idxs   = cfg.get("mid_layer_idxs", [7]),
                deep_layer_idxs  = cfg.get("deep_layer_idxs", [12]),
                mid_layer_idxs2  = cfg.get("mid_layer_idxs2") or cfg.get("mid_layer_idxs", [7]),
                deep_layer_idxs2 = cfg.get("deep_layer_idxs2") or cfg.get("deep_layer_idxs", [12]),
                fps_id           = cfg.get("fps_id", 6),
                motion_bucket_id = cfg.get("motion_bucket_id", 127),
                cond_aug         = cfg.get("cond_aug", 0.02),
                add_noise        = cfg.get("add_noise", True),
                device           = cfg.get("device", "cuda"),
                feat_suffix      = self.feat_suffix,
            )

            with self._lock:
                self._extracted.add(cls)
                self._load_extracted_classes([cls])

        # Update embed_dim if not set yet
        if self.embed_dim is None and self._feat_cache:
            sample = next(iter(self._feat_cache.values()))
            self.embed_dim = sample.shape[-1]

    def _get_episode_class_ids(self) -> List[str]:
        """Classes that have enough extracted samples for an episode."""
        return sorted([
            c for c, vids in self._class_to_videos.items()
            if len(vids) >= self.k_shot + self.n_query
        ])

    def _generate_episodes(self) -> List:
        eligible = self._get_episode_class_ids()
        if len(eligible) < self.n_way:
            raise RuntimeError(
                f"Only {len(eligible)} classes have ≥{self.k_shot + self.n_query} features "
                f"(need n_way={self.n_way}). Extract more classes first."
            )
        episodes = []
        for _ in range(self.num_episodes):
            classes = self._rng.sample(eligible, self.n_way)
            support_videos, query_videos = [], []
            for c in classes:
                vids = self._rng.sample(self._class_to_videos[c], self.k_shot + self.n_query)
                support_videos.append(vids[: self.k_shot])
                query_videos.append(vids[self.k_shot:])
            episodes.append((classes, support_videos, query_videos))
        return episodes

    def _pad_or_truncate(self, feat: torch.Tensor):
        T = feat.shape[0]
        if T >= self.max_len:
            return feat[: self.max_len], torch.ones(self.max_len)
        pad_shape = (self.max_len - T,) + feat.shape[1:]
        pad  = torch.zeros(pad_shape)
        mask = torch.zeros(self.max_len)
        mask[:T] = 1.0
        return torch.cat([feat, pad], dim=0), mask

    # ------------------------------------------------------------------
    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        # Ensure episodes are generated (requires at least n_way classes extracted)
        if self._episodes is None:
            # Extract a bootstrap set of classes if needed
            eligible = self._get_episode_class_ids()
            if len(eligible) < self.n_way:
                needed = self.n_way - len(eligible)
                candidates = [c for c in self._all_classes if c not in self._extracted]
                bootstrap = candidates[:needed + self.n_way]  # extract a few extra
                self._ensure_extracted(bootstrap)
            self._episodes = self._generate_episodes()

        classes, support_videos, query_videos = self._episodes[idx % len(self._episodes)]

        # Ensure all episode classes are extracted
        self._ensure_extracted(classes)

        N = self.n_way
        K = self.k_shot
        Q = self.n_query
        T = self.max_len
        D = self.embed_dim

        support_features = torch.zeros(N, K, T, D)
        query_features   = torch.zeros(N * Q, T, D)
        support_masks    = torch.zeros(N, K, T)
        query_masks      = torch.zeros(N * Q, T)

        for i, (svids, qvids) in enumerate(zip(support_videos, query_videos)):
            for j, vname in enumerate(svids):
                if vname in self._feat_cache:
                    f, m = self._pad_or_truncate(self._feat_cache[vname].float())
                    support_features[i, j] = f
                    support_masks[i, j]    = m
            for q, vname in enumerate(qvids):
                if vname in self._feat_cache:
                    f, m = self._pad_or_truncate(self._feat_cache[vname].float())
                    query_features[i * Q + q] = f
                    query_masks[i * Q + q]    = m

        query_labels = torch.arange(N).repeat_interleave(Q)
        return support_features, support_masks, query_features, query_masks, query_labels
