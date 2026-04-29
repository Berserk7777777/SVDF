"""
datasets/ucf101.py – UCF101 dataset classes for SVD feature extraction and FSAR training.

UCF101 directory layout expected:
    ucf101/
    └── videos/
        ├── ApplyEyeMakeup/
        │   ├── v_ApplyEyeMakeup_g01_c01.avi
        │   └── ...
        ├── Archery/
        └── ...  (101 action classes total)

Video naming convention for feature files:
    {class_name}__{video_stem}_mean.npy
    (double underscore separator guarantees global uniqueness across 101 classes)

Few-shot class split (standard ARN splits):
    70 base / 10 val / 21 test  (total = 101)
    Read from splits/ucf_ARN/{train,val,test}list03.txt
"""

import glob
import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


# =============================================================================
# UCF101Dataset  –  for feature extraction via generate_svd_maps.py
# =============================================================================

class UCF101Dataset:
    """
    Scans all video files under data_root/{class_name}/*.avi and provides
    get_meta(idx) compatible with the generate_svd_maps.py extraction loop.

    video_name = '{class_name}__{video_stem}'  (flat, globally unique)
    Features are saved by the extractor as:
        feats/combined/{class_name}__{video_stem}_mean.npy
    """

    def __init__(
        self,
        split=None,                          # API compatibility; unused
        data_root: str = "../ucf101/videos",
        video_extensions=(".avi", ".mp4"),
    ):
        self.data_root = data_root
        self.actions = []                    # API compatibility

        self.meta: list = []
        for ext in video_extensions:
            for vpath in sorted(glob.glob(
                os.path.join(data_root, "**", f"*{ext}"), recursive=True
            )):
                class_name = os.path.basename(os.path.dirname(vpath))
                stem = os.path.splitext(os.path.basename(vpath))[0]
                video_name = f"{class_name}__{stem}"
                self.meta.append({"name": video_name, "path": vpath})

        self.meta.sort(key=lambda x: x["name"])
        print(f"[UCF101Dataset] {len(self.meta)} videos found under {data_root}")

    def __len__(self):
        return len(self.meta)

    def get_meta(self, idx):
        return self.meta[idx]

    def __getitem__(self, idx):
        video, _, _ = torchvision.io.read_video(self.meta[idx]["path"])
        return video


# =============================================================================
# UCF101FewShotDataset  –  for FSAR training via train_fsar.py
# =============================================================================

class UCF101FewShotDataset(Dataset):
    """
    Few-shot episode dataset for UCF101 FSAR.

    Class split: read from standard ARN splits in splits_root/
        trainlist03.txt → 70 base classes
        vallist03.txt   → 10 val classes
        testlist03.txt  → 21 test classes

    split='base'  → meta-training on base classes
    split='val'   → evaluation on val novel classes
    split='test'  → evaluation on test novel classes

    Feature formats (auto-detected from features_root):
        Grid (new, preferred): {features_root}/{class}__{stem}_grid.pt
            → shape (T, P, D), P=grid_size²  — sets self.n_patches = P
        GAP  (legacy):         {features_root}/{class}__{stem}_mean.npy
            → shape (T, D)                   — sets self.n_patches = None
    """

    def __init__(
        self,
        split: str = "base",
        features_root: str = "results/ucf101/extract_ucf101/feats/combined",
        data_root: str = "../ucf101/videos",        # kept for API compat
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        num_episodes: int = 600,
        max_len: int = 50,
        use_mean: bool = True,
        n_base_classes: int = 70,                   # ignored (splits file is authoritative)
        n_val_classes: int = 10,                    # ignored (splits file is authoritative)
        seed: int = 42,
        splits_root: str = None,                    # path to splits/ucf_ARN/; None = auto-detect
        video_extensions=(".avi", ".mp4"),
        feat_noise_std: float = 0.0,
        frame_drop_prob: float = 0.0,
    ):
        assert split in ("base", "val", "test"), f"split must be base/val/test, got {split}"

        self.features_root = features_root
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.num_episodes = num_episodes
        self.max_len = max_len
        self.use_mean = use_mean
        self.split = split
        self._noise_std = feat_noise_std
        self._frame_drop = frame_drop_prob
        self._augment = (split == "base") and (feat_noise_std > 0 or frame_drop_prob > 0)

        # ── 探测特征格式（grid .pt 优先于 gap .npy）────────────────────────
        _probe_pt  = glob.glob(os.path.join(features_root, "*_grid.pt"))
        _probe_npy = glob.glob(os.path.join(features_root, "*_mean.npy"))
        if _probe_pt:
            self._feat_format = "grid"
            _feat_suffix = "_grid.pt"
        elif _probe_npy:
            self._feat_format = "gap"
            _feat_suffix = "_mean.npy" if use_mean else ".npy"
        else:
            raise FileNotFoundError(
                f"No *_grid.pt or *_mean.npy files found in {features_root}"
            )

        # ── Resolve splits_root ────────────────────────────────────────────
        if splits_root is None:
            _here = os.path.dirname(os.path.abspath(__file__))
            splits_root = os.path.normpath(os.path.join(_here, "../../splits/ucf_ARN"))

        # ── Load class list from standard ARN splits file ──────────────────
        _split_file_map = {
            "base": "trainlist03.txt",
            "val":  "vallist03.txt",
            "test": "testlist03.txt",
        }
        split_file = os.path.join(splits_root, _split_file_map[split])
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Splits file not found: {split_file}\n"
                f"  Expected splits/ucf_ARN/ under project root."
            )

        with open(split_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        # ── Build class_name → [video_name, ...] from split file ──────────
        # Each line: "class_name/video_stem" (no extension)
        # Feature file: {features_root}/{class_name}__{video_stem}{_feat_suffix}
        class_to_videos: dict = {}
        for line in lines:
            parts = line.split("/", 1)
            if len(parts) != 2:
                continue
            class_name, video_stem = parts
            video_name = f"{class_name}__{video_stem}"
            feat_path = os.path.join(features_root, f"{video_name}{_feat_suffix}")
            if os.path.exists(feat_path):
                if class_name not in class_to_videos:
                    class_to_videos[class_name] = []
                class_to_videos[class_name].append(video_name)

        self.episode_class_ids = sorted([
            cls for cls, vids in class_to_videos.items()
            if len(vids) >= k_shot + n_query
        ])
        self.class_to_videos = {cls: class_to_videos[cls] for cls in self.episode_class_ids}

        assert len(self.episode_class_ids) >= n_way, (
            f"Only {len(self.episode_class_ids)} classes have enough samples "
            f"(need n_way={n_way}). Check features_root or lower k_shot/n_query."
        )

        # ── RAM cache ─────────────────────────────────────────────────────
        all_videos = list({v for vids in self.class_to_videos.values() for v in vids})
        print(f"  [UCF101FewShotDataset] Caching {len(all_videos)} {self._feat_format} features …",
              end=" ", flush=True)
        self._feat_cache: dict[str, torch.Tensor] = {}
        for vname in all_videos:
            fpath = os.path.join(features_root, f"{vname}{_feat_suffix}")
            if self._feat_format == "grid":
                data = torch.load(fpath, map_location="cpu")
                feat = data["feats"].half()
            else:
                arr  = np.load(fpath)
                feat = torch.from_numpy(arr).half()
                if feat.ndim == 1:
                    feat = feat.unsqueeze(0)
            self._feat_cache[vname] = feat
        print("done.")

        # ── 推断维度 ──────────────────────────────────────────────────────
        sample_feat = next(iter(self._feat_cache.values()))
        if self._feat_format == "grid":
            self.n_patches = sample_feat.shape[1]
            self.embed_dim = sample_feat.shape[2]
        else:
            self.n_patches = None
            self.embed_dim = sample_feat.shape[-1]

        t_lengths = [f.shape[0] for f in self._feat_cache.values()]
        auto_max_len = max(int(np.percentile(t_lengths, 95)), 1)
        self.max_len = max_len if max_len > 0 else auto_max_len

        self._rng = random.Random(seed + 1)
        self._episodes = self._generate_episodes()

        _patch_info = f"  n_patches={self.n_patches}" if self.n_patches else "  [GAP]"
        print(
            f"[UCF101FewShotDataset] split={split}  "
            f"classes={len(self.episode_class_ids)}  "
            f"embed_dim={self.embed_dim}  max_len={self.max_len}"
            f"{_patch_info}  (T_median={int(np.median(t_lengths))})  episodes={num_episodes}"
        )

    # ------------------------------------------------------------------
    def _generate_episodes(self):
        episodes = []
        for _ in range(self.num_episodes):
            classes = self._rng.sample(self.episode_class_ids, self.n_way)
            support_videos, query_videos = [], []
            for c in classes:
                vids = self._rng.sample(
                    self.class_to_videos[c], self.k_shot + self.n_query
                )
                support_videos.append(vids[: self.k_shot])
                query_videos.append(vids[self.k_shot:])
            episodes.append((classes, support_videos, query_videos))
        return episodes

    def _load_feat(self, video_name: str) -> torch.Tensor:
        feat = self._feat_cache[video_name].float()
        if self._augment:
            if self._noise_std > 0:
                feat = feat + torch.randn_like(feat) * feat.std() * self._noise_std
            if self._frame_drop > 0 and feat.shape[0] > 1:
                keep = torch.bernoulli(torch.full((feat.shape[0],), 1 - self._frame_drop))
                if keep.sum() > 0:
                    expand = keep.view(-1, *([1] * (feat.ndim - 1)))
                    feat = feat * expand
        return feat

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
        classes, support_videos, query_videos = self._episodes[idx]
        N = self.n_way
        K = self.k_shot
        Q = self.n_query
        T = self.max_len
        D = self.embed_dim
        P = self.n_patches

        if P is not None:
            support_features = torch.zeros(N, K, T, P, D)
            query_features   = torch.zeros(N * Q, T, P, D)
        else:
            support_features = torch.zeros(N, K, T, D)
            query_features   = torch.zeros(N * Q, T, D)
        support_masks = torch.zeros(N, K, T)
        query_masks   = torch.zeros(N * Q, T)
        query_labels  = torch.zeros(N * Q, dtype=torch.long)

        for n, cls_vids in enumerate(support_videos):
            for k, vname in enumerate(cls_vids):
                support_features[n, k], support_masks[n, k] = \
                    self._pad_or_truncate(self._load_feat(vname))

        qi = 0
        for label_idx, cls_vids in enumerate(query_videos):
            for vname in cls_vids:
                query_features[qi], query_masks[qi] = \
                    self._pad_or_truncate(self._load_feat(vname))
                query_labels[qi] = label_idx
                qi += 1

        return support_features, support_masks, query_features, query_masks, query_labels
