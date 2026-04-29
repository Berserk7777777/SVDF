"""
datasets/hmdb51.py – HMDB51 dataset classes for SVD feature extraction and FSAR training.

HMDB51 directory layout expected:
    hmdb51/
    └── videos/
        ├── brush_hair/
        │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi
        │   └── ...
        ├── catch/
        └── ...  (51 action classes total)

Video naming convention for feature files:
    {class_name}__{video_stem}_mean.npy
    (double underscore separator guarantees global uniqueness across 51 classes)

Few-shot class split (standard ARN splits):
    31 base / 10 val / 10 test  (total = 51)
    Read from splits/hmdb_ARN/{train,val,test}list03.txt
"""

import glob
import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


# =============================================================================
# HMDB51Dataset  –  for feature extraction via generate_svd_maps.py
# =============================================================================

class HMDB51Dataset:
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
        data_root: str = "../hmdb51/videos",
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
        print(f"[HMDB51Dataset] {len(self.meta)} videos found under {data_root}")

    def __len__(self):
        return len(self.meta)

    def get_meta(self, idx):
        return self.meta[idx]

    def __getitem__(self, idx):
        video, _, _ = torchvision.io.read_video(self.meta[idx]["path"])
        return video


# =============================================================================
# HMDB51FewShotDataset  –  for FSAR training via train_fsar.py
# =============================================================================

class HMDB51FewShotDataset(Dataset):
    """
    Few-shot episode dataset for HMDB51 FSAR.

    Class split: read from standard ARN splits in splits_root/
        trainlist03.txt → 31 base classes
        vallist03.txt   → 10 val classes
        testlist03.txt  → 10 test classes

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
        features_root: str = "results/hmdb51/extract_hmdb51/feats/combined",
        data_root: str = "../hmdb51/videos",        # kept for API compat, no longer used for split
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        num_episodes: int = 600,
        max_len: int = 50,
        use_mean: bool = True,
        n_base_classes: int = 31,                   # ignored (splits file is authoritative)
        n_val_classes: int = 10,                    # ignored (splits file is authoritative)
        seed: int = 42,
        splits_root: str = None,                    # path to splits/hmdb_ARN/; None = auto-detect
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
            splits_root = os.path.normpath(os.path.join(_here, "../../splits/hmdb_ARN"))

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
                f"  Expected splits/hmdb_ARN/ under project root."
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

        # episode_class_ids = sorted class name strings with enough samples
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
        print(f"  [HMDB51FewShotDataset] Caching {len(all_videos)} {self._feat_format} features …",
              end=" ", flush=True)
        self._feat_cache: dict[str, torch.Tensor] = {}
        for vname in all_videos:
            fpath = os.path.join(features_root, f"{vname}{_feat_suffix}")
            if self._feat_format == "grid":
                data = torch.load(fpath, map_location="cpu")
                feat = data["feats"].half()          # (T, P, D)
            else:
                arr  = np.load(fpath)
                feat = torch.from_numpy(arr).half()  # (T, D)
                if feat.ndim == 1:
                    feat = feat.unsqueeze(0)
            self._feat_cache[vname] = feat
        print("done.")

        # ── 推断维度 ──────────────────────────────────────────────────────
        sample_feat = next(iter(self._feat_cache.values()))
        if self._feat_format == "grid":
            # (T, P, D)
            self.n_patches = sample_feat.shape[1]   # P
            self.embed_dim = sample_feat.shape[2]   # D
        else:
            self.n_patches = None
            self.embed_dim = sample_feat.shape[-1]  # D

        t_lengths = [f.shape[0] for f in self._feat_cache.values()]
        auto_max_len = max(int(np.percentile(t_lengths, 95)), 1)
        self.max_len = max_len if max_len > 0 else auto_max_len

        self._rng = random.Random(seed + 1)
        self._episodes = self._generate_episodes()

        _patch_info = f"  n_patches={self.n_patches}" if self.n_patches else "  [GAP]"
        print(
            f"[HMDB51FewShotDataset] split={split}  "
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
                    # unsqueeze to broadcast over spatial/channel dims
                    expand = keep.view(-1, *([1] * (feat.ndim - 1)))
                    feat = feat * expand
        return feat

    def _pad_or_truncate(self, feat: torch.Tensor):
        """Works for both (T, D) and (T, P, D) tensors."""
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
        P = self.n_patches  # None for GAP

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


# =============================================================================
# HMDB51PhasedFewShotDataset  –  三段式相位特征版本
# =============================================================================

class HMDB51PhasedFewShotDataset(Dataset):
    """
    基于三段式相位特征（begin / middle / end）的 Few-Shot 数据集。

    与 HMDB51FewShotDataset 的区别：
    - 特征文件：每视频 3 个 .npy 文件（_begin_mean / _middle_mean / _end_mean）
    - 每个文件形状 (D,)，三段拼接后 → (3*D,)
    - 返回 T=1 的伪时序格式，与现有训练框架完全兼容
      support_features : (N, K, 1, 3*D)
      query_features   : (N*Q, 1, 3*D)
    - n_patches 为 None（1D 特征路径）
    - embed_dim = 3*D（拼接维度）

    特征文件位置（由 extract_phase_features.py 生成）：
        {features_root}/{video_name}_begin_mean.npy
        {features_root}/{video_name}_middle_mean.npy
        {features_root}/{video_name}_end_mean.npy
    """

    def __init__(
        self,
        split: str = "base",
        features_root: str = "results/hmdb51/extract_hmdb51_phase3/feats",
        data_root: str = "../hmdb51/videos",
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        num_episodes: int = 600,
        n_base_classes: int = 31,
        n_val_classes: int = 10,
        seed: int = 42,
        video_extensions=(".avi", ".mp4"),
        # 以下参数保留 API 兼容性，实际无效（相位特征已无时序维度）
        max_len: int = 1,
        use_mean: bool = True,
        feat_noise_std: float = 0.0,
        frame_drop_prob: float = 0.0,
    ):
        assert split in ("base", "val", "test")

        self.features_root = features_root
        self.n_way         = n_way
        self.k_shot        = k_shot
        self.n_query       = n_query
        self.num_episodes  = num_episodes
        self.split         = split
        self.max_len       = 1       # 每视频 1 个相位拼接向量 (T=1)
        self.n_patches     = None    # 标识为 1D 特征，使用 1D 模型路径

        # ── 类别发现 ──────────────────────────────────────────────────────
        class_names = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ])
        num_total = len(class_names)

        n_test = num_total - n_base_classes - n_val_classes
        assert n_test > 0, (
            f"n_base({n_base_classes}) + n_val({n_val_classes}) must be < total({num_total})"
        )

        rng      = random.Random(seed)
        shuffled = list(range(num_total))
        rng.shuffle(shuffled)
        split_map = {
            "base": set(shuffled[:n_base_classes]),
            "val":  set(shuffled[n_base_classes: n_base_classes + n_val_classes]),
            "test": set(shuffled[n_base_classes + n_val_classes:]),
        }
        self.episode_class_ids = sorted(split_map[split])

        # ── 构建 class → [video_name, ...] 映射（仅保留三段文件齐全的视频）──
        phase_suffixes = ["_begin_mean.npy", "_middle_mean.npy", "_end_mean.npy"]
        class_to_videos: dict = {c: [] for c in self.episode_class_ids}

        for class_id in self.episode_class_ids:
            class_name = class_names[class_id]
            class_dir  = os.path.join(data_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for ext in video_extensions:
                for vpath in sorted(glob.glob(os.path.join(class_dir, f"*{ext}"))):
                    stem       = os.path.splitext(os.path.basename(vpath))[0]
                    video_name = f"{class_name}__{stem}"
                    # 三段文件必须全部存在
                    if all(
                        os.path.exists(os.path.join(features_root, f"{video_name}{suf}"))
                        for suf in phase_suffixes
                    ):
                        class_to_videos[class_id].append(video_name)

        # 过滤样本不足的类
        self.episode_class_ids = [
            c for c in self.episode_class_ids
            if len(class_to_videos[c]) >= k_shot + n_query
        ]
        self.class_to_videos = {c: class_to_videos[c] for c in self.episode_class_ids}

        assert len(self.episode_class_ids) >= n_way, (
            f"Only {len(self.episode_class_ids)} classes have enough phased samples "
            f"(need n_way={n_way}). Run extract_phase_features.py first."
        )

        # ── RAM 缓存（float16，减半内存）───────────────────────────────────
        all_videos = list({v for vids in self.class_to_videos.values() for v in vids})
        print(f"  [HMDB51Phased] Caching {len(all_videos)} videos (3 phases each) …",
              end=" ", flush=True)
        self._feat_cache: dict[str, torch.Tensor] = {}
        for vname in all_videos:
            parts = []
            for suf in phase_suffixes:
                arr = np.load(os.path.join(features_root, f"{vname}{suf}"))
                parts.append(torch.from_numpy(arr).float())
            concat = torch.cat(parts, dim=0)   # (3*D,)
            self._feat_cache[vname] = concat.half()
        print("done.")

        # ── 推断特征维度 ──────────────────────────────────────────────────
        self.embed_dim = next(iter(self._feat_cache.values())).shape[0]  # 3*D

        # ── 预生成 episodes ───────────────────────────────────────────────
        self._rng      = random.Random(seed + 1)
        self._episodes = self._generate_episodes()

        print(
            f"[HMDB51PhasedFewShotDataset] split={split}  "
            f"classes={len(self.episode_class_ids)}  "
            f"embed_dim={self.embed_dim} (= 3×{self.embed_dim//3})  "
            f"episodes={num_episodes}"
        )

    def _generate_episodes(self):
        episodes = []
        for _ in range(self.num_episodes):
            classes = self._rng.sample(self.episode_class_ids, self.n_way)
            support_videos, query_videos = [], []
            for c in classes:
                vids = self._rng.sample(self.class_to_videos[c], self.k_shot + self.n_query)
                support_videos.append(vids[: self.k_shot])
                query_videos.append(vids[self.k_shot:])
            episodes.append((classes, support_videos, query_videos))
        return episodes

    def _load_feat(self, video_name: str) -> torch.Tensor:
        return self._feat_cache[video_name].float()   # fp16 → fp32

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        """
        Returns:
            support_features : (N, K, 1, 3*D)   T=1 伪时序
            support_masks    : (N, K, 1)          全 1
            query_features   : (N*Q, 1, 3*D)
            query_masks      : (N*Q, 1)
            query_labels     : (N*Q,)
        """
        classes, support_videos, query_videos = self._episodes[idx]
        N, K, Q = self.n_way, self.k_shot, self.n_query
        D = self.embed_dim   # 3*D_orig

        support_features = torch.zeros(N, K, 1, D)
        support_masks    = torch.ones(N, K, 1)
        query_features   = torch.zeros(N * Q, 1, D)
        query_masks      = torch.ones(N * Q, 1)
        query_labels     = torch.zeros(N * Q, dtype=torch.long)

        for n, cls_vids in enumerate(support_videos):
            for k, vname in enumerate(cls_vids):
                support_features[n, k, 0] = self._load_feat(vname)

        qi = 0
        for label_idx, cls_vids in enumerate(query_videos):
            for vname in cls_vids:
                query_features[qi, 0] = self._load_feat(vname)
                query_labels[qi]      = label_idx
                qi += 1

        return support_features, support_masks, query_features, query_masks, query_labels
