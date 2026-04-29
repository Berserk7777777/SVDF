import os
import glob
import json
import random
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SSV2Dataset(Dataset):
    def __init__(
        self,
        split=None,
        data_root="ssv2/videos",
        video_extensions=(".webm",),
    ):
        self.data_root = data_root
        self.split = split
        self.video_extensions = video_extensions

        # Placeholder (not used), kept consistent with ucf.py / hmdb.py
        self.actions = []

        # SSV2 videos are typically stored flat under one folder.
        # We keep split for interface consistency, but do not use it
        # to change the search directory here.
        search_dir = self.data_root

        self.video_paths = []
        for ext in self.video_extensions:
            self.video_paths.extend(
                glob.glob(os.path.join(search_dir, f"*{ext}"))
            )

        self.video_paths.sort()

        self.video_names = [
            os.path.splitext(os.path.basename(path))[0]
            for path in self.video_paths
        ]

        self.meta = [
            {"name": video_name, "path": video_path}
            for video_name, video_path in zip(self.video_names, self.video_paths)
        ]

    def __len__(self):
        return len(self.meta)

    def get_meta(self, idx):
        return self.meta[idx]

    def __getitem__(self, idx):
        # This method is not actually used in the feature extraction script
        # but is required for the Dataset interface
        video_path = self.video_paths[idx]
        video, _, _ = torchvision.io.read_video(video_path)
        return video


class SSV2FeatureDataset(Dataset):
    def __init__(
        self,
        split=None,
        features_root="data/ssv2",
        data_root="ssv2/labels",
        max_len=300,
        num_frames=16,
        use_mean=False,
    ):
        self.features_root = features_root
        self.data_root = data_root
        self.max_len = max_len
        self.num_frames = num_frames
        self.split = split

        split_map = {
            "train": "train.json",
            "val": "validation.json",
            "validation": "validation.json",
            "test": "test.json",
        }

        if self.split is not None:
            if self.split not in split_map:
                raise ValueError(f"Unsupported split: {self.split}")
            annotation_path = os.path.join(self.data_root, split_map[self.split])
        else:
            annotation_path = os.path.join(self.data_root, "train.json")

        labels_path = os.path.join(self.data_root, "labels.json")

        with open(labels_path, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)

        if isinstance(raw_labels, dict):
            # Official SSV2 labels.json is commonly {"action template": "index"}
            self.label_map = {k: int(v) for k, v in raw_labels.items()}
            self.all_actions = [
                k for k, _ in sorted(self.label_map.items(), key=lambda x: x[1])
            ]
        elif isinstance(raw_labels, list):
            self.all_actions = list(raw_labels)
            self.label_map = {name: i for i, name in enumerate(self.all_actions)}
        else:
            raise ValueError("Unsupported labels.json format")

        self.num_classes = len(self.label_map)

        with open(annotation_path, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)

        self.meta_data = []
        for row in raw_meta:
            video_name = str(row["id"])

            if "template" not in row and "label" not in row:
                # test split: no annotation
                label_idx = -1
            else:
                if "template" in row:
                    # Strip placeholder brackets: "Holding [something]" -> "Holding something"
                    action_name = row["template"].replace("[", "").replace("]", "")
                else:
                    action_name = row["label"]

                if action_name not in self.label_map:
                    raise KeyError(f"Action name not found in labels.json: {action_name!r}")

                label_idx = self.label_map[action_name]

            self.meta_data.append((video_name, label_idx))

        self.video_names = [meta[0] for meta in self.meta_data]

        if use_mean:
            self.feature_paths = [
                os.path.join(self.features_root, f"{video_name}_mean.npy")
                for video_name in self.video_names
            ]
        else:
            self.feature_paths = [
                os.path.join(self.features_root, f"{video_name}.npy")
                for video_name in self.video_names
            ]

        self.labels = [int(meta[1]) for meta in self.meta_data]

        if len(self.feature_paths) > 0 and os.path.exists(self.feature_paths[0]):
            arr = np.load(self.feature_paths[0])
            self.embed_dim = arr.shape[1] if arr.ndim > 1 else arr.shape[0]
        else:
            self.embed_dim = 1280
            print("Warning: No feature files found to determine embedding dimension")

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        label_idx = self.labels[idx]

        feature = torch.from_numpy(np.load(feature_path)).float()

        if feature.ndim == 1:
            feature = feature.unsqueeze(0)

        label_tensor = torch.zeros(self.num_classes)
        if label_idx >= 0:
            label_tensor[label_idx] = 1.0

        T = feature.shape[0]
        if T > self.max_len:
            feature = feature[:self.max_len]
        elif T < self.max_len:
            pad_shape = (self.max_len - T,) + tuple(feature.shape[1:])
            feature = torch.cat([feature, torch.zeros(pad_shape, dtype=feature.dtype)], dim=0)

        mask = torch.ones(self.max_len)
        if T < self.max_len:
            mask[T:] = 0

        return torch.zeros(1), feature, mask, label_tensor

    def calculate_map(self, predictions, targets):
        pred_classes = torch.argmax(predictions, dim=1)
        target_classes = torch.argmax(targets, dim=1)

        correct = (pred_classes == target_classes).sum().item()
        total = targets.size(0)

        return correct / total


class SSV2FewShotDataset(Dataset):
    """
    Few-shot action recognition episode dataset for SSV2.

    Each item is one N-way K-shot episode:
        support_features : (N, K, max_len, D)
        support_masks    : (N, K, max_len)
        query_features   : (N*n_query, max_len, D)
        query_masks      : (N*n_query, max_len)
        query_labels     : (N*n_query,)  int in [0, N)

    split='base'  -> train on base classes   (64 classes in OTAM split)
    split='val'   -> evaluate on val novel   (12 classes in OTAM split)
    split='test'  -> evaluate on test novel  (24 classes in OTAM split)

    Class splits are read from standard OTAM splits in splits_root/
        trainlist07.txt → 64 base classes
        vallist07.txt   → 12 val classes
        testlist07.txt  → 24 test classes
    Each line: "class_folder/video_id" (e.g. "train8/78687")
    """

    def __init__(
        self,
        split: str = "base",
        features_root: str = "results/ssv2/extract_ssv2_svdxt_t20/feats/layer_1",
        data_root: str = "ssv2/labels",             # kept for API compat, no longer used for split
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        num_episodes: int = 600,
        max_len: int = 50,
        use_mean: bool = True,
        n_base_classes: int = 100,                  # ignored (splits file is authoritative)
        n_val_classes: int = 37,                    # ignored (splits file is authoritative)
        seed: int = 42,
        splits_root: str = None,                    # path to splits/ssv2_OTAM/; None = auto-detect
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

        # ── Resolve splits_root ────────────────────────────────────────────
        if splits_root is None:
            _here = os.path.dirname(os.path.abspath(__file__))
            splits_root = os.path.normpath(os.path.join(_here, "../../splits/ssv2_small"))

        # ── Load class/video list from splits file ─────────────────────────
        # Supports two naming conventions:
        #   new: {split}_few_shot.txt  (ssv2_small / ssv2_full)
        #   old: {split}list07.txt     (ssv2_OTAM, legacy)
        _split_file_map_new = {
            "base": "train_few_shot.txt",
            "val":  "val_few_shot.txt",
            "test": "test_few_shot.txt",
        }
        _split_file_map_old = {
            "base": "trainlist07.txt",
            "val":  "vallist07.txt",
            "test": "testlist07.txt",
        }
        split_file = os.path.join(splits_root, _split_file_map_new[split])
        if not os.path.exists(split_file):
            split_file = os.path.join(splits_root, _split_file_map_old[split])
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Splits file not found in {splits_root}\n"
                f"  Expected train_few_shot.txt / val_few_shot.txt / test_few_shot.txt\n"
                f"  or trainlist07.txt / vallist07.txt / testlist07.txt"
            )

        suffix = "_mean.npy" if use_mean else ".npy"
        with open(split_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        # ── Build class_folder → [video_id, ...] from split file ──────────
        # Each line: "class_folder/video_id" (e.g. "train8/78687")
        # Feature file: {features_root}/{video_id}{suffix}
        class_to_videos: dict = {}
        for line in lines:
            parts = line.split("/", 1)
            if len(parts) != 2:
                continue
            class_folder, video_id = parts
            feat_path = os.path.join(features_root, f"{video_id}{suffix}")
            if os.path.exists(feat_path):
                if class_folder not in class_to_videos:
                    class_to_videos[class_folder] = []
                class_to_videos[class_folder].append(video_id)

        # episode_class_ids = sorted class folder strings with enough samples
        self.episode_class_ids = sorted([
            cls for cls, vids in class_to_videos.items()
            if len(vids) >= k_shot + n_query
        ])
        self.class_to_videos = {cls: class_to_videos[cls] for cls in self.episode_class_ids}

        assert len(self.episode_class_ids) >= n_way, (
            f"Only {len(self.episode_class_ids)} classes have enough samples "
            f"(need {n_way}). Lower k_shot/n_query or check features_root."
        )

        # determine feature dim from first available feature
        sample_video = self.class_to_videos[self.episode_class_ids[0]][0]
        arr = np.load(os.path.join(features_root, f"{sample_video}{suffix}"))
        self.embed_dim = arr.shape[1] if arr.ndim > 1 else arr.shape[0]

        # pre-generate episode class/sample indices for reproducibility
        self._rng = random.Random(seed + 1)
        self._episodes = self._generate_episodes()

    # ------------------------------------------------------------------
    def _generate_episodes(self):
        episodes = []
        for _ in range(self.num_episodes):
            classes = self._rng.sample(self.episode_class_ids, self.n_way)
            support_videos = []
            query_videos = []
            for c in classes:
                videos = self._rng.sample(
                    self.class_to_videos[c], self.k_shot + self.n_query
                )
                support_videos.append(videos[: self.k_shot])
                query_videos.append(videos[self.k_shot:])
            episodes.append((classes, support_videos, query_videos))
        return episodes

    # ------------------------------------------------------------------
    def _load_feat(self, video_name: str) -> torch.Tensor:
        suffix = "_mean.npy" if self.use_mean else ".npy"
        arr = np.load(os.path.join(self.features_root, f"{video_name}{suffix}"))
        feat = torch.from_numpy(arr).float()
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)
        return feat

    def _pad_or_truncate(self, feat: torch.Tensor):
        T = feat.shape[0]
        if T >= self.max_len:
            feat = feat[: self.max_len]
            mask = torch.ones(self.max_len)
        else:
            pad = torch.zeros(self.max_len - T, feat.shape[1])
            feat = torch.cat([feat, pad], dim=0)
            mask = torch.zeros(self.max_len)
            mask[:T] = 1.0
        return feat, mask

    # ------------------------------------------------------------------
    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        classes, support_videos, query_videos = self._episodes[idx]

        support_feats_list = []
        support_masks_list = []
        for cls_vids in support_videos:
            cls_feats, cls_masks = [], []
            for vname in cls_vids:
                f, m = self._pad_or_truncate(self._load_feat(vname))
                cls_feats.append(f)
                cls_masks.append(m)
            support_feats_list.append(torch.stack(cls_feats))   # (K, T, D)
            support_masks_list.append(torch.stack(cls_masks))   # (K, T)

        query_feats_list = []
        query_masks_list = []
        query_labels = []
        for label_idx, cls_vids in enumerate(query_videos):
            for vname in cls_vids:
                f, m = self._pad_or_truncate(self._load_feat(vname))
                query_feats_list.append(f)
                query_masks_list.append(m)
                query_labels.append(label_idx)

        support_features = torch.stack(support_feats_list)   # (N, K, T, D)
        support_masks = torch.stack(support_masks_list)      # (N, K, T)
        query_features = torch.stack(query_feats_list)       # (N*Q, T, D)
        query_masks = torch.stack(query_masks_list)          # (N*Q, T)
        query_labels = torch.tensor(query_labels, dtype=torch.long)  # (N*Q,)

        return support_features, support_masks, query_features, query_masks, query_labels


# =============================================================================
# SSV2EpisodicVideoDataset
# =============================================================================

class SSV2EpisodicVideoDataset(Dataset):
    """
    SSv2-Small 元学习情节采样数据集（原始视频版）。

    每次 __getitem__ 返回一个完整的 N-way K-shot 情节 (episode)，包含：
        support_x : (N*K, T, C, H, W)  支持集视频帧张量，值域 [-1, 1]
        support_y : (N*K,)              支持集局部标签 [0, N-1]
        query_x   : (N*Q, T, C, H, W)  查询集视频帧张量，值域 [-1, 1]
        query_y   : (N*Q,)              查询集局部标签 [0, N-1]

    类别划分（SSv2-Small 共 100 个类，按固定 seed 随机划分）：
        meta_train : 64 个类
        meta_val   : 12 个类
        meta_test  : 24 个类

    效率提示：SVD 特征提取极慢。生产环境强烈建议先用 dataset=ssv2 离线提取
    所有视频特征（.npy），再在特征上使用 SSV2FewShotDataset 做 Episodic 采样，
    效率可提升数个数量级。本类适用于在线调试或小规模实验。
    """

    # SSv2-Small 三路类别划分数量
    N_META_TRAIN = 64
    N_META_VAL   = 12
    N_META_TEST  = 24

    def __init__(
        self,
        split: str = "meta_train",               # 'meta_train' | 'meta_val' | 'meta_test'
        video_root: str = "../../ssv2/videos",   # 相对于脚本执行目录的视频路径
        label_root: str = "../../ssv2/labels",   # 相对于脚本执行目录的标签路径
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        n_frames: int = 8,           # 每个视频均匀采样的帧数
        num_episodes: int = 600,
        img_size: int = 320,         # Resize 短边（与 SVD preprocess 对齐）
        crop_size: int = 256,        # CenterCrop 尺寸（需可被 8 整除）
        seed: int = 42,
        video_extensions=(".webm", ".mp4"),
    ):
        assert split in ("meta_train", "meta_val", "meta_test"), (
            f"split 必须为 meta_train / meta_val / meta_test，当前: {split}"
        )

        # 将相对路径锚定到本文件（datasets/ssv2.py）所在目录，
        # 使路径解析与 CWD 无关（兼容从任意目录调用的情形）
        _here = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(video_root):
            video_root = os.path.normpath(os.path.join(_here, video_root))
        if not os.path.isabs(label_root):
            label_root = os.path.normpath(os.path.join(_here, label_root))

        self.video_root   = video_root
        self.label_root   = label_root
        self.n_way        = n_way
        self.k_shot       = k_shot
        self.n_query      = n_query
        self.n_frames     = n_frames
        self.num_episodes = num_episodes
        self.crop_size    = crop_size

        # ── 预处理变换（适配 SVD 输入）──
        # SVD preprocess() 等价：uint8 → [0,1] → [-1,1]
        # 这里对 PIL Image 做等价操作（ToTensor→[0,1]，Normalize→[-1,1]）
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),                             # (C,H,W), [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5],        # → [-1,1]
                                 std=[0.5, 0.5, 0.5]),
        ])

        # ── 读取标签文件，构建全局 label_map ──
        labels_path = os.path.join(label_root, "labels.json")
        with open(labels_path, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)

        if isinstance(raw_labels, dict):
            label_map = {k: int(v) for k, v in raw_labels.items()}
        else:
            label_map = {name: i for i, name in enumerate(raw_labels)}

        all_class_ids = list(range(len(label_map)))

        # ── 用固定 seed 对类别进行随机三路划分 ──
        rng = random.Random(seed)
        shuffled = all_class_ids[:]
        rng.shuffle(shuffled)

        n_tr = self.N_META_TRAIN
        n_val = self.N_META_VAL
        n_test = self.N_META_TEST

        base_ids = set(shuffled[:n_tr])
        val_ids = set(shuffled[n_tr: n_tr + n_val])
        test_ids = set(shuffled[n_tr + n_val: n_tr + n_val + n_test])

        split_id_map = {
            "meta_train": base_ids,
            "meta_val":   val_ids,
            "meta_test":  test_ids,
        }
        episode_class_ids = sorted(split_id_map[split])

        # ── 读取标注文件，构建 class_id → [video_path, ...] 映射 ──
        class_to_videos: dict = {c: [] for c in episode_class_ids}

        for ann_file in ("train.json", "validation.json"):
            ann_path = os.path.join(label_root, ann_file)
            if not os.path.exists(ann_path):
                continue
            with open(ann_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            for row in rows:
                if "template" not in row and "label" not in row:
                    continue
                video_name = str(row["id"])
                action_name = (
                    row["template"].replace("[", "").replace("]", "")
                    if "template" in row else row["label"]
                )
                if action_name not in label_map:
                    continue
                label_idx = label_map[action_name]
                if label_idx not in class_to_videos:
                    continue
                # 查找视频文件（支持多种扩展名）
                for ext in video_extensions:
                    candidate = os.path.join(video_root, f"{video_name}{ext}")
                    if os.path.exists(candidate):
                        class_to_videos[label_idx].append(candidate)
                        break  # 找到即停止

        # 过滤样本数不足的类别（至少需要 k_shot + n_query 个视频）
        self.episode_class_ids = [
            c for c in episode_class_ids
            if len(class_to_videos[c]) >= k_shot + n_query
        ]
        self.class_to_videos = {c: class_to_videos[c] for c in self.episode_class_ids}

        assert len(self.episode_class_ids) >= n_way, (
            f"可用类别数 {len(self.episode_class_ids)} < n_way={n_way}，"
            "请确认视频文件路径正确，或降低 k_shot/n_query"
        )

        print(
            f"[SSV2EpisodicVideoDataset] split={split}，"
            f"可用类别={len(self.episode_class_ids)}，"
            f"n_way={n_way}，k_shot={k_shot}，n_query={n_query}，"
            f"n_frames={n_frames}，num_episodes={num_episodes}"
        )

        # ── 预生成所有 episode 的采样计划（保证可复现）──
        self._rng = random.Random(seed + 1)
        self._episodes = self._generate_episodes()

    # ------------------------------------------------------------------
    def _generate_episodes(self):
        """预先生成 num_episodes 个情节的类别与视频采样索引。"""
        episodes = []
        for _ in range(self.num_episodes):
            classes = self._rng.sample(self.episode_class_ids, self.n_way)
            support_vids, query_vids = [], []
            for c in classes:
                vids = self._rng.sample(
                    self.class_to_videos[c], self.k_shot + self.n_query
                )
                support_vids.append(vids[: self.k_shot])
                query_vids.append(vids[self.k_shot :])
            episodes.append((classes, support_vids, query_vids))
        return episodes

    # ------------------------------------------------------------------
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """
        读取视频并均匀采样 n_frames 帧，不足时循环重复补帧。

        Returns:
            Tensor (T, C, H, W)，值域 [-1, 1]，dtype=float32
        """
        try:
            raw, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
            # raw: (T, H, W, C), uint8
        except Exception as e:
            print(f"[警告] 读取视频失败 {video_path}: {e}，返回零帧")
            return torch.zeros(self.n_frames, 3, self.crop_size, self.crop_size)

        total = raw.shape[0]
        if total == 0:
            print(f"[警告] 空视频 {video_path}，返回零帧")
            return torch.zeros(self.n_frames, 3, self.crop_size, self.crop_size)

        # 均匀采样；帧数不足时循环重复补帧
        if total >= self.n_frames:
            indices = torch.linspace(0, total - 1, self.n_frames).long()
        else:
            # 重复补帧策略：循环列表直到凑够 n_frames 帧
            base = list(range(total))
            repeated = (base * ((self.n_frames // total) + 1))[: self.n_frames]
            indices = torch.tensor(repeated)

        sampled = raw[indices]  # (n_frames, H, W, C), uint8

        # 逐帧应用 Resize → CenterCrop → ToTensor → Normalize
        processed = []
        for i in range(self.n_frames):
            frame_np = sampled[i].numpy()          # (H, W, C), uint8
            pil_img  = Image.fromarray(frame_np)   # PIL RGB Image
            processed.append(self.transform(pil_img))  # (C, H, W), float32 [-1,1]

        return torch.stack(processed)  # (T, C, H, W)

    # ------------------------------------------------------------------
    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        """
        返回一个 N-way K-shot 情节。

        局部标签映射（Label Remapping）：
            每个 episode 内，全局类别 ID（如 0~99）通过 target_mapping
            映射为局部标签（0~N-1），使分类头的 Cross-Entropy Loss 维度对齐。

        Returns:
            support_x : Tensor (N*K, T, C, H, W)  支持集帧序列
            support_y : Tensor (N*K,)              支持集局部标签 [0, N-1]
            query_x   : Tensor (N*Q, T, C, H, W)  查询集帧序列
            query_y   : Tensor (N*Q,)              查询集局部标签 [0, N-1]
        """
        classes, support_vids, query_vids = self._episodes[idx]

        # ── 局部标签映射（关键步骤，缺失会导致 Loss 维度错乱）──
        # global_class_id → local_label (0 到 N-1)
        target_mapping = {
            global_cls: local_label
            for local_label, global_cls in enumerate(classes)
        }

        support_x_list, support_y_list = [], []
        query_x_list,   query_y_list   = [], []

        # 加载支持集（N * K 个视频）
        for global_cls, cls_vids in zip(classes, support_vids):
            local_label = target_mapping[global_cls]
            for vid_path in cls_vids:
                frames = self._load_video_frames(vid_path)  # (T, C, H, W)
                support_x_list.append(frames)
                support_y_list.append(local_label)

        # 加载查询集（N * Q 个视频）
        for global_cls, cls_vids in zip(classes, query_vids):
            local_label = target_mapping[global_cls]
            for vid_path in cls_vids:
                frames = self._load_video_frames(vid_path)  # (T, C, H, W)
                query_x_list.append(frames)
                query_y_list.append(local_label)

        support_x = torch.stack(support_x_list)                      # (N*K, T, C, H, W)
        support_y = torch.tensor(support_y_list, dtype=torch.long)   # (N*K,)
        query_x   = torch.stack(query_x_list)                        # (N*Q, T, C, H, W)
        query_y   = torch.tensor(query_y_list,   dtype=torch.long)   # (N*Q,)

        return support_x, support_y, query_x, query_y