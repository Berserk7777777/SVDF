"""
train_fsar.py – Episode-based meta-learning training for SSV2 FSAR.

Usage example (run from generative-models/):
    python scripts/train_fsar.py \
        --features_root results/ssv2/extract_ssv2_svdxt_t20/feats/layer_1 \
        --data_root ../ssv2/labels \
        --n_way 5 --k_shot 1 --n_query 15 \
        --hidden_dim 512 \
        --num_train_episodes 6000 \
        --num_val_episodes 600 \
        --num_epochs 50 \
        --lr 1e-4 \
        --save_dir results/fsar_ssv2
"""

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from datasets.ssv2 import SSV2FewShotDataset
from datasets.hmdb51 import HMDB51FewShotDataset
from datasets.ucf101 import UCF101FewShotDataset
from datasets.kinetics400 import Kinetics400FewShotDataset
from models.FSARModel import PhaseAwareProtoNet, SpatioTemporalTRXNet


# ---------------------------------------------------------------------------
# Dry-Run Dataset – reads pre-saved episode_XXXX.pt files directly
# ---------------------------------------------------------------------------

class DryRunEpisodeDataset(Dataset):
    """
    Reads pre-saved episode .pt files produced by the feature-extraction stage.

    Each file must contain:
        support_feats : (N, T, D) or (N, T, P, D)  – one support sample per class (K=1)
        support_y     : (N,)                         – class indices in [0, N)
        query_feats   : (N*Q, T, D) or (N*Q, T, P, D)
        query_y       : (N*Q,)                       – class indices in [0, N)

    Returns a tuple compatible with episode_collate / run_episode_batch:
        sup_f  : (N, 1, T, D) or (N, 1, T, P, D)  float
        sup_m  : (N, 1, T)                          all-ones mask
        qry_f  : (N*Q, T, D)  or (N*Q, T, P, D)   float
        qry_m  : (N*Q, T)                           all-ones mask
        qry_l  : (N*Q,)                             long

    augment=True 时施加三层数据增强，防止模型记住固定 episode：
        1. 高斯噪声（相对特征标准差缩放）
        2. 随机时序帧掩码（独立按帧伯努利采样）
        3. 随机空间 patch 掩码（仅 4D 特征）
    """

    def __init__(
        self,
        feats_dir: str,
        indices=None,
        augment: bool = False,
        noise_std: float = 0.02,
        frame_drop_prob: float = 0.1,
        patch_drop_prob: float = 0.1,
    ):
        all_files = sorted(glob.glob(os.path.join(feats_dir, "episode_*.pt")))
        if not all_files:
            raise FileNotFoundError(f"No episode_*.pt files found in {feats_dir}")
        self.files = [all_files[i] for i in indices] if indices is not None else all_files
        self.augment        = augment
        self.noise_std      = noise_std
        self.frame_drop_prob = frame_drop_prob
        self.patch_drop_prob = patch_drop_prob

        # Infer meta-data from the first episode
        ep0 = torch.load(self.files[0], map_location="cpu")
        sf  = ep0["support_feats"]
        self.embed_dim = sf.shape[-1]            # D
        self.n_way     = sf.shape[0]             # N
        self.seq_len   = sf.shape[1]             # T
        # 4D → new spatial format (T, P, D); 3D → legacy (T, D)
        self.n_patches = sf.shape[2] if sf.ndim == 4 else None   # P or None

    def _augment_feats(self, x: torch.Tensor) -> torch.Tensor:
        """
        对特征张量施加轻量增强，防止过拟合到固定 episode。
        x: (B, T, D) 或 (B, T, P, D)
        """
        # 1. 高斯噪声，按特征整体标准差缩放，保持相对幅度
        if self.noise_std > 0:
            scale = x.std().clamp(min=1e-6)
            x = x + self.noise_std * scale * torch.randn_like(x)

        # 2. 随机时序帧掩码：每帧以 frame_drop_prob 概率置零
        T = x.shape[1]
        if self.frame_drop_prob > 0 and T > 1:
            mask = torch.bernoulli(torch.full((T,), 1.0 - self.frame_drop_prob))
            if mask.sum() == 0:          # 保证至少保留一帧
                mask[torch.randint(T, (1,)).item()] = 1.0
            # shape: (1, T, 1) 或 (1, T, 1, 1) 以便广播
            view_shape = [1, T] + [1] * (x.ndim - 2)
            x = x * mask.view(view_shape)

        # 3. 随机空间 patch 掩码（仅 4D 特征: B, T, P, D）
        if self.patch_drop_prob > 0 and x.ndim == 4:
            P = x.shape[2]
            mask = torch.bernoulli(torch.full((P,), 1.0 - self.patch_drop_prob))
            if mask.sum() == 0:          # 保证至少保留一个 patch
                mask[torch.randint(P, (1,)).item()] = 1.0
            x = x * mask.view(1, 1, P, 1)

        return x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ep    = torch.load(self.files[idx], map_location="cpu")
        sup_f = ep["support_feats"].float()   # (N, T, D) or (N, T, P, D)
        qry_f = ep["query_feats"].float()     # (N*Q, T, D) or (N*Q, T, P, D)
        qry_l = ep["query_y"].long()          # (N*Q,)

        if self.augment:
            sup_f = self._augment_feats(sup_f)
            qry_f = self._augment_feats(qry_f)

        N   = sup_f.shape[0]
        T   = sup_f.shape[1]
        NQ  = qry_f.shape[0]

        sup_f = sup_f.unsqueeze(1)            # (N, K=1, T, D) or (N, K=1, T, P, D)
        sup_m = torch.ones(N, 1, T)           # (N, K=1, T)  — mask is T-only
        qry_m = torch.ones(NQ, T)             # (N*Q, T)

        return sup_f, sup_m, qry_f, qry_m, qry_l


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def episode_collate(batch):
    """Stack a list of single episodes into a batched tensor set."""
    (sup_f, sup_m, qry_f, qry_m, qry_l) = zip(*batch)
    return (
        torch.stack(sup_f),   # (B, N, K, T, D) or (B, N, K, T, P, D)
        torch.stack(sup_m),   # (B, N, K, T)
        torch.stack(qry_f),   # (B, N*Q, T, D) or (B, N*Q, T, P, D)
        torch.stack(qry_m),   # (B, N*Q, T)
        torch.stack(qry_l),   # (B, N*Q)
    )


def run_episode_batch(model, sup_f, sup_m, qry_f, qry_m, qry_l, n_way, k_shot, device):
    """
    Process one batch of episodes.
    Returns logits (B*N*Q, N) and labels (B*N*Q,).

    Handles both feature formats:
        Legacy  5-D: sup_f (B, N, K, T, D),    qry_f (B, Q_total, T, D)
        Spatial 6-D: sup_f (B, N, K, T, P, D), qry_f (B, Q_total, T, P, D)

    GPU optimisation (SpatioTemporalTRXNet only):
        Batch ALL B*N*K support and B*Q_total query tokens into a SINGLE encoder
        call each, then loop over episodes for the cheap matching head.
        This reduces encoder invocations from 2*B (old per-episode loop) to 2,
        eliminating inter-episode GPU idle time and maximising CUDA occupancy.
    """
    Q_total = qry_f.shape[1]

    if sup_f.ndim == 6:
        B, N, K, T, P, D = sup_f.shape
        sup_f = sup_f.view(B * N * K, T, P, D).to(device)
        qry_f = qry_f.view(B * Q_total, T, P, D).to(device)
    else:
        B, N, K, T, D = sup_f.shape
        sup_f = sup_f.view(B * N * K, T, D).to(device)
        qry_f = qry_f.view(B * Q_total, T, D).to(device)

    sup_m = sup_m.view(B * N * K, T).to(device)
    qry_m = qry_m.view(B * Q_total, T).to(device)
    qry_l = qry_l.view(B * Q_total).to(device)

    all_logits = []
    all_labels = []

    if isinstance(model, SpatioTemporalTRXNet):
        # ── 批量编码：一次 encoder 前向传递所有 B 个 episode 的 support/query ──
        # 原版每轮 batch 调用 model() B 次 → 2B 次 encoder，大量 GPU idle。
        # 现在：2 次大 encoder call + B 次轻量 matching head（无参数或极少参数）。
        all_sup_enc = model.encoder(sup_f, sup_m)   # (B*N*K,  n_phases, P, H)
        all_qry_enc = model.encoder(qry_f, qry_m)   # (B*Q_total, n_phases, P, H)

        for ep in range(B):
            s_start = ep * N * K;    s_end = s_start + N * K
            q_start = ep * Q_total;  q_end = q_start + Q_total
            logits = model.trx(
                all_qry_enc[q_start:q_end],   # (Q_total, n_phases, P, H)
                all_sup_enc[s_start:s_end],   # (N*K,     n_phases, P, H)
                n_way=N, k_shot=K,
            )  # (Q_total, N)
            all_logits.append(logits)
            all_labels.append(qry_l[q_start:q_end])
    else:
        # Legacy PhaseAwareProtoNet: per-episode loop（编码和匹配不分离）
        for ep in range(B):
            s_start = ep * N * K;    s_end = s_start + N * K
            q_start = ep * Q_total;  q_end = q_start + Q_total
            logits = model(
                support_features=sup_f[s_start:s_end],
                support_masks=sup_m[s_start:s_end],
                query_features=qry_f[q_start:q_end],
                query_masks=qry_m[q_start:q_end],
                n_way=N, k_shot=K,
            )  # (Q_total, N)
            all_logits.append(logits)
            all_labels.append(qry_l[q_start:q_end])

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Train / Val loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, n_way, k_shot, device,
                scheduler=None, scaler=None):
    model.train()
    total_loss = total_acc = 0.0
    use_amp = scaler is not None and device.type == torch.device("cuda").type

    for sup_f, sup_m, qry_f, qry_m, qry_l in tqdm(loader, desc="Train", dynamic_ncols=True):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, labels = run_episode_batch(
                model, sup_f, sup_m, qry_f, qry_m, qry_l, n_way, k_shot, device
            )
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, n_way, k_shot, device, use_amp=False):
    model.eval()
    total_loss = total_acc = 0.0

    for sup_f, sup_m, qry_f, qry_m, qry_l in tqdm(loader, desc="Val  ", dynamic_ncols=True):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, labels = run_episode_batch(
                model, sup_f, sup_m, qry_f, qry_m, qry_l, n_way, k_shot, device
            )
            loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Warmup + cosine decay scheduler  (LR floor 防止后期衰减至 0)
# ---------------------------------------------------------------------------

def make_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio: float = 0.01):
    """
    线性 warmup + cosine decay，但 LR 不衰减到 0 而是保留 min_lr_ratio 比例。
    例如 lr=1e-4, min_lr_ratio=0.01 → 最低 LR = 1e-6。
    这样训练后期模型仍有足够梯度，不会因 LR=0 陷入过拟合静止态。
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        # 保留 min_lr_ratio，防止 LR 降到 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--dataset", type=str, default="ssv2",
                   choices=["ssv2", "hmdb51", "ucf101", "kinetics400"],
                   help="Dataset to use for training")
    p.add_argument("--ssv2_variant", type=str, default="small",
                   choices=["small", "full"],
                   help="SSV2 split variant: 'small'→splits/ssv2_small, 'full'→splits/ssv2_full. "
                        "Ignored when --dataset is not ssv2 or --splits_root is set explicitly.")
    p.add_argument("--features_root", type=str,
                   default="results/ssv2/extract_ssv2_svdxt_t20/feats/layer_1")
    p.add_argument("--data_root", type=str, default="../ssv2/labels")
    p.add_argument("--splits_root", type=str, default=None,
                   help="Override splits directory path directly. "
                        "If None, auto-resolved from --ssv2_variant (ssv2) "
                        "or splits/hmdb_ARN (hmdb51).")
    p.add_argument("--n_way",    type=int, default=5)
    p.add_argument("--k_shot",   type=int, default=1)
    p.add_argument("--n_query",  type=int, default=15)
    p.add_argument("--max_len",  type=int, default=50,
                   help="Sequence length after padding/truncation")
    p.add_argument("--num_train_episodes", type=int, default=6000)
    p.add_argument("--num_val_episodes",   type=int, default=600)
    p.add_argument("--num_test_episodes",  type=int, default=600)
    p.add_argument("--n_base_classes",     type=int, default=100,
                   help="Ignored when splits_root is used (splits file is authoritative)")
    p.add_argument("--n_val_classes",      type=int, default=37,
                   help="Ignored when splits_root is used (splits file is authoritative)")
    # Model
    p.add_argument("--hidden_dim",  type=int,   default=256,
                   help="Projection dim (256 saves ~4× VRAM vs 512)")
    p.add_argument("--n_phases",    type=int,   default=3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--distance",    type=str,   default="cosine",
                   choices=["cosine", "euclidean"])
    p.add_argument("--dropout",     type=float, default=0.5,
                   help="Dropout rate；小样本固定 episode 推荐 0.5 以强力压制记忆效应")
    p.add_argument("--head",        type=str,   default="proto",
                   choices=["proto", "trx"],
                   help="匹配头类型：proto=cosine prototype（推荐，无类相关参数，泛化强）"
                        "；trx=cross-attention TRX（参数多，易过拟合到基类）")
    p.add_argument("--amp",         action="store_true", default=False,
                   help="Enable automatic mixed precision (fp16) to reduce VRAM")
    # Training
    p.add_argument("--num_epochs",      type=int,   default=30)
    p.add_argument("--patience",        type=int,   default=10,
                   help="Early stopping patience（acc + loss 双指标均无改善才停止）")
    p.add_argument("--batch_size",      type=int,   default=4,
                   help="Number of episodes per batch")
    p.add_argument("--lr",              type=float, default=5e-5,
                   help="初始学习率；过拟合严重时降至 5e-5 减缓 logit 过度自信")
    p.add_argument("--lr_min_ratio",    type=float, default=0.05,
                   help="Cosine decay 最低 LR = lr * lr_min_ratio；5e-5 × 0.05 = 2.5e-6")
    p.add_argument("--weight_decay",    type=float, default=2e-3,
                   help="AdamW weight decay；固定小数据集推荐 2e-3（1e-3~5e-3 均可尝试）")
    p.add_argument("--label_smoothing", type=float, default=0.05,
                   help="CE label smoothing；0.05 比 0.1 更贴近 acc 曲线，减少 loss/acc 背离")
    p.add_argument("--num_warmup_epochs", type=int, default=5,
                   help="Warmup 轮数；LR 低时需要更长 warmup 才能稳定收敛")
    p.add_argument("--num_workers",     type=int,   default=0,
                   help="DataLoader workers (0=main process, safe on Windows)")
    p.add_argument("--seed",   type=int,   default=42)
    p.add_argument("--device", type=str,   default="cuda")
    # 数据增强（仅 dry-run 模式有效）
    p.add_argument("--feat_noise_std",  type=float, default=0.02,
                   help="训练增强：高斯噪声强度（相对特征标准差）")
    p.add_argument("--frame_drop_prob", type=float, default=0.10,
                   help="训练增强：随机帧掩码概率")
    p.add_argument("--patch_drop_prob", type=float, default=0.10,
                   help="训练增强：随机 patch 掩码概率（仅 4D 空间特征）")
    # 第二阶段：在 train+val 上微调
    p.add_argument("--finetune_on_val_epochs", type=int, default=0,
                   help="主训练结束后，在 train+val 合并数据上继续微调 N 轮（0=不微调）")
    # I/O
    p.add_argument("--save_dir",  type=str,  default="results/fsar_ssv2")
    p.add_argument("--ckpt_path", type=str,  default=None)
    p.add_argument("--wandb",     action="store_true")
    # Dry-run: load pre-saved episode_XXXX.pt files instead of SSV2FewShotDataset
    p.add_argument(
        "--dry_run_feats", type=str, default=None,
        help="Directory of episode_*.pt files for TRAIN (meta_train classes). "
             "Bypasses SSV2FewShotDataset."
    )
    p.add_argument(
        "--dry_run_val_feats", type=str, default=None,
        help="[Recommended] Separate directory of episode_*.pt for VAL "
             "(meta_val classes, disjoint from train). "
             "If omitted, val is cut from --dry_run_feats by index (class leakage!)."
    )
    p.add_argument(
        "--dry_run_test_feats", type=str, default=None,
        help="[Recommended] Separate directory of episode_*.pt for TEST "
             "(meta_test classes, disjoint from train/val). "
             "If omitted, test is cut from --dry_run_feats by index (class leakage!)."
    )
    # On-demand feature extraction
    p.add_argument("--on_demand", action="store_true", default=False,
                   help="按需提取：训练时发现缺失特征则立即用 SVD 提取，无需预先跑 generate_svd_maps.py")
    p.add_argument("--video_root", type=str, default=None,
                   help="原始视频根目录（data_root/{class}/{video}.avi），on_demand 模式必填")
    p.add_argument("--on_demand_diffusion_step",  type=int,   default=23)
    p.add_argument("--on_demand_diffusion_step2", type=int,   default=None)
    p.add_argument("--on_demand_mid_layer_idxs",  type=str,   default="7",  nargs='?', const="",
                   help="逗号分隔，如 '7'；传空字符串表示不提取该组")
    p.add_argument("--on_demand_deep_layer_idxs", type=str,   default="12", nargs='?', const="",
                   help="逗号分隔，如 '12'；传空字符串表示不提取该组")
    p.add_argument("--on_demand_mid_layer_idxs2", type=str,   default=None, nargs='?', const="")
    p.add_argument("--on_demand_deep_layer_idxs2",type=str,   default=None, nargs='?', const="")
    p.add_argument("--on_demand_version",         type=str,   default="svd_xt",
                   choices=["svd", "svd_xt"])
    p.add_argument("--on_demand_num_frames",      type=int,   default=25)
    p.add_argument("--on_demand_num_steps",       type=int,   default=30)
    p.add_argument("--on_demand_prefetch_all",    action="store_true", default=False,
                   help="初始化时提取所有类（等同于先跑 generate_svd_maps.py）")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Datasets ---
    print("Building datasets …")
    if args.dry_run_feats is not None:
        # ---- Dry-run path: load pre-saved episode_XXXX.pt files ----

        # ── Train dataset: always from --dry_run_feats，训练时开启增强 ──────
        all_train_pt = sorted(glob.glob(os.path.join(args.dry_run_feats, "episode_*.pt")))
        if not all_train_pt:
            raise FileNotFoundError(f"No episode_*.pt files in {args.dry_run_feats}")

        _aug_kwargs = dict(
            augment        = True,
            noise_std      = args.feat_noise_std,
            frame_drop_prob = args.frame_drop_prob,
            patch_drop_prob = args.patch_drop_prob,
        )
        train_dataset = DryRunEpisodeDataset(args.dry_run_feats, **_aug_kwargs)

        # ── Val dataset（评估时不做增强，保证指标可复现）───────────────────
        if args.dry_run_val_feats is not None:
            # CORRECT: separate class-isolated val episodes
            val_dataset = DryRunEpisodeDataset(args.dry_run_val_feats)
            val_src = f"{args.dry_run_val_feats}  [class-isolated ✓]"
        else:
            # FALLBACK (class leakage!): cut 10% of train as val
            print(
                "\n  *** WARNING: --dry_run_val_feats not set. ***\n"
                "  Val episodes are cut from the SAME directory as train.\n"
                "  Val and train SHARE the same classes → val acc is inflated.\n"
                "  Generate val episodes with split=meta_val and pass --dry_run_val_feats.\n"
            )
            n_total = len(all_train_pt)
            n_per   = max(1, (n_total // 10))
            n_train_s = n_total - 2 * n_per          # 剩余全部给 train
            if n_train_s <= 0:
                raise ValueError(
                    f"Too few episodes ({n_total}) to split train/val/test. "
                    "Provide --dry_run_val_feats and --dry_run_test_feats."
                )
            val_idx   = list(range(n_train_s, n_train_s + n_per))
            train_idx = list(range(0, n_train_s))
            train_dataset = DryRunEpisodeDataset(
                args.dry_run_feats, indices=train_idx, **_aug_kwargs
            )
            val_dataset = DryRunEpisodeDataset(
                args.dry_run_feats, indices=val_idx   # no augment for val
            )
            val_src = f"{args.dry_run_feats}  [WARNING: class leakage!]"

        # ── Test dataset ─────────────────────────────────────────────────
        if args.dry_run_test_feats is not None:
            # CORRECT: separate class-isolated test episodes
            test_dataset = DryRunEpisodeDataset(args.dry_run_test_feats)
            test_src = f"{args.dry_run_test_feats}  [class-isolated ✓]"
        else:
            # FALLBACK: 只从 train 目录切出独立的 10%（不与 val 重叠）
            # 注意：当 val 来自独立目录时，train 目录里没有 val 文件，
            # 所以此处从 train 目录尾部切出 test，存在类泄漏风险，需明确警告。
            print(
                "\n  *** WARNING: --dry_run_test_feats not set. ***\n"
                "  Test episodes are cut from the SAME directory as train.\n"
                "  Test acc does NOT measure generalization to novel classes.\n"
                "  Generate test episodes with split=meta_test and pass --dry_run_test_feats.\n"
            )
            n_total = len(all_train_pt)
            n_per   = max(1, (n_total // 10))
            # 如果 val 也是从 train 目录切出的，需偏移避免重叠
            if args.dry_run_val_feats is None:
                n_train_s = n_total - 2 * n_per
                test_start = n_train_s + n_per
            else:
                # val 在独立目录，train 目录全部给 train；test 从末尾切
                n_train_s  = n_total - n_per
                test_start = n_train_s
            test_idx = list(range(test_start, min(test_start + n_per, n_total)))
            if not test_idx:
                test_idx = [n_total - 1]   # 至少用最后一个 episode
            test_dataset = DryRunEpisodeDataset(
                args.dry_run_feats, indices=test_idx
            )
            test_src = f"{args.dry_run_feats}  [WARNING: class leakage!]"

        print(f"  [DryRun] Train  : {len(train_dataset)} episodes  ← {args.dry_run_feats}")
        print(f"  [DryRun] Val    : {len(val_dataset)} episodes  ← {val_src}")
        print(f"  [DryRun] Test   : {len(test_dataset)} episodes  ← {test_src}")
        _patch_info = (f"  P={train_dataset.n_patches} (spatial grid)"
                       if train_dataset.n_patches else "  [legacy GAP features]")
        print(f"  [DryRun] N-way={train_dataset.n_way}  K-shot=1"
              f"  T={train_dataset.seq_len}  D={train_dataset.embed_dim}{_patch_info}")
        print(f"  [Augment] noise_std={args.feat_noise_std}  "
              f"frame_drop={args.frame_drop_prob}  patch_drop={args.patch_drop_prob}")
        args.n_way  = train_dataset.n_way
        args.k_shot = 1
    else:
        # ---- Normal path: FewShotDataset with .npy features ----
        # Auto-resolve splits_root when not set explicitly
        _splits_root = args.splits_root
        if _splits_root is None:
            _scripts_dir = os.path.dirname(os.path.abspath(__file__))
            if args.dataset == "ssv2":
                _splits_root = os.path.normpath(
                    os.path.join(_scripts_dir, f"../../splits/ssv2_{args.ssv2_variant}")
                )
                print(f"  [Splits] SSV2 variant={args.ssv2_variant}  → {_splits_root}")
            elif args.dataset == "hmdb51":
                _splits_root = os.path.normpath(
                    os.path.join(_scripts_dir, "../../splits/hmdb_ARN")
                )
                print(f"  [Splits] HMDB51  → {_splits_root}")
            elif args.dataset == "ucf101":
                _splits_root = os.path.normpath(
                    os.path.join(_scripts_dir, "../../splits/ucf_ARN")
                )
                print(f"  [Splits] UCF101  → {_splits_root}")
            elif args.dataset == "kinetics400":
                _splits_root = os.path.normpath(
                    os.path.join(_scripts_dir, "../../splits/kinetics_CMN")
                )
                print(f"  [Splits] Kinetics400  → {_splits_root}")

        if args.on_demand:
            # ---- On-demand path: extract SVD features lazily per episode ----
            if not args.video_root:
                raise ValueError("--video_root is required when --on_demand is set")
            from datasets.on_demand_extractor import OnDemandFewShotDataset

            def _parse_idxs(s):
                return [int(x) for x in s.split(",") if x.strip()] if s else []

            # Dataset-specific splits file map and line format
            _SPLIT_FILE_MAPS = {
                "hmdb51":     {"base": "trainlist03.txt", "val": "vallist03.txt",  "test": "testlist03.txt"},
                "ucf101":     {"base": "trainlist03.txt", "val": "vallist03.txt",  "test": "testlist03.txt"},
                "ssv2":       {"base": "train.txt",       "val": "val.txt",        "test": "test.txt"},
                "kinetics400":{"base": "train.txt",       "val": "val.txt",        "test": "test.txt"},
            }
            _LINE_FORMATS = {
                "hmdb51": "class/video", "ucf101": "class/video",
                "ssv2": "class_only",    "kinetics400": "class_only",
            }
            _sfm = _SPLIT_FILE_MAPS.get(args.dataset, {"base": "train.txt", "val": "val.txt", "test": "test.txt"})
            _lf  = _LINE_FORMATS.get(args.dataset, "class_only")

            _od_cfg = dict(
                diffusion_step   = args.on_demand_diffusion_step,
                diffusion_step2  = args.on_demand_diffusion_step2,
                mid_layer_idxs   = _parse_idxs(args.on_demand_mid_layer_idxs),
                deep_layer_idxs  = _parse_idxs(args.on_demand_deep_layer_idxs),
                mid_layer_idxs2  = _parse_idxs(args.on_demand_mid_layer_idxs2) or None,
                deep_layer_idxs2 = _parse_idxs(args.on_demand_deep_layer_idxs2) or None,
                num_frames       = args.on_demand_num_frames,
                num_steps        = args.on_demand_num_steps,
                version          = args.on_demand_version,
                device           = args.device,
            )
            _od_common = dict(
                splits_root    = _splits_root,
                split_file_map = _sfm,
                line_format    = _lf,
                features_root  = args.features_root,
                video_root     = args.video_root,
                extractor_cfg  = _od_cfg,
                n_way          = args.n_way,
                k_shot         = args.k_shot,
                n_query        = args.n_query,
                max_len        = args.max_len,
                seed           = args.seed,
            )
            train_dataset = OnDemandFewShotDataset(split="base", num_episodes=args.num_train_episodes, **_od_common)
            val_dataset   = OnDemandFewShotDataset(split="val",  num_episodes=args.num_val_episodes,   **_od_common)
            test_dataset  = OnDemandFewShotDataset(split="test", num_episodes=args.num_test_episodes,  **_od_common)
            print(f"  [OnDemand] video_root={args.video_root}")
        else:
            FewShotDataset = (
                HMDB51FewShotDataset    if args.dataset == "hmdb51"     else
                UCF101FewShotDataset    if args.dataset == "ucf101"     else
                Kinetics400FewShotDataset if args.dataset == "kinetics400" else
                SSV2FewShotDataset
            )
            common_kwargs = dict(
                features_root=args.features_root,
                data_root=args.data_root,
                splits_root=_splits_root,
                n_way=args.n_way,
                k_shot=args.k_shot,
                n_query=args.n_query,
                max_len=args.max_len,
                use_mean=True,
                seed=args.seed,
            )
            train_dataset = FewShotDataset(split="base",  num_episodes=args.num_train_episodes, **common_kwargs)
            val_dataset   = FewShotDataset(split="val",   num_episodes=args.num_val_episodes,   **common_kwargs)
            test_dataset  = FewShotDataset(split="test",  num_episodes=args.num_test_episodes,  **common_kwargs)
            print(f"  Train classes : {len(train_dataset.episode_class_ids)}")
            print(f"  Val   classes : {len(val_dataset.episode_class_ids)}")
            print(f"  Test  classes : {len(test_dataset.episode_class_ids)}")
            print(f"  Feature dim   : {train_dataset.embed_dim}")

    _pin = device.type == "cuda"   # pin_memory 加速 CPU→GPU 传输，仅 CUDA 有效
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=episode_collate,
        pin_memory=_pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=episode_collate,
        pin_memory=_pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=episode_collate,
        pin_memory=_pin,
    )

    # --- Model ---
    # Auto-select: SpatioTemporalTRXNet for new (T,P,D) spatial features,
    # PhaseAwareProtoNet for legacy (T,D) GAP features.
    # On-demand mode: bootstrap extraction if embed_dim not yet known.
    if getattr(train_dataset, "embed_dim", None) is None:
        print("  [OnDemand] Bootstrapping first class to determine embed_dim …")
        train_dataset._ensure_extracted(train_dataset._all_classes[:1])
    n_patches = getattr(train_dataset, "n_patches", None)
    if n_patches is not None:
        model = SpatioTemporalTRXNet(
            input_dim   = train_dataset.embed_dim,
            hidden_dim  = args.hidden_dim,
            n_phases    = args.n_phases,
            n_heads     = 4,
            max_patches = max(64, n_patches),
            n_patches   = n_patches,
            dropout     = args.dropout,
            head_type   = args.head,
        ).to(device)
        print(f"  [Model] SpatioTemporalTRXNet  P={n_patches}  H={args.hidden_dim}"
              f"  head={args.head}")
    else:
        model = PhaseAwareProtoNet(
            input_dim   = train_dataset.embed_dim,
            hidden_dim  = args.hidden_dim,
            n_phases    = args.n_phases,
            temperature = args.temperature,
            distance    = args.distance,
            dropout     = args.dropout,
        ).to(device)
        print(f"  [Model] PhaseAwareProtoNet (legacy)  H={args.hidden_dim}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # label_smoothing 降为 0.05：使 val loss 与 val acc 曲线方向保持一致；
    # 0.1 会在模型对错误预测过于自信时让 loss 飙升而 acc 仍微升，形成 U 型。
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # AMP scaler – only active when --amp and CUDA available
    use_amp = args.amp and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  [AMP] Mixed precision (fp16) enabled – VRAM ~halved")

    steps_per_epoch = len(train_loader)
    total_steps  = steps_per_epoch * args.num_epochs
    warmup_steps = steps_per_epoch * args.num_warmup_epochs

    start_epoch       = 0
    best_val_acc      = 0.0
    best_val_loss     = float("inf")
    epochs_no_improve = 0   # 双指标均无改善的轮数

    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch       = ckpt.get("epoch", 0)
        best_val_acc      = ckpt.get("val_acc",  0.0)
        best_val_loss     = ckpt.get("val_loss",  float("inf"))
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        # Resume 时用剩余轮次重建调度器，不再加载旧 scheduler state，
        # 避免 total_steps 不一致导致 LR 异常跳变。
        remaining = max(1, args.num_epochs - start_epoch)
        total_steps  = steps_per_epoch * remaining
        warmup_steps = 0   # resume 无需再 warmup
        print(f"Resumed from epoch {start_epoch},"
              f" best val acc={best_val_acc:.4f}, best val loss={best_val_loss:.4f}")

    scheduler = make_scheduler(
        optimizer, warmup_steps, total_steps, min_lr_ratio=args.lr_min_ratio
    )

    # --- WandB (optional) ---
    if args.wandb:
        import wandb
        wandb.init(project="fsar-ssv2", config=vars(args))

    log_path = os.path.join(args.save_dir, "training_log.txt")
    # Append mode when resuming so previous epochs are not lost
    log_mode = "a" if (args.ckpt_path is not None and os.path.exists(log_path)) else "w"
    with open(log_path, log_mode) as f:
        if log_mode == "w":
            f.write("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\tlr\n")

    # --- Training loop ---
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            args.n_way, args.k_shot, device, scheduler, scaler
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, args.n_way, args.k_shot, device, use_amp
        )
        current_lr = optimizer.param_groups[0]["lr"]
        gap = train_acc - val_acc   # 过拟合程度指示

        print(
            f"  train  loss={train_loss:.4f}  acc={train_acc*100:.2f}%"
            f"\n  val    loss={val_loss:.4f}  acc={val_acc*100:.2f}%"
            f"  (gap={gap*100:+.1f}%)"
            f"\n  lr={current_lr:.2e}"
        )

        if args.wandb:
            import wandb
            wandb.log({
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss":   val_loss,   "val/acc":   val_acc,
                "overfit_gap": gap,
                "lr": current_lr, "epoch": epoch + 1,
            })

        with open(log_path, "a") as f:
            f.write(f"{epoch+1}\t{train_loss:.4f}\t{train_acc:.4f}\t"
                    f"{val_loss:.4f}\t{val_acc:.4f}\t{current_lr:.6f}\n")

        # ── 双指标模型保存 ────────────────────────────────────────────────
        # 规则：val_acc 提升 → 保存 best_model.pth（主指标）
        #       val_loss 降低 → 保存 best_loss_model.pth（辅助指标）
        # 这样即使出现 loss/acc 背离，两个视角的最优模型都被保留。
        ckpt_payload = {
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc":              val_acc,
            "val_loss":             val_loss,
            "epochs_no_improve":    epochs_no_improve,
        }

        acc_improved  = val_acc  > best_val_acc
        loss_improved = val_loss < best_val_loss

        if acc_improved:
            best_val_acc = val_acc
            torch.save(ckpt_payload,
                       os.path.join(args.save_dir, "best_model.pth"))
            print(f"  *** saved best_model (val_acc={val_acc*100:.2f}%) ***")

        if loss_improved:
            best_val_loss = val_loss
            torch.save(ckpt_payload,
                       os.path.join(args.save_dir, "best_loss_model.pth"))
            print(f"  *** saved best_loss_model (val_loss={val_loss:.4f}) ***")

        # ── 双指标早停 ────────────────────────────────────────────────────
        # 只有当 acc 和 loss 双双没有改善时才计数；任意一个改善则重置。
        # 这解决了"loss 升但 acc 也升"时不应停止的问题。
        if acc_improved or loss_improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(
                    f"\n  [Early Stop] val acc 和 val loss 均连续 {args.patience} 轮无改善，停止训练。"
                )
                break

        # 保存 latest checkpoint（断点续训用）
        torch.save(ckpt_payload,
                   os.path.join(args.save_dir, "latest_model.pth"))

    # ── 第二阶段：train + val 合并微调（--finetune_on_val_epochs > 0）────
    if args.finetune_on_val_epochs > 0:
        print(f"\n{'='*60}")
        print(f"[Stage-2] Fine-tuning on train+val for "
              f"{args.finetune_on_val_epochs} epochs …")
        print(f"  加载 best_model.pth (val_acc={best_val_acc*100:.2f}%) 作为起点")
        print(f"{'='*60}")

        best_ckpt = torch.load(
            os.path.join(args.save_dir, "best_model.pth"), map_location=device
        )
        model.load_state_dict(best_ckpt["model_state_dict"])

        # val 数据集在微调时也开启增强，与 train 同等处理
        from torch.utils.data import ConcatDataset
        if isinstance(val_dataset, DryRunEpisodeDataset):
            val_aug_dataset = DryRunEpisodeDataset(
                val_dataset.files[0].rsplit("/", 1)[0]
                if hasattr(val_dataset, "files") else args.dry_run_val_feats,
                # 直接重用 files 列表，不重扫目录
                augment=True,
                noise_std=args.feat_noise_std,
                frame_drop_prob=args.frame_drop_prob,
                patch_drop_prob=args.patch_drop_prob,
            )
            # 覆盖 files 以保证与原 val_dataset 完全相同的 episode 集合
            val_aug_dataset.files = list(val_dataset.files)
        else:
            val_aug_dataset = val_dataset   # SSV2FewShotDataset 已内置随机采样

        merged_dataset = ConcatDataset([train_dataset, val_aug_dataset])
        merged_loader  = DataLoader(
            merged_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=episode_collate,
        )

        ft_steps     = len(merged_loader) * args.finetune_on_val_epochs
        ft_optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr * 0.1,            # 微调 LR = 原始的 10%
            weight_decay=args.weight_decay,
        )
        ft_scheduler = make_scheduler(
            ft_optimizer, warmup_steps=0,
            total_steps=ft_steps, min_lr_ratio=args.lr_min_ratio
        )
        ft_scaler = torch.amp.GradScaler("cuda") if use_amp else None

        for ft_ep in range(args.finetune_on_val_epochs):
            print(f"\n[Stage-2] Epoch {ft_ep+1}/{args.finetune_on_val_epochs}")
            ft_loss, ft_acc = train_epoch(
                model, merged_loader, criterion, ft_optimizer,
                args.n_way, args.k_shot, device, ft_scheduler, ft_scaler
            )
            ft_lr = ft_optimizer.param_groups[0]["lr"]
            print(f"  finetune  loss={ft_loss:.4f}  acc={ft_acc*100:.2f}%"
                  f"  lr={ft_lr:.2e}")

            if args.wandb:
                import wandb
                wandb.log({
                    "finetune/loss": ft_loss, "finetune/acc": ft_acc,
                    "finetune/lr":   ft_lr,
                    "finetune_epoch": ft_ep + 1,
                })

        ft_save_path = os.path.join(args.save_dir, "finetune_model.pth")
        torch.save({"model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc}, ft_save_path)
        print(f"\n  [Stage-2] 微调完成，模型已保存 → {ft_save_path}")

    # --- Final test evaluation ---
    print("\nLoading best model for test evaluation …")
    best_ckpt = torch.load(
        os.path.join(args.save_dir, "best_model.pth"), map_location=device
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, args.n_way, args.k_shot, device, use_amp
    )
    print(f"Test  loss={test_loss:.4f}  acc={test_acc*100:.2f}%")
    print(f"Best val acc: {best_val_acc*100:.2f}%")

    with open(log_path, "a") as f:
        f.write(f"\ntest\t{test_loss:.4f}\t{test_acc:.4f}\n")
        f.write(f"best_val_acc\t{best_val_acc:.4f}\n")

    if args.wandb:
        import wandb
        wandb.log({"test/loss": test_loss, "test/acc": test_acc})
        wandb.finish()


if __name__ == "__main__":
    main()
