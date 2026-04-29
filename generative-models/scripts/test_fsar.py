"""
test_fsar.py – Standalone evaluation script for PhaseAwareProtoNet on SSV2-small.

Outputs mean accuracy ± 95% CI over N episodes, suitable for comparison
with published FSAR benchmarks.

Usage (dry-run .pt episodes):
    python scripts/test_fsar.py \
        --ckpt_path results/fsar_dryrun/best_model.pth \
        --dry_run_feats results/ssv2/ssv2_episodic_train/feats \
        --n_episodes 600 \
        --device cuda

Usage (standard SSV2 .npy features):
    python scripts/test_fsar.py \
        --ckpt_path results/fsar_ssv2/best_model.pth \
        --features_root results/ssv2/extract_ssv2_svdxt_t20/feats/layer_1 \
        --data_root ../ssv2/labels \
        --split test \
        --n_way 5 --k_shot 1 --n_query 15 \
        --n_episodes 600 \
        --device cuda
"""

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from datasets.ssv2 import SSV2FewShotDataset
from datasets.hmdb51 import HMDB51FewShotDataset
from datasets.ucf101 import UCF101FewShotDataset
from datasets.kinetics400 import Kinetics400FewShotDataset
from models.FSARModel import PhaseAwareProtoNet, SpatioTemporalTRXNet


# ---------------------------------------------------------------------------
# DryRunEpisodeDataset  (identical to the one in train_fsar.py)
# ---------------------------------------------------------------------------

class DryRunEpisodeDataset(Dataset):
    """
    Reads pre-saved episode_XXXX.pt files.

    Each file must contain:
        support_feats : (N, T, D) or (N, T, P, D)  – one support per class (K=1)
        query_feats   : (N*Q, T, D) or (N*Q, T, P, D)
        query_y       : (N*Q,)  – labels in [0, N)

    Returns:
        sup_f  : (N, 1, T, D) or (N, 1, T, P, D)  float
        sup_m  : (N, 1, T)   all-ones mask
        qry_f  : (N*Q, T, D) or (N*Q, T, P, D)    float
        qry_m  : (N*Q, T)    all-ones mask
        qry_l  : (N*Q,)      long
    """

    def __init__(self, feats_dir: str, indices=None):
        all_files = sorted(glob.glob(os.path.join(feats_dir, "episode_*.pt")))
        if not all_files:
            raise FileNotFoundError(f"No episode_*.pt files found in {feats_dir}")
        self.files = [all_files[i] for i in indices] if indices is not None else all_files

        ep0 = torch.load(self.files[0], map_location="cpu")
        sf  = ep0["support_feats"]
        self.embed_dim = sf.shape[-1]
        self.n_way     = sf.shape[0]
        self.seq_len   = sf.shape[1]
        self.n_patches = sf.shape[2] if sf.ndim == 4 else None  # P or None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ep    = torch.load(self.files[idx], map_location="cpu")
        sup_f = ep["support_feats"].float()   # (N, T, D) or (N, T, P, D)
        qry_f = ep["query_feats"].float()     # (N*Q, T, D) or (N*Q, T, P, D)
        qry_l = ep["query_y"].long()

        N  = sup_f.shape[0]
        T  = sup_f.shape[1]
        NQ = qry_f.shape[0]

        sup_f = sup_f.unsqueeze(1)           # (N, K=1, T, ...)
        sup_m = torch.ones(N, 1, T)          # mask is T-only
        qry_m = torch.ones(NQ, T)

        return sup_f, sup_m, qry_f, qry_m, qry_l


# ---------------------------------------------------------------------------
# Collate & forward helpers
# ---------------------------------------------------------------------------

def episode_collate(batch):
    sup_f, sup_m, qry_f, qry_m, qry_l = zip(*batch)
    return (
        torch.stack(sup_f),
        torch.stack(sup_m),
        torch.stack(qry_f),
        torch.stack(qry_m),
        torch.stack(qry_l),
    )


@torch.no_grad()
def evaluate_episodes(model, loader, device, use_amp=False):
    """
    Run model over all episodes in loader.

    Returns:
        per_episode_acc : list[float]  – accuracy for each episode
    """
    model.eval()
    per_episode_acc = []

    for sup_f, sup_m, qry_f, qry_m, qry_l in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
        Q_total = qry_f.shape[1]

        if sup_f.ndim == 6:
            B, N, K, T, P, D = sup_f.shape
            sup_f_dev = sup_f.view(B * N * K, T, P, D).to(device)
            qry_f_dev = qry_f.view(B * Q_total, T, P, D).to(device)
        else:
            B, N, K, T, D = sup_f.shape
            sup_f_dev = sup_f.view(B * N * K, T, D).to(device)
            qry_f_dev = qry_f.view(B * Q_total, T, D).to(device)

        sup_m_dev = sup_m.view(B * N * K, T).to(device)
        qry_m_dev = qry_m.view(B * Q_total, T).to(device)
        qry_l_dev = qry_l.view(B * Q_total).to(device)

        for ep in range(B):
            s0, s1 = ep * N * K,  (ep + 1) * N * K
            q0, q1 = ep * Q_total, (ep + 1) * Q_total

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(
                    support_features=sup_f_dev[s0:s1],
                    support_masks=sup_m_dev[s0:s1],
                    query_features=qry_f_dev[q0:q1],
                    query_masks=qry_m_dev[q0:q1],
                    n_way=N,
                    k_shot=K,
                )  # (Q_total, N)

            preds  = logits.argmax(dim=-1)
            labels = qry_l_dev[q0:q1]
            acc    = (preds == labels).float().mean().item()
            per_episode_acc.append(acc)

    return per_episode_acc


def ci95(values):
    """95% confidence interval: 1.96 * std / sqrt(n)."""
    arr = np.array(values)
    return 1.96 * arr.std() / math.sqrt(len(arr))


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate PhaseAwareProtoNet on SSV2-small FSAR benchmark."
    )
    # Checkpoint
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Path to best_model.pth produced by train_fsar.py")
    p.add_argument("--dataset", type=str, default="ssv2",
                   choices=["ssv2", "hmdb51", "ucf101", "kinetics400"],
                   help="Dataset to evaluate on")
    p.add_argument("--ssv2_variant", type=str, default="small",
                   choices=["small", "full"],
                   help="SSV2 split variant. Ignored when --dataset is not ssv2.")
    # Data – dry-run mode
    p.add_argument("--dry_run_feats", type=str, default=None,
                   help="Directory of episode_*.pt files. "
                        "Pass meta_test episodes here for a clean evaluation. "
                        "Falls back to temporal slice (class leakage) if "
                        "--dry_run_split is used without class-isolated files.")
    p.add_argument("--dry_run_split", type=str, default="all",
                   choices=["train", "val", "test", "all"],
                   help="Temporal slice to use when --dry_run_feats contains mixed classes. "
                        "'all' uses every file in the directory (correct when the directory "
                        "already contains only meta_test episodes).")
    # Data – standard SSV2 mode
    p.add_argument("--features_root", type=str,
                   default="results/ssv2/extract_ssv2_svdxt_t20/feats/layer_1")
    p.add_argument("--data_root",     type=str, default="../ssv2/labels")
    p.add_argument("--splits_root",   type=str, default=None,
                   help="Path to splits/{hmdb_ARN,ssv2_OTAM}/ dir. "
                        "None = auto-detect from project root.")
    p.add_argument("--split",         type=str, default="test",
                   choices=["base", "val", "test"])
    p.add_argument("--n_way",    type=int, default=5)
    p.add_argument("--k_shot",   type=int, default=1)
    p.add_argument("--n_query",  type=int, default=15)
    p.add_argument("--max_len",  type=int, default=50)
    p.add_argument("--n_base_classes", type=int, default=100,
                   help="Ignored when splits_root is used")
    p.add_argument("--n_val_classes",  type=int, default=37,
                   help="Ignored when splits_root is used")
    p.add_argument("--seed",           type=int, default=42)
    # Evaluation
    p.add_argument("--n_episodes",  type=int, default=600,
                   help="Number of episodes to evaluate (standard benchmark = 600)")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    # Model architecture (must match training)
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--n_phases",    type=int,   default=3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--distance",    type=str,   default="cosine",
                   choices=["cosine", "euclidean"])
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--head",        type=str,   default="proto",
                   choices=["proto", "trx"],
                   help="匹配头类型，必须与训练时一致")
    # Misc
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--amp",        action="store_true", default=False,
                   help="Enable fp16 autocast during evaluation")
    p.add_argument("--output_csv", type=str, default=None,
                   help="Optional path to save per-episode accuracies as CSV")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.dry_run_feats is not None:
        all_files = sorted(glob.glob(os.path.join(args.dry_run_feats, "episode_*.pt")))
        n_total = len(all_files)
        if n_total == 0:
            raise FileNotFoundError(f"No episode_*.pt files in {args.dry_run_feats}")

        # ── Class-isolation check ──────────────────────────────────────
        # If --dry_run_split != "all", the user is slicing a mixed-class directory
        # by episode index → classes in the test slice overlap with training classes.
        if args.dry_run_split != "all":
            print(
                "\n  *** DATA-LEAKAGE WARNING ***\n"
                f"  --dry_run_split='{args.dry_run_split}' slices {args.dry_run_feats}\n"
                "  by episode index. The evaluated episodes come from the SAME class\n"
                "  pool as training episodes → accuracy is inflated (not novel-class).\n"
                "  For a clean benchmark, generate episodes with split=meta_test and\n"
                "  point --dry_run_feats at that directory with --dry_run_split=all.\n"
            )
            n_usable = (n_total // 10) * 10
            n_per    = n_usable // 10
            n_train  = 8 * n_per
            n_val    = n_per
            n_test   = n_per
            split_indices = {
                "train": list(range(0, n_train)),
                "val":   list(range(n_train, n_train + n_val)),
                "test":  list(range(n_train + n_val, n_train + n_val + n_test)),
            }
            indices = split_indices[args.dry_run_split]
        else:
            # "all" = the directory is already class-isolated (recommended)
            indices = list(range(n_total))

        # Cap to requested number of episodes
        if len(indices) < args.n_episodes:
            print(
                f"  [warn] Requested {args.n_episodes} episodes but only "
                f"{len(indices)} available. Using all {len(indices)}."
            )
        else:
            indices = indices[: args.n_episodes]

        dataset = DryRunEpisodeDataset(args.dry_run_feats, indices=indices)
        embed_dim = dataset.embed_dim
        n_way     = dataset.n_way
        k_shot    = 1

        _fmt = (f"  P={dataset.n_patches} [spatial]" if dataset.n_patches else "  [legacy GAP]")
        print(f"[DryRun] Episodes : {len(dataset)}  split='{args.dry_run_split}'")
        print(f"         N-way={n_way}  K-shot={k_shot}  T={dataset.seq_len}  D={embed_dim}{_fmt}")

    else:
        # Auto-resolve splits_root when not set explicitly
        _splits_root = args.splits_root
        if _splits_root is None:
            _scripts_dir = os.path.dirname(os.path.abspath(__file__))
            if args.dataset == "ssv2":
                _splits_root = os.path.normpath(
                    os.path.join(_scripts_dir, f"../../splits/ssv2_{args.ssv2_variant}")
                )
                print(f"  [Splits] SSV2 variant={args.ssv2_variant}  → {_splits_root}")
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

        FewShotDataset = (
            HMDB51FewShotDataset    if args.dataset == "hmdb51"     else
            UCF101FewShotDataset    if args.dataset == "ucf101"     else
            Kinetics400FewShotDataset if args.dataset == "kinetics400" else
            SSV2FewShotDataset
        )
        dataset = FewShotDataset(
            features_root=args.features_root,
            data_root=args.data_root,
            splits_root=_splits_root,
            split=args.split,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_query=args.n_query,
            max_len=args.max_len,
            use_mean=True,
            num_episodes=args.n_episodes,
            seed=args.seed,
        )
        embed_dim = dataset.embed_dim
        n_way     = args.n_way
        k_shot    = args.k_shot
        print(f"[{args.dataset.upper()}] Episodes={len(dataset)}  split='{args.split}'  D={embed_dim}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=episode_collate,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # Auto-select based on feature format detected from the dataset.
    n_patches = getattr(dataset, "n_patches", None)
    if n_patches is not None:
        model = SpatioTemporalTRXNet(
            input_dim   = embed_dim,
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
            input_dim   = embed_dim,
            hidden_dim  = args.hidden_dim,
            n_phases    = args.n_phases,
            temperature = args.temperature,
            distance    = args.distance,
            dropout     = args.dropout,
        ).to(device)
        print(f"  [Model] PhaseAwareProtoNet (legacy)  H={args.hidden_dim}")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)   # handle both wrapped and raw state dicts
    model.load_state_dict(state)
    trained_epoch = ckpt.get("epoch", "?")
    trained_acc   = ckpt.get("val_acc", float("nan"))
    print(f"Loaded checkpoint: epoch={trained_epoch}  val_acc={trained_acc*100:.2f}%")
    print(f"Model parameters : {sum(p.numel() for p in model.parameters()):,}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        print("  [AMP] fp16 autocast enabled")
    per_ep = evaluate_episodes(model, loader, device, use_amp)

    mean_acc = np.mean(per_ep) * 100
    ci       = ci95(per_ep)    * 100

    print()
    print("=" * 52)
    print(f"  {n_way}-way {k_shot}-shot  |  {len(per_ep)} episodes")
    print(f"  Accuracy : {mean_acc:.2f}% ± {ci:.2f}%  (95% CI)")
    print("=" * 52)

    # ── Optional CSV ──────────────────────────────────────────────────────────
    if args.output_csv is not None:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "accuracy"])
            for i, acc in enumerate(per_ep):
                writer.writerow([i, f"{acc:.6f}"])
        print(f"Per-episode results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
