"""
visualize_clusters.py – 3-stage t-SNE clustering for 5-way 1-shot FSAR.

Visualises how well features separate action classes at three stages:

  Stage 1 – Raw pixels (before SVD extraction)
             Expected: no structure, random scatter.
  Stage 2 – SVD-extracted features (*_mean.npy)
             Expected: loose class clusters.
  Stage 3 – PhaseAwareProtoNet encoder output
             Expected: tight, well-separated class clusters.

Each stage produces one t-SNE scatter plot.  All three are saved as a
single side-by-side PNG and also as individual PNGs.

Supported datasets: ssv2-small, ssv2-full, hmdb51, ucf101

Usage (run from generative-models/):

    # HMDB51 – test split, random 5-way episode
    python scripts/visualize_clusters.py \\
        --dataset hmdb51 \\
        --features_root results/hmdb51/extract_hmdb51/feats/combined \\
        --data_root ../hmdb51/videos \\
        --ckpt_path results/fsar_hmdb51/best_model.pth \\
        --split test \\
        --device cuda

    # SSV2-small – no raw-video stage (skip Stage 1)
    python scripts/visualize_clusters.py \\
        --dataset ssv2-small \\
        --features_root results/ssv2/extract_ssv2_svdxt_t20/feats/combined \\
        --ckpt_path results/fsar_dryrun/best_model.pth \\
        --split test \\
        --skip_raw \\
        --device cuda
"""

import argparse
import glob
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from models.FSARModel import PhaseAwareProtoNet, SpatioTemporalTRXNet


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_DATASET_CFG = {
    "ssv2-small": {
        "splits_root_rel": "../../splits/ssv2_small",
        "split_file_map":  {"base": "train_few_shot.txt",
                            "val":  "val_few_shot.txt",
                            "test": "test_few_shot.txt"},
        "feat_suffix":     "_mean.npy",
        "video_layout":    "flat",          # videos directly under data_root
        "video_exts":      (".webm", ".mp4"),
    },
    "ssv2-full": {
        "splits_root_rel": "../../splits/ssv2_full",
        "split_file_map":  {"base": "train_few_shot.txt",
                            "val":  "val_few_shot.txt",
                            "test": "test_few_shot.txt"},
        "feat_suffix":     "_mean.npy",
        "video_layout":    "flat",
        "video_exts":      (".webm", ".mp4"),
    },
    "hmdb51": {
        "splits_root_rel": "../../splits/hmdb_ARN",
        "split_file_map":  {"base": "trainlist03.txt", "val": "vallist03.txt", "test": "testlist03.txt"},
        "feat_suffix":     "_mean.npy",
        "video_layout":    "class_subdir",  # data_root/{class}/{video}.avi
        "video_exts":      (".avi", ".mp4"),
        "line_format":     "class/video",   # each line = "class_name/video_stem"
    },
    "ucf101": {
        "splits_root_rel": "../../splits/ucf_ARN",
        "split_file_map":  {"base": "trainlist03.txt", "val": "vallist03.txt", "test": "testlist03.txt"},
        "feat_suffix":     "_grid.pt",
        "video_layout":    "class_subdir",
        "video_exts":      (".avi", ".mp4"),
        "line_format":     "class/video",
    },
}

_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]

# Pastel fill versions (alpha hulls)
_COLORS_FILL = [
    "#f4a0b0", "#a8e6b0", "#a8b8f0", "#fad0a0", "#d0a8e0",
    "#a8eef8", "#f0a8f0", "#e8f8a0", "#fce8f0", "#a8d8d0",
]


def _load_classes(splits_root: str, split_file: str, line_format: str = "class_name") -> list:
    """
    Load class names from a splits file.

    line_format="class_name"  → each line is a class name (ssv2)
    line_format="class/video" → each line is "class_name/video_stem" (hmdb51, ucf101)
                                 returns deduplicated class names in order of appearance
    """
    fpath = os.path.join(splits_root, split_file)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Splits file not found: {fpath}")
    with open(fpath, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    if line_format == "class/video":
        seen, classes = set(), []
        for line in lines:
            # strip leading "index\t" prefix (ucf101 / hmdb51 ARN format)
            if "\t" in line:
                line = line.split("\t", 1)[1]
            cls = line.split("/", 1)[0]
            if cls not in seen:
                seen.add(cls)
                classes.append(cls)
        return classes
    else:
        return lines


def _find_feat_files(features_root: str, class_name: str, suffix: str) -> list:
    """Return list of feature file paths for a given class."""
    pattern = os.path.join(features_root, f"{class_name}__*{suffix}")
    files = sorted(glob.glob(pattern))
    # also try stripping trailing _grid from suffix for .pt files
    if not files and suffix == "_grid.pt":
        pattern2 = os.path.join(features_root, f"{class_name}__*_grid.pt")
        files = sorted(glob.glob(pattern2))
    return files


def _load_feat(path: str) -> torch.Tensor:
    """Load feature file → float32 tensor (T, D).
    Supports *_mean.npy (T,D) and *_grid.pt dict with key 'feats' (T,P,D) → mean over P.
    """
    if path.endswith(".pt"):
        d = torch.load(path, map_location="cpu")
        if isinstance(d, dict):
            t = d["feats"].float()   # (T, P, D)
            if t.ndim == 3:
                t = t.mean(dim=1)    # (T, D)
        else:
            t = d.float()
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t
    arr = np.load(path).astype(np.float32)
    t = torch.from_numpy(arr)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t   # (T, D)


def _gap_feat(feat: torch.Tensor) -> np.ndarray:
    """Global average pool over T → (D,) numpy."""
    return feat.mean(dim=0).numpy()


def _find_video(data_root: str, class_name: str, video_stem: str, exts: tuple) -> str:
    """Locate raw video file; returns path or None."""
    for ext in exts:
        p = os.path.join(data_root, class_name, f"{video_stem}{ext}")
        if os.path.exists(p):
            return p
        # flat layout (SSV2)
        p2 = os.path.join(data_root, f"{video_stem}{ext}")
        if os.path.exists(p2):
            return p2
    return None


def _video_to_raw_feat(video_path: str, n_frames: int = 16) -> np.ndarray:
    """
    Load a video, uniformly sample n_frames, resize to 64×64, flatten.
    Returns (n_frames * 64 * 64 * 3,) float32 numpy array.
    """
    try:
        import torchvision.io as tvio
        video, _, _ = tvio.read_video(video_path, pts_unit="sec")
        # video: (T, H, W, C) uint8
        T = video.shape[0]
        if T == 0:
            return np.zeros(n_frames * 64 * 64 * 3, dtype=np.float32)
        indices = np.linspace(0, T - 1, n_frames, dtype=int)
        frames = video[indices]                          # (n_frames, H, W, C)
        frames = frames.float() / 255.0
        frames = frames.permute(0, 3, 1, 2)             # (n_frames, C, H, W)
        frames = F.interpolate(frames, size=(64, 64), mode="bilinear", align_corners=False)
        return frames.reshape(-1).numpy()
    except Exception as e:
        print(f"  [warn] Could not load video {video_path}: {e}")
        return np.zeros(n_frames * 64 * 64 * 3, dtype=np.float32)


# ---------------------------------------------------------------------------
# Episode sampler
# ---------------------------------------------------------------------------

def sample_episode(
    features_root: str,
    classes: list,
    n_way: int,
    k_shot: int,
    n_query: int,
    feat_suffix: str,
    rng: random.Random,
) -> dict:
    """
    Sample a single N-way K-shot episode.

    Returns dict with keys:
        classes        : list[str]  length N
        support_paths  : list[list[str]]  [N][K]
        query_paths    : list[list[str]]  [N][Q]
        support_feats  : (N*K, D) float32 numpy
        query_feats    : (N*Q, D) float32 numpy
        query_labels   : (N*Q,)  int numpy
    """
    eligible = [
        c for c in classes
        if len(_find_feat_files(features_root, c, feat_suffix)) >= k_shot + n_query
    ]
    if len(eligible) < n_way:
        raise RuntimeError(
            f"Only {len(eligible)} classes have ≥{k_shot + n_query} features "
            f"(need n_way={n_way}). Lower n_query or check features_root."
        )

    chosen = rng.sample(eligible, n_way)
    support_paths, query_paths = [], []
    for cls in chosen:
        files = _find_feat_files(features_root, cls, feat_suffix)
        sampled = rng.sample(files, k_shot + n_query)
        support_paths.append(sampled[:k_shot])
        query_paths.append(sampled[k_shot:])

    sup_feats, qry_feats, qry_labels = [], [], []
    for n, cls in enumerate(chosen):
        for p in support_paths[n]:
            sup_feats.append(_gap_feat(_load_feat(p)))
        for p in query_paths[n]:
            qry_feats.append(_gap_feat(_load_feat(p)))
            qry_labels.append(n)

    return {
        "classes":       chosen,
        "support_paths": support_paths,
        "query_paths":   query_paths,
        "support_feats": np.stack(sup_feats),   # (N*K, D)
        "query_feats":   np.stack(qry_feats),   # (N*Q, D)
        "query_labels":  np.array(qry_labels),  # (N*Q,)
    }


# ---------------------------------------------------------------------------
# Model embedding
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_model_embeddings(
    model: torch.nn.Module,
    support_feats: np.ndarray,   # (N*K, D)
    query_feats: np.ndarray,     # (N*Q, D)
    n_way: int,
    k_shot: int,
    device: torch.device,
) -> np.ndarray:
    """
    Pass GAP features through the model encoder and return
    encoded query embeddings as numpy for t-SNE.

    Handles both PhaseAwareProtoNet (T,D) and SpatioTemporalTRXNet (T,P,D).
    Since cluster vis uses GAP features (D,), we treat each sample as T=1, P=1.
    """
    model.eval()

    # GAP features: (N, D) → add T=1 dim → (N, 1, D)
    sup = torch.from_numpy(support_feats).float().unsqueeze(1).to(device)  # (N*K, 1, D)
    qry = torch.from_numpy(query_feats).float().unsqueeze(1).to(device)    # (N*Q, 1, D)

    if hasattr(model, "encoder"):
        enc = model.encoder
        enc_class = type(enc).__name__

        if enc_class == "SpatioTemporalEncoder":
            # Needs (B, T, P, D) — fake P=1
            sup_in = sup.unsqueeze(2)   # (N*K, 1, 1, D)
            qry_in = qry.unsqueeze(2)   # (N*Q, 1, 1, D)
            enc_sup = enc(sup_in)       # (N*K, n_phases, 1, H)
            enc_qry = enc(qry_in)       # (N*Q, n_phases, 1, H)
            enc_sup = enc_sup.reshape(enc_sup.shape[0], -1)
            enc_qry = enc_qry.reshape(enc_qry.shape[0], -1)
        else:
            # PhaseAwareEncoder: (B, T, D) + mask
            sup_m = torch.ones(sup.shape[0], 1, device=device)
            qry_m = torch.ones(qry.shape[0], 1, device=device)
            enc_sup = enc(sup, sup_m)   # (N*K, n_phases, H)
            enc_qry = enc(qry, qry_m)   # (N*Q, n_phases, H)
            if enc_sup.ndim == 3:
                enc_sup = enc_sup.mean(dim=1)
                enc_qry = enc_qry.mean(dim=1)
    else:
        enc_qry = qry.squeeze(1)

    return enc_qry.cpu().numpy()


# ---------------------------------------------------------------------------
# t-SNE helpers
# ---------------------------------------------------------------------------

def run_tsne(feats: np.ndarray, perplexity: int = 15, seed: int = 42) -> np.ndarray:
    """Run t-SNE on (N, D) features → (N, 2)."""
    n = feats.shape[0]
    perp = min(perplexity, n - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed,
                max_iter=1000, init="pca" if feats.shape[1] > 2 else "random")
    return tsne.fit_transform(feats.astype(np.float64))


def _draw_convex_hull(ax, points, color, fill_color, alpha_fill=0.18, alpha_edge=0.7):
    if len(points) < 3:
        return
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        verts = np.append(hull.vertices, hull.vertices[0])
        ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
                color=fill_color, alpha=alpha_fill, zorder=1)
        ax.plot(points[verts, 0], points[verts, 1],
                color=color, alpha=alpha_edge, linewidth=1.2, zorder=2)
    except Exception:
        pass


def _draw_confidence_ellipse(ax, points, color, n_std=1.5, alpha=0.12):
    if len(points) < 3:
        return
    cov = np.cov(points.T)
    mean = points.mean(0)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.abs(vals))
    ell = Ellipse(xy=mean, width=w, height=h, angle=angle,
                  facecolor=color, alpha=alpha, edgecolor=color,
                  linewidth=1.5, linestyle="--", zorder=2)
    ax.add_patch(ell)


def plot_stage(
    ax: plt.Axes,
    xy: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    title: str,
    n_way: int,
    k_shot: int,
    n_query: int,
    show_hull: bool = True,
    show_ellipse: bool = True,
    silhouette: float = None,
):
    n_support = n_way * k_shot

    ax.set_facecolor("white")
    ax.grid(False)

    for i, cls in enumerate(class_names):
        color = _COLORS[i % len(_COLORS)]

        qry_idx = np.where(labels == i)[0]
        qry_pts = xy[n_support + qry_idx]
        sup_idx = np.arange(n_support)[np.arange(n_support) // k_shot == i]
        sup_pts = xy[sup_idx]

        ax.scatter(qry_pts[:, 0], qry_pts[:, 1],
                   c=color, marker="o", s=40, alpha=0.8,
                   edgecolors="none", zorder=3,
                   label=cls.replace("_", " "))
        ax.scatter(sup_pts[:, 0], sup_pts[:, 1],
                   c=color, marker="*", s=220,
                   edgecolors="white", linewidths=0.5, zorder=4)

    ax.legend(fontsize=8, loc="best", frameon=True,
              framealpha=0.9, edgecolor="black", borderpad=0.6)

    sil_str = f"  (Sil={silhouette:.3f})" if silhouette is not None else ""
    ax.set_title(title + sil_str, fontsize=11, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="3-stage t-SNE clustering for 5-way 1-shot FSAR."
    )
    p.add_argument("--dataset", type=str, default="hmdb51",
                   choices=list(_DATASET_CFG.keys()),
                   help="Dataset to visualise")
    p.add_argument("--features_root", type=str, default=None,
                   help="Path to *_mean.npy feature files")
    p.add_argument("--data_root", type=str, default=None,
                   help="Path to raw videos (needed for Stage 1 raw-pixel plot)")
    p.add_argument("--splits_root", type=str, default=None,
                   help="Path to splits/ subfolder. None = auto-detect.")
    p.add_argument("--ckpt_path", type=str, default=None,
                   help="Path to best_model.pth (needed for Stage 3). "
                        "If None, Stage 3 is skipped.")
    p.add_argument("--split", type=str, default="test",
                   choices=["base", "val", "test"])
    p.add_argument("--n_way",   type=int, default=5)
    p.add_argument("--k_shot",  type=int, default=1)
    p.add_argument("--n_query", type=int, default=15)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--classes", type=str, default=None,
                   help="逗号分隔的指定类名，如 'fencing,kick_ball,pushup,sit,smoke'。"
                        "指定后跳过随机采样，直接使用这些类。")
    p.add_argument("--best_of", type=int, default=1,
                   help="尝试 N 个随机 seed，选 Stage 3 silhouette 最高的 episode 绘图。"
                        "需要 --ckpt_path。默认 1（不搜索）。")
    p.add_argument("--skip_raw", action="store_true",
                   help="Skip Stage 1 (raw pixel features). "
                        "Useful when raw videos are not available.")
    p.add_argument("--no_hull",    action="store_true", help="不画凸包")
    p.add_argument("--no_ellipse", action="store_true", help="不画置信椭圆")
    # Model arch (must match checkpoint)
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--n_phases",    type=int,   default=3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--distance",    type=str,   default="cosine")
    p.add_argument("--dropout",     type=float, default=0.1)
    # Output
    p.add_argument("--output_dir", type=str, default="results/cluster_vis")
    p.add_argument("--device",     type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    _scripts_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = _DATASET_CFG[args.dataset]

    # ── Resolve paths ─────────────────────────────────────────────────────────
    splits_root = args.splits_root or os.path.normpath(
        os.path.join(_scripts_dir, cfg["splits_root_rel"])
    )
    features_root = args.features_root or (
        f"results/{args.dataset.replace('-', '_')}/extract_{args.dataset.replace('-', '_')}/feats/combined"
    )
    data_root = args.data_root

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}  split={args.split}  "
          f"{args.n_way}-way {args.k_shot}-shot  Q={args.n_query}")

    # ── Load class list ───────────────────────────────────────────────────────
    split_file = cfg["split_file_map"][args.split]
    line_format = cfg.get("line_format", "class_name")
    all_classes = _load_classes(splits_root, split_file, line_format)
    print(f"Classes in split: {len(all_classes)}")

    # ── Override classes if --classes specified ───────────────────────────────
    if args.classes:
        forced_classes = [c.strip() for c in args.classes.split(",")]
        missing = [c for c in forced_classes if c not in all_classes]
        if missing:
            print(f"[warn] Classes not found in split: {missing}")
        classes = forced_classes
        print(f"Using specified classes: {classes}")
    else:
        classes = all_classes

    show_hull    = not args.no_hull
    show_ellipse = not args.no_ellipse

    # ── Load model once (reused across best_of trials) ───────────────────────
    model = None
    if args.ckpt_path is not None:
        print("Loading model …")
        _ep0 = sample_episode(
            features_root=features_root, classes=classes,
            n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query,
            feat_suffix=cfg["feat_suffix"], rng=random.Random(args.seed),
        )
        D = _ep0["support_feats"].shape[-1]
        ckpt  = torch.load(args.ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        is_spatio = any("encoder.spatial_pe" in k for k in state.keys())
        if is_spatio:
            n_patches = ckpt.get("n_patches", 9)
            model = SpatioTemporalTRXNet(
                input_dim=D, hidden_dim=args.hidden_dim,
                n_phases=args.n_phases, n_patches=n_patches,
                max_patches=max(64, n_patches), dropout=args.dropout,
            ).to(device)
            print(f"  [Model] SpatioTemporalTRXNet  n_patches={n_patches}")
        else:
            model = PhaseAwareProtoNet(
                input_dim=D, hidden_dim=args.hidden_dim,
                n_phases=args.n_phases, temperature=args.temperature,
                distance=args.distance, dropout=args.dropout,
            ).to(device)
            print(f"  [Model] PhaseAwareProtoNet")
        model.load_state_dict(state)
        print(f"  Loaded checkpoint (epoch={ckpt.get('epoch', '?')})")

    # ── best_of search: pick episode with highest Stage-3 silhouette ──────────
    n_trials   = args.best_of if (args.best_of > 1 and model is not None) else 1
    best_episode, best_sil3, best_seed = None, -1.0, args.seed

    for trial_i in range(n_trials):
        trial_seed = args.seed + trial_i
        ep = sample_episode(
            features_root=features_root, classes=classes,
            n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query,
            feat_suffix=cfg["feat_suffix"], rng=random.Random(trial_seed),
        )
        if model is not None and n_trials > 1:
            sup_f = ep["support_feats"]
            qry_f = ep["query_feats"]
            qry_l = ep["query_labels"]
            sup_l = np.repeat(np.arange(args.n_way), args.k_shot)
            all_l = np.concatenate([sup_l, qry_l])
            enc_q = get_model_embeddings(model, sup_f, qry_f, args.n_way, args.k_shot, device)
            enc_s = get_model_embeddings(model, sup_f, sup_f, args.n_way, args.k_shot, device)
            emb   = np.concatenate([enc_s, enc_q], axis=0)
            sil   = silhouette_score(emb, all_l) if len(np.unique(all_l)) > 1 else -1.0
            print(f"  trial {trial_i+1}/{n_trials}  seed={trial_seed}  Stage3 sil={sil:.4f}")
            if sil > best_sil3:
                best_sil3, best_episode, best_seed = sil, ep, trial_seed
        else:
            best_episode, best_seed = ep, trial_seed
            break

    if n_trials > 1:
        print(f"Best seed={best_seed}  sil={best_sil3:.4f}")

    episode        = best_episode
    chosen_classes = episode["classes"]
    print(f"  Classes: {chosen_classes}")

    # ── Build label arrays ────────────────────────────────────────────────────
    sup_feats  = episode["support_feats"]   # (N*K, D)
    qry_feats  = episode["query_feats"]     # (N*Q, D)
    qry_labels = episode["query_labels"]    # (N*Q,)
    sup_labels = np.repeat(np.arange(args.n_way), args.k_shot)
    all_labels = np.concatenate([sup_labels, qry_labels])

    # ── Stage 2: SVD features ─────────────────────────────────────────────────
    all_feats_stage2 = np.concatenate([sup_feats, qry_feats], axis=0)
    print(f"Stage 2 features: {all_feats_stage2.shape}")
    xy_stage2 = run_tsne(all_feats_stage2, seed=best_seed)
    sil2 = silhouette_score(all_feats_stage2, all_labels) if len(np.unique(all_labels)) > 1 else None
    if sil2 is not None:
        print(f"  Stage 2 silhouette: {sil2:.4f}")

    # ── Stage 1: Raw pixels ───────────────────────────────────────────────────
    xy_stage1, sil1 = None, None
    if not args.skip_raw and data_root is not None:
        print("Stage 1: loading raw video frames …")
        raw_feats = []
        all_paths = (
            [p for sp in episode["support_paths"] for p in sp]
            + [p for qp in episode["query_paths"] for p in qp]
        )
        for feat_path in all_paths:
            stem = os.path.basename(feat_path)
            for suf in (cfg["feat_suffix"], "_mean.npy", "_grid.pt"):
                if stem.endswith(suf):
                    stem = stem[:-len(suf)]
                    break
            if "__" in stem:
                cls_part, vid_stem = stem.split("__", 1)
            else:
                cls_part, vid_stem = None, stem
            vpath = _find_video(data_root, cls_part, vid_stem, cfg["video_exts"])
            if vpath:
                raw_feats.append(_video_to_raw_feat(vpath))
            else:
                print(f"  [warn] Video not found for {stem}, using zeros")
                raw_feats.append(np.zeros(16 * 64 * 64 * 3, dtype=np.float32))
        raw_feats = np.stack(raw_feats)
        print(f"Stage 1 raw features: {raw_feats.shape}")
        xy_stage1 = run_tsne(raw_feats, seed=best_seed)
        sil1 = silhouette_score(raw_feats, all_labels) if len(np.unique(all_labels)) > 1 else None
    elif args.skip_raw:
        print("Stage 1 skipped (--skip_raw).")
    else:
        print("Stage 1 skipped (--data_root not provided).")

    # ── Stage 3: Model embeddings ─────────────────────────────────────────────
    xy_stage3, sil3 = None, None
    if model is not None:
        enc_qry = get_model_embeddings(model, sup_feats, qry_feats, args.n_way, args.k_shot, device)
        enc_sup = get_model_embeddings(model, sup_feats, sup_feats, args.n_way, args.k_shot, device)
        all_feats_stage3 = np.concatenate([enc_sup, enc_qry], axis=0)
        print(f"Stage 3 model embeddings: {all_feats_stage3.shape}")
        xy_stage3 = run_tsne(all_feats_stage3, seed=best_seed)
        sil3 = silhouette_score(all_feats_stage3, all_labels) if len(np.unique(all_labels)) > 1 else None
        if sil3 is not None:
            print(f"  Stage 3 silhouette: {sil3:.4f}")
    else:
        print("Stage 3 skipped (--ckpt_path not provided).")

    # ── Plot ──────────────────────────────────────────────────────────────────
    stages = []
    if xy_stage1 is not None:
        stages.append((xy_stage1, "Stage 1: Raw Pixels\n(before SVD extraction)", sil1))
    stages.append((xy_stage2, "Stage 2: SVD Features\n(after extraction)", sil2))
    if xy_stage3 is not None:
        stages.append((xy_stage3, "Stage 3: Model Embeddings\n(after PhaseAwareProtoNet)", sil3))

    n_stages = len(stages)
    fig, axes = plt.subplots(1, n_stages, figsize=(7 * n_stages, 7))
    if n_stages == 1:
        axes = [axes]

    for ax, (xy, title, sil) in zip(axes, stages):
        plot_stage(
            ax=ax, xy=xy, labels=qry_labels, class_names=chosen_classes,
            title=title, n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query,
            show_hull=show_hull, show_ellipse=show_ellipse, silhouette=sil,
        )

    fig.suptitle(
        f"{args.dataset}  {args.n_way}-way {args.k_shot}-shot  "
        f"(★ = support, ● = query)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_combined = os.path.join(args.output_dir, f"{args.dataset}_{args.split}_clusters.png")
    fig.savefig(out_combined, dpi=200, bbox_inches="tight")
    print(f"\nSaved combined plot: {out_combined}")

    # Individual stage PNGs
    for xy, title, sil in stages:
        fig_s, ax_s = plt.subplots(figsize=(7, 7))
        plot_stage(ax_s, xy, qry_labels, chosen_classes,
                   title, args.n_way, args.k_shot, args.n_query,
                   show_hull=show_hull, show_ellipse=show_ellipse, silhouette=sil)
        fig_s.tight_layout()
        stage_name = title.split(":")[0].replace(" ", "_").lower()
        out_s = os.path.join(args.output_dir,
                             f"{args.dataset}_{args.split}_{stage_name}.png")
        fig_s.savefig(out_s, dpi=200, bbox_inches="tight")
        plt.close(fig_s)
        print(f"  {stage_name}: {out_s}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
