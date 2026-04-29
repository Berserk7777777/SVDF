"""
FSARModel.py – Diffusion-based Few-Shot Action Recognition

Two model families:

1. SpatioTemporalTRXNet  (primary, for new [T,P,D] features)
   • Accepts (N*K, T, P, D) features extracted by the updated
     generate_svd_maps.py (AdaptiveAvgPool2d grid, P = g² patches)
   • SpatioTemporalEncoder: proj → temporal PE + spatial PE →
     spatio-temporal self-attention → DP phase segmentation →
     phase-spatial mean pool → (B, K, P, H)
   • SpatioPhaseTRX: per-phase cross-attention between query patches
     and support patches → cosine similarity → sum over phases → logits

2. PhaseAwareProtoNet  (legacy, for old [T,D] features / old checkpoints)
   • Kept unchanged for backward compatibility

Temporal clustering — DP Optimal Temporal Segmentation
-------------------------------------------------------
Partition T ordered frames into K contiguous segments minimising
total within-cluster sum of squares (WCSS).  Solved exactly in
O(T²·K) via dynamic programming.  See OptimalTemporalSegmenter.

Spatio-Phase TRX cross-attention
---------------------------------
For each of K phases and each (query, support) pair:
    query patches (P, H) attend to support patches (P, H) as K/V
    → attended output mean vs query mean → cosine similarity
    → accumulate over K phases → (Q, N) logits
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DP Optimal Temporal Segmenter  (shared by both model families)
# ---------------------------------------------------------------------------

class OptimalTemporalSegmenter:
    """
    Exact optimal temporal segmentation via dynamic programming.

    Solves: partition T ordered frames into n_phases contiguous segments
    to minimise total within-segment sum of squares (WCSS).

    Complexity: O(T² · n_phases)  — negligible for T≤64, K≤5.

    Gradients flow through the returned segment tensors (slice of the
    original PyTorch tensor); only the split-point computation is in
    numpy (non-differentiable, treated as a hard assignment).
    """

    @staticmethod
    def _dp_split_points(feats_np: np.ndarray, n_phases: int) -> List[int]:
        """
        Args:
            feats_np : (T, D) float32 numpy array
            n_phases : K ≥ 2
        Returns:
            boundaries : list of length n_phases+1 with boundaries[0]=0,
                         boundaries[-1]=T, strictly increasing.
        """
        T, D = feats_np.shape

        # ── Prefix sums (O(T)) ──────────────────────────────────────────
        psum = np.zeros((T + 1, D), dtype=np.float64)   # Σ x[i]
        psq  = np.zeros(T + 1, dtype=np.float64)        # Σ ‖x[i]‖²
        for t in range(T):
            psum[t + 1] = psum[t] + feats_np[t]
            psq[t + 1]  = psq[t]  + float(np.dot(feats_np[t], feats_np[t]))

        def seg_cost(s: int, e: int) -> float:
            """WCSS of segment [s, e)  (s inclusive, e exclusive)."""
            n = e - s
            if n <= 0:
                return 0.0
            S  = psum[e] - psum[s]                    # (D,)
            SS = psq[e]  - psq[s]                     # scalar
            return float(SS - np.dot(S, S) / n)

        # ── DP ──────────────────────────────────────────────────────────
        INF   = float("inf")
        dp    = [[INF] * (T + 1) for _ in range(n_phases + 1)]
        split = [[0]   * (T + 1) for _ in range(n_phases + 1)]
        dp[0][0] = 0.0

        for k in range(1, n_phases + 1):
            for t in range(k, T + 1):
                for s in range(k - 1, t):
                    c = dp[k - 1][s] + seg_cost(s, t)
                    if c < dp[k][t]:
                        dp[k][t] = c
                        split[k][t] = s

        # ── Backtrack ───────────────────────────────────────────────────
        boundaries = [T]
        t = T
        for k in range(n_phases, 0, -1):
            s = split[k][t]
            boundaries.append(s)
            t = s
        boundaries.reverse()   # [0, s1, …, s_{K-1}, T]
        return boundaries

    @staticmethod
    def segment(features: torch.Tensor, n_phases: int = 3) -> List[torch.Tensor]:
        """
        Args:
            features : (T, D) float tensor (any device)
            n_phases : number of phases to produce (≥ 1)
        Returns:
            List of n_phases tensors each shaped (T_k, D), covering
            the full sequence without overlap.  Gradients are preserved.
        """
        T = features.shape[0]

        # Edge case: not enough frames to form distinct segments
        if T <= n_phases:
            segs = [features[i: i + 1] for i in range(min(T, n_phases))]
            while len(segs) < n_phases:
                segs.append(segs[-1])
            return segs

        # Run DP on a float32 CPU numpy copy (hard assignment only)
        feats_np = features.detach().cpu().float().numpy()  # (T, D)
        boundaries = OptimalTemporalSegmenter._dp_split_points(feats_np, n_phases)

        # Slice original PyTorch tensor so gradients flow through pooling
        segs: List[torch.Tensor] = []
        for i in range(n_phases):
            s, e = boundaries[i], boundaries[i + 1]
            if s >= e:           # degenerate: reuse previous segment
                segs.append(segs[-1] if segs else features[0:1])
            else:
                segs.append(features[s:e])

        return segs


# ---------------------------------------------------------------------------
# Positional Encoding  (shared)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(1, max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Layer  (shared)
# ---------------------------------------------------------------------------

class TemporalTransformerLayer(nn.Module):
    """
    Single Transformer encoder layer.
    Input / Output : (B, L, hidden_dim)  where L can be T, P, or T*P.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn  = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        ff_dim = hidden_dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(h))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


# ===========================================================================
#  PRIMARY MODEL: SpatioTemporalTRXNet
#  Input features shape: (N*K, T, P, D)
# ===========================================================================

class SpatioTemporalEncoder(nn.Module):
    """
    Encodes (B, T, P, D) → (B, n_phases, P, H).

    Pipeline:
        1. Linear D→H applied per patch  → (B, T, P, H)
        2. Temporal PE added over T      → (B, T, P, H)
        3. Spatial PE (learnable) over P → (B, T, P, H)
        4. Flatten T×P → self-attn → reshape back → (B, T, P, H)
        5. T-avg → (B, T, H) used only for DP boundary decisions
        6. Slice phases along T, mean-pool time, keep P dim → (B, K, P, H)

    Dimension trace (B=5, T=8, P=9, D=1792, H=256, K=3):
        input      : (5, 8, 9, 1792)
        proj       : (5, 8, 9, 256)
        +tem_pe    : (5, 8, 9, 256)   broadcast over P
        +spa_pe    : (5, 8, 9, 256)   broadcast over T
        flatten    : (5, 72, 256)
        transformer: (5, 72, 256)
        reshape    : (5, 8, 9, 256)
        DP on avg  : boundaries per sample
        phase pool : (5, 3, 9, 256)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_phases: int = 3,
        n_heads: int = 4,
        max_patches: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_phases  = n_phases
        self.hidden_dim = hidden_dim

        # Per-patch linear projection
        # Dropout 在 GELU 之后：投影输出的随机神经元屏蔽是最有效的正则化位置，
        # 强制网络不依赖任何单一特征维度来区分类别。
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal PE (sinusoidal, over T)
        self.temporal_pe = PositionalEncoding(hidden_dim, dropout=0.0, max_len=512)

        # Spatial PE (learnable, over P)
        self.spatial_pe = nn.Parameter(torch.zeros(1, 1, max_patches, hidden_dim))
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)

        # Spatio-temporal self-attention (over flattened T*P tokens)
        self.transformer = TemporalTransformerLayer(
            hidden_dim, n_heads=n_heads, ff_mult=2, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            features : (B, T, P, D)
            mask     : (B, T) float, 1=valid frame, 0=padding  [optional]
        Returns:
            phase_feats : (B, n_phases, P, H)
        """
        B, T, P, D = features.shape
        H = self.hidden_dim

        # 1. Project D → H per patch
        x = self.proj(features)                          # (B, T, P, H)

        # 2. Temporal PE: pe over T, broadcast to all patches
        x_t = x.reshape(B, T, P * H)                    # view for PE on T dim
        # apply PE on (B, T, H) view: reshape to (B*P, T, H) for PositionalEncoding
        x_tp = x.permute(0, 2, 1, 3).reshape(B * P, T, H)   # (B*P, T, H)
        x_tp = self.temporal_pe(x_tp)                         # (B*P, T, H)
        x = x_tp.reshape(B, P, T, H).permute(0, 2, 1, 3)     # (B, T, P, H)

        # 3. Spatial PE: learned embedding per patch, broadcast over T & B
        x = x + self.spatial_pe[:, :, :P, :]            # (B, T, P, H)
        x = self.dropout(x)

        # 4. Spatio-temporal self-attention over T*P tokens
        x_flat = x.reshape(B, T * P, H)                 # (B, T*P, H)
        x_flat = self.transformer(x_flat)                # (B, T*P, H)
        x = x_flat.reshape(B, T, P, H)                  # (B, T, P, H)

        # 5 & 6. 向量化均匀相位池化（全 GPU，无 CPU/numpy 同步）
        #
        # 原 DP 分割方案每个样本都要做 .detach().cpu().numpy() → GPU 停顿，
        # 导致 GPU 利用率只有 ~3%。均匀分割对 T=16、K=3 效果相当，
        # 且完全消除 CPU 瓶颈。
        #
        # 若 mask 存在，取批内最小有效帧长（保守做法，避免 padding 帧污染均值）。
        if mask is not None:
            valid_len = int(mask.sum(dim=1).min().item())
            valid_len = max(valid_len, self.n_phases)
        else:
            valid_len = T

        x_eff = x[:, :valid_len]  # (B, T_eff, P, H)

        # --- 替换开始：启用 DP 动态规划最优分割 ---
        phase_feats_list = []
        for b in range(x_eff.shape[0]):  # 遍历 batch 里的每个样本
            # 将空间特征在 P 维度平均，转为 (T_eff, H) 后送入 DP 计算最优边界
            x_b_avg = x_eff[b].mean(dim=1).detach().cpu().float().numpy()
            boundaries = OptimalTemporalSegmenter._dp_split_points(x_b_avg, self.n_phases)

            segs = []
            for i in range(self.n_phases):
                s, e = boundaries[i], boundaries[i + 1]
                if s >= e:  # 边界异常时复用上一个分段
                    segs.append(segs[-1] if segs else x_eff[b, 0:1])
                else:
                    segs.append(x_eff[b, s:e])  # 截取该阶段特征: (T_k, P, H)

            # 对时间维度(T_k)进行均值池化 -> 变成 K 个 (P, H)
            pooled = [seg.mean(dim=0) for seg in segs]
            phase_feats_list.append(torch.stack(pooled, dim=0))

        phase_feats = torch.stack(phase_feats_list, dim=0)  # (B, K, P, H)
        # --- 替换结束 ---                                                  # (B, K, P, H)

        return phase_feats


class SpatioPhaseTRX(nn.Module):
    """
    Cross-attention-based query-support matching per temporal phase.

    For each phase k:
        q_k : (Q,   P, H)  query  patch descriptors
        s_k : (N,   P, H)  support patch prototypes  (mean over k_shot)

        Expand to (Q*N, P, H) pairs → cross-attention (Q as query, S as K/V)
        → cosine similarity of attended output vs query → (Q, N)

    Accumulate similarities across K phases → final (Q, N) logits.

    Temperature is a learned scalar (log-parameterised for stability).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_phases: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_phases = n_phases

        # Shared cross-attention layer (reused across all phases)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm    = nn.LayerNorm(hidden_dim)
        self.out_drop = nn.Dropout(dropout)   # applied after norm to combat overfitting

        # sim_drop：在 cosine similarity 计算前对 attended mean 施加 Dropout。
        # 训练时随机屏蔽约 50% 的特征维度，迫使模型基于部分维度做出可靠的
        # 相似度判断（等价于"分类头正则化"）；eval 模式下自动关闭，不影响推理。
        self.sim_drop = nn.Dropout(dropout)

        # Learnable temperature (log-parameterised).
        # Clamped to [0.05, 5.0] after exp to prevent over-confident / flat logits.
        self.log_temp = nn.Parameter(torch.zeros(1))   # exp(0) = 1.0 init

    def forward(
        self,
        q_feats: torch.Tensor,
        s_feats: torch.Tensor,
        n_way: int,
        k_shot: int,
    ) -> torch.Tensor:
        """
        Args:
            q_feats  : (Q,   n_phases, P, H)
            s_feats  : (N*K, n_phases, P, H)
            n_way    : N
            k_shot   : K (shots per class)
        Returns:
            logits   : (Q, N)   higher = more similar

        Dimension trace (Q=75, N=5, K=1, n_phases=3, P=9, H=256):
            q_feats  : (75, 3, 9, 256)
            protos   : (5,  3, 9, 256)   after mean over k_shot
            per phase: q_k (75,9,256), s_k (5,9,256)
                       expand → (375,9,256) pairs
                       cross-attn → (375,9,256)
                       mean over P → (375,256)
                       cosine sim  → (375,) → reshape (75,5)
            logits   : (75, 5)
        """
        Q, K_ph, P, H = q_feats.shape
        NK = s_feats.shape[0]
        N  = n_way

        # Mean over k_shot to get prototypes
        protos = s_feats.view(N, k_shot, K_ph, P, H).mean(dim=1)  # (N, K, P, H)

        total_sim = q_feats.new_zeros(Q, N)

        for k in range(K_ph):
            q_k = q_feats[:, k, :, :]        # (Q, P, H)
            s_k = protos[:, k, :, :]          # (N, P, H)

            # Expand to (Q*N, P, H) pairs
            q_exp = q_k.unsqueeze(1).expand(Q, N, P, H).reshape(Q * N, P, H)
            s_exp = s_k.unsqueeze(0).expand(Q, N, P, H).reshape(Q * N, P, H)
            # contiguous needed after expand+reshape
            q_exp = q_exp.contiguous()
            s_exp = s_exp.contiguous()

            # Cross-attention: query patches attend to support patches
            attended, _ = self.cross_attn(q_exp, s_exp, s_exp, need_weights=False)
            attended = self.out_drop(self.norm(attended))  # (Q*N, P, H)  + dropout

            # Mean-pool over patches → (Q*N, H)
            # sim_drop 在 cosine_similarity 计算前施加：训练时随机屏蔽特征维度，
            # 强迫模型对部分特征缺失保持鲁棒；eval 模式自动关闭。
            att_mean = self.sim_drop(attended.mean(dim=1))   # (Q*N, H)
            q_mean   = q_exp.mean(dim=1)                     # (Q*N, H)

            # Cosine similarity between attended output and query
            sim_k = F.cosine_similarity(att_mean, q_mean, dim=-1)  # (Q*N,)
            total_sim = total_sim + sim_k.reshape(Q, N)

        # Scale by temperature; clamp prevents over-confident (→0) or flat (→∞) logits
        temp = self.log_temp.exp().clamp(min=0.05, max=5.0)
        logits = total_sim / temp                    # (Q, N)
        return logits


class SpatioProtoHead(nn.Module):
    """
    文档推荐的 metric-based 保序 prototype 匹配头。

    严格按照设计文档中的公式：
        D(q, c) = Σ_k  d(q^(k), p_c^(k))
    其中 d 为 cosine 相似度，p_c^(k) 为第 k 阶段的 support 类原型。

    与 SpatioPhaseTRX 的本质区别：
    - 无跨样本交叉注意力（不依赖 base-class 交互模式）
    - 匹配头可学习参数：~10 个（vs TRX 的 ~260K）
    - 对未见新类的泛化能力大幅提升

    额外改进（第一性原理）：
    1. 多尺度时序：per-phase sim + global sim（mean over phases）
       → 同时捕获局部动作阶段相似度 & 整体语义相似度
    2. 可学习空间 patch 重要性权重（9 个参数，类无关，可泛化）
       → 端到端学习哪些空间位置对匹配更有判断力

    Dimension trace (Q=75, N=5, K=1, K_ph=3, P=9, H=256):
        q_feats      : (75, 3, 9, 256)
        protos       : (5,  3, 9, 256)   after mean over k_shot
        q_agg        : (75, 3, 256)      after weighted patch aggregation
        s_agg        : (5,  3, 256)
        per-phase sim: (75, 5) × K_ph
        global sim   : (75, 5)           mean over phases
        logits       : (75, 5)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_phases: int = 3,
        n_patches: int = 9,
    ):
        super().__init__()
        self.n_phases = n_phases
        # 可学习空间 patch 权重（类无关 → 对新类也有效）
        self.patch_weight = nn.Parameter(torch.ones(n_patches))
        # 可学习温度（log 参数化保持数值稳定）
        self.log_temp = nn.Parameter(torch.zeros(1))

    def _weighted_pool(self, x: torch.Tensor) -> torch.Tensor:
        """(..., P, H) → (..., H)，按 softmax patch 权重加权平均"""
        w = torch.softmax(self.patch_weight, dim=0)          # (P,)
        shape = [1] * (x.ndim - 2) + [w.shape[0], 1]
        return (x * w.view(shape)).sum(dim=-2)               # (..., H)

    def forward(
        self,
        q_feats: torch.Tensor,
        s_feats: torch.Tensor,
        n_way: int,
        k_shot: int,
    ) -> torch.Tensor:
        """
        Args:
            q_feats : (Q,   K_ph, P, H)
            s_feats : (N*K, K_ph, P, H)
            n_way   : N
            k_shot  : K
        Returns:
            logits  : (Q, N)
        """
        Q, K_ph, P, H = q_feats.shape
        N = n_way

        # Support prototype: mean over k_shot
        protos = s_feats.view(N, k_shot, K_ph, P, H).mean(dim=1)   # (N, K_ph, P, H)

        # Weighted spatial aggregation: (*, P, H) → (*, H)
        q_agg = self._weighted_pool(q_feats)   # (Q, K_ph, H)
        s_agg = self._weighted_pool(protos)    # (N, K_ph, H)

        total_sim = q_feats.new_zeros(Q, N)

        # ── Scale 1: 保序 per-phase cosine similarity ────────────────────
        for k in range(K_ph):
            q_k = F.normalize(q_agg[:, k], dim=-1)   # (Q, H)
            s_k = F.normalize(s_agg[:, k], dim=-1)   # (N, H)
            total_sim = total_sim + (q_k @ s_k.t())  # (Q, N)

        # ── Scale 2: global cosine similarity（多尺度时序）────────────────
        # 将所有阶段特征平均后做一次全局匹配，捕获整体动作语义
        q_g = F.normalize(q_agg.mean(dim=1), dim=-1)  # (Q, H)
        s_g = F.normalize(s_agg.mean(dim=1), dim=-1)  # (N, H)
        total_sim = total_sim + (q_g @ s_g.t())        # (Q, N)

        temp = self.log_temp.exp().clamp(min=0.05, max=5.0)
        return total_sim / temp


class SpatioTemporalTRXNet(nn.Module):
    """
    Fine-grained few-shot action recognition via spatio-temporal
    cross-attention on DP-segmented temporal phases.

    Accepts SVD features extracted with AdaptiveAvgPool2d grid:
        support : (N*K, T, P, D)
        query   : (Q,   T, P, D)
    where P = grid_size² spatial patches, D = concat channel dim.

    Architecture:
        SpatioTemporalEncoder → (B, n_phases, P, H)
        head_type='trx'  → SpatioPhaseTRX   (cross-attention matching)
        head_type='proto' → SpatioProtoHead  (cosine prototype matching)

    Keeps the same forward() signature as PhaseAwareProtoNet for
    drop-in compatibility with train_fsar.py / test_fsar.py after
    Task-6 tensor-shape update.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_phases: int = 3,
        n_heads: int = 4,
        max_patches: int = 64,
        n_patches: int = 9,
        dropout: float = 0.1,
        head_type: str = "proto",
    ):
        super().__init__()
        self.n_phases = n_phases

        self.encoder = SpatioTemporalEncoder(
            input_dim   = input_dim,
            hidden_dim  = hidden_dim,
            n_phases    = n_phases,
            n_heads     = n_heads,
            max_patches = max_patches,
            dropout     = dropout,
        )

        if head_type == "proto":
            self.trx = SpatioProtoHead(
                hidden_dim = hidden_dim,
                n_phases   = n_phases,
                n_patches  = n_patches,
            )
        else:
            self.trx = SpatioPhaseTRX(
                hidden_dim = hidden_dim,
                n_phases   = n_phases,
                n_heads    = n_heads,
                dropout    = dropout,
            )

    def forward(
        self,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
        query_features: torch.Tensor,
        query_masks: torch.Tensor,
        n_way: int,
        k_shot: int,
    ) -> torch.Tensor:
        """
        Args:
            support_features : (N*K, T, P, D)
            support_masks    : (N*K, T)         1=valid frame, 0=pad
            query_features   : (Q,   T, P, D)
            query_masks      : (Q,   T)
            n_way            : N
            k_shot           : K
        Returns:
            logits           : (Q, N)   higher = more similar to class

        Dimension trace (N=5, K=1, Q=75, T=8, P=9, D=1792, H=256, phases=3):
            support_features  : (5,  8, 9, 1792)
            sup_phase         : (5,  3, 9, 256)   after encoder
            query_features    : (75, 8, 9, 1792)
            qry_phase         : (75, 3, 9, 256)
            TRX logits        : (75, 5)
        """
        NK = n_way * k_shot
        assert support_features.shape[0] == NK, (
            f"Expected {NK} support samples, got {support_features.shape[0]}"
        )

        # Encode support → (N*K, n_phases, P, H)
        sup_phase = self.encoder(support_features, support_masks)

        # Encode query → (Q, n_phases, P, H)
        qry_phase = self.encoder(query_features, query_masks)

        # Spatio-phase TRX matching → (Q, N)
        logits = self.trx(qry_phase, sup_phase, n_way=n_way, k_shot=k_shot)
        return logits


# ===========================================================================
#  LEGACY MODEL: PhaseAwareProtoNet
#  Input features shape: (N*K, T, D)  — kept for old checkpoints
# ===========================================================================

class PhaseAwareEncoder(nn.Module):
    """
    Pipeline:
        (B, T, input_dim)
            ↓  linear projection + LayerNorm + GELU
        (B, T, hidden_dim)
            ↓  sinusoidal positional encoding
            ↓  Temporal Transformer (multi-head self-attention)
        (B, T, hidden_dim)   ← globally-informed frame features
            ↓  DP optimal segmentation per sample → n_phases segments
            ↓  mean pooling within each segment
        (B, n_phases, hidden_dim)

    Input  : (B, T, input_dim)  + optional binary mask (B, T)
    Output : (B, n_phases, hidden_dim)
    """

    def __init__(
        self,
        input_dim: int = 10880,
        hidden_dim: int = 512,
        n_phases: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_phases = n_phases

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)

        self.temporal_transformer = TemporalTransformerLayer(
            hidden_dim, n_heads=n_heads, ff_mult=2, dropout=dropout
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = features.shape

        proj = self.proj(features)          # (B, T, H)
        proj = self.pos_enc(proj)           # (B, T, H)
        proj = self.temporal_transformer(proj)  # (B, T, H)

        phase_feats = []
        for b in range(B):
            valid_len = int(mask[b].sum().item()) if mask is not None else T
            valid_len = max(valid_len, 1)

            valid_proj = proj[b, :valid_len]                           # (T_v, H)
            segs       = OptimalTemporalSegmenter.segment(valid_proj, self.n_phases)
            pooled     = [seg.mean(0) for seg in segs]                 # K × (H,)
            phase_feats.append(torch.stack(pooled, dim=0))             # (K, H)

        return torch.stack(phase_feats, dim=0)   # (B, K, H)


class PhaseAwareProtoNet(nn.Module):
    """
    Legacy: few-shot action classifier on (N*K, T, D) features.
    Use SpatioTemporalTRXNet for new (N*K, T, P, D) features.
    """

    def __init__(
        self,
        input_dim: int = 10880,
        hidden_dim: int = 512,
        n_phases: int = 3,
        n_heads: int = 4,
        temperature: float = 0.1,
        distance: str = "cosine",
        dropout: float = 0.1,
    ):
        super().__init__()
        assert distance in ("cosine", "euclidean")
        self.encoder     = PhaseAwareEncoder(
            input_dim, hidden_dim, n_phases, n_heads, dropout
        )
        self.n_phases    = n_phases
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.distance    = distance

    @staticmethod
    def _cosine_dist(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, dim=-1).unsqueeze(1)   # (Q, 1, K, D)
        p = F.normalize(p, dim=-1).unsqueeze(0)   # (1, N, K, D)
        cos_sim  = (q * p).sum(dim=-1)            # (Q, N, K)
        return (1.0 - cos_sim).sum(dim=-1)        # (Q, N)

    @staticmethod
    def _euclidean_dist(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        q = q.unsqueeze(1)   # (Q, 1, K, D)
        p = p.unsqueeze(0)   # (1, N, K, D)
        return ((q - p) ** 2).sum(dim=-1).sum(dim=-1)   # (Q, N)

    def forward(
        self,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
        query_features: torch.Tensor,
        query_masks: torch.Tensor,
        n_way: int,
        k_shot: int,
    ) -> torch.Tensor:
        NK = n_way * k_shot
        assert support_features.shape[0] == NK

        sup_phase  = self.encoder(support_features, support_masks)
        sup_phase  = sup_phase.view(n_way, k_shot, self.n_phases, -1)
        prototypes = sup_phase.mean(dim=1)

        qry_phase = self.encoder(query_features, query_masks)

        if self.distance == "cosine":
            dist = self._cosine_dist(qry_phase, prototypes)
        else:
            dist = self._euclidean_dist(qry_phase, prototypes)

        temp   = self.temperature.abs().clamp(min=1e-4)
        return -dist / temp
