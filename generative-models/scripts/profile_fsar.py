# -*- coding: utf-8 -*-
"""
profile_fsar.py - FLOPs, params, latency, memory profiling script

用法(在 generative-models/ 目录下运行)：

    # SpatioTemporalTRXNet(grid 特征, T=8, P=9, D=1920)
    python scripts/profile_fsar.py \
        --ckpt_path results/fsar_ucf101/best_model.pth \
        --input_dim 1920 --n_patches 9 --T 8 \
        --hidden_dim 256 --n_phases 3 \
        --n_way 5 --k_shot 1 --n_query 15 \
        --device cuda

    # PhaseAwareProtoNet(mean 特征, T=16, D=1920)
    python scripts/profile_fsar.py \
        --ckpt_path results/fsar_hmdb51/best_model.pth \
        --input_dim 1920 --T 16 \
        --hidden_dim 256 --n_phases 3 \
        --n_way 5 --k_shot 1 --n_query 15 \
        --device cuda

输出指标：
    - 模型Params(M)
    - FLOPs per episode(GFLOPs)—— 使用 thop / fvcore(自动选择)
    - GPU Latency(ms/episode, warm-up 后取均值)
    - GPU Peak GPU mem(MB)
    - 时间复杂度 / 空间复杂度理论分析(打印到 stdout)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import numpy as np

from models.FSARModel import PhaseAwareProtoNet, SpatioTemporalTRXNet


# ---------------------------------------------------------------------------
# Complexity analysis (theoretical, printed as text)
# ---------------------------------------------------------------------------

def print_complexity(args, is_spatio: bool):
    N, K, Q = args.n_way, args.k_shot, args.n_query
    T, P, D, H = args.T, args.n_patches, args.input_dim, args.hidden_dim
    K_ph = args.n_phases

    print("\n" + "=" * 60)
    print("  Theoretical Complexity Analysis")
    print("=" * 60)

    if is_spatio:
        print(f"\n  Model: SpatioTemporalTRXNet")
        print(f"  Input: support (N*K={N*K}, T={T}, P={P}, D={D})")
        print(f"        query   (Q={Q},   T={T}, P={P}, D={D})")
        print()
        print("  -- Encoder (SpatioTemporalEncoder) --")
        print(f"  1. Linear proj  D->H : O(B * T * P * D * H)")
        print(f"     = O({N*K+Q} * {T} * {P} * {D} * {H})")
        proj_ops = (N*K + Q) * T * P * D * H
        print(f"     ~ {proj_ops/1e9:.3f} GFLOPs")
        print()
        L = T * P
        print(f"  2. Self-Attention (L=T*P={L}) : O(B * L^2 * H)")
        attn_ops = (N*K + Q) * L * L * H
        print(f"     = O({N*K+Q} * {L}^2 * {H}) ~ {attn_ops/1e9:.3f} GFLOPs")
        print()
        print(f"  3. DP segmentation : O(B * T^2 * K_ph) = O({N*K+Q} * {T}^2 * {K_ph})")
        print(f"     (CPU numpy, not counted in GPU FLOPs)")
        print()
        print("  -- SpatioProtoHead --")
        print(f"  4. Cosine sim (K_ph={K_ph} phases) : O(Q * N * H * K_ph)")
        head_ops = Q * N * H * K_ph
        print(f"     ~ {head_ops/1e6:.2f} MFLOPs (negligible)")
        total = proj_ops + attn_ops
        print(f"\n  Total GPU FLOPs ~ {total/1e9:.3f} GFLOPs / episode")
        print()
        print("  Space complexity:")
        print(f"  - Peak activation : O(B * T * P * H) = O({N*K+Q} * {T} * {P} * {H})")
        mem_mb = (N*K + Q) * T * P * H * 4 / 1e6
        print(f"    ~ {mem_mb:.1f} MB (fp32)")
        print(f"  - Params: mainly from proj Linear({D}->{H}) + Transformer")
        param_proj = D * H + H  # Linear weight + bias
        param_attn = 4 * H * H  # Q/K/V/O projections
        param_ff   = 2 * H * (H * 2)
        total_param = param_proj + param_attn + param_ff
        print(f"    proj~{param_proj/1e3:.0f}K  attn~{param_attn/1e3:.0f}K  ff~{param_ff/1e3:.0f}K")

    else:
        print(f"\n  Model: PhaseAwareProtoNet (legacy)")
        print(f"  Input: support (N*K={N*K}, T={T}, D={D})")
        print(f"        query   (Q={Q},   T={T}, D={D})")
        print()
        print("  -- PhaseAwareEncoder --")
        print(f"  1. Linear proj  D->H : O(B * T * D * H)")
        proj_ops = (N*K + Q) * T * D * H
        print(f"     ~ {proj_ops/1e9:.3f} GFLOPs")
        print()
        print(f"  2. Self-Attention (L=T={T}) : O(B * T^2 * H)")
        attn_ops = (N*K + Q) * T * T * H
        print(f"     ~ {attn_ops/1e9:.4f} GFLOPs")
        print()
        print(f"  3. DP segmentation : O(B * T^2 * K_ph) — CPU only")
        print()
        print("  -- Prototype matching --")
        print(f"  4. Cosine dist : O(Q * N * H * K_ph) ~ negligible")
        total = proj_ops + attn_ops
        print(f"\n  Total GPU FLOPs ~ {total/1e9:.3f} GFLOPs / episode")
        print()
        print("  Space complexity:")
        mem_mb = (N*K + Q) * T * H * 4 / 1e6
        print(f"  - Peak activation : O(B * T * H) ~ {mem_mb:.2f} MB (fp32)")

    print("=" * 60)


# ---------------------------------------------------------------------------
# FLOPs measurement
# ---------------------------------------------------------------------------

def measure_flops(model, sup_f, sup_m, qry_f, qry_m, n_way, k_shot, device):
    """Try thop first, fall back to fvcore."""
    try:
        from thop import profile as thop_profile
        # thop needs a single forward call with positional args — wrap it
        class _Wrapper(torch.nn.Module):
            def __init__(self, m, nw, ks):
                super().__init__()
                self.m = m
                self.nw = nw
                self.ks = ks
            def forward(self, sf, sm, qf, qm):
                return self.m(sf, sm, qf, qm, self.nw, self.ks)

        wrapper = _Wrapper(model, n_way, k_shot)
        macs, params = thop_profile(
            wrapper,
            inputs=(sup_f, sup_m, qry_f, qry_m),
            verbose=False,
        )
        return macs * 2, params, "thop"   # MACs -> FLOPs
    except Exception:
        pass

    try:
        from fvcore.nn import FlopCountAnalysis
        class _Wrapper2(torch.nn.Module):
            def __init__(self, m, nw, ks):
                super().__init__()
                self.m = m
                self.nw = nw
                self.ks = ks
            def forward(self, sf, sm, qf, qm):
                return self.m(sf, sm, qf, qm, self.nw, self.ks)
        wrapper2 = _Wrapper2(model, n_way, k_shot)
        flops = FlopCountAnalysis(wrapper2, (sup_f, sup_m, qry_f, qry_m))
        flops.unsupported_ops_warnings(False)
        return flops.total(), sum(p.numel() for p in model.parameters()), "fvcore"
    except Exception:
        return None, sum(p.numel() for p in model.parameters()), "unavailable"


# ---------------------------------------------------------------------------
# Latency & memory benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_latency(model, sup_f, sup_m, qry_f, qry_m, n_way, k_shot,
                       device, n_warmup=10, n_runs=100):
    model.eval()
    # Warm-up
    for _ in range(n_warmup):
        _ = model(sup_f, sup_m, qry_f, qry_m, n_way, k_shot)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        start_evt = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
        end_evt   = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
        for i in range(n_runs):
            start_evt[i].record()
            _ = model(sup_f, sup_m, qry_f, qry_m, n_way, k_shot)
            end_evt[i].record()
        torch.cuda.synchronize()
        times_ms = [s.elapsed_time(e) for s, e in zip(start_evt, end_evt)]
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        times_ms = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(sup_f, sup_m, qry_f, qry_m, n_way, k_shot)
            times_ms.append((time.perf_counter() - t0) * 1000)
        peak_mem_mb = 0.0

    return np.mean(times_ms), np.std(times_ms), peak_mem_mb


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FSAR model profiling: FLOPs, latency, memory")
    p.add_argument("--ckpt_path",  type=str, default=None,
                   help="Path to best_model.pth. If None, uses random weights.")
    p.add_argument("--input_dim",  type=int, default=1920,
                   help="Feature dim D (1920 for grid_3x3 UCF101/HMDB51)")
    p.add_argument("--n_patches",  type=int, default=9,
                   help="Spatial patches P. Set to 0 to use PhaseAwareProtoNet.")
    p.add_argument("--T",          type=int, default=8,  help="Temporal frames")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_phases",   type=int, default=3)
    p.add_argument("--n_way",      type=int, default=5)
    p.add_argument("--k_shot",     type=int, default=1)
    p.add_argument("--n_query",    type=int, default=15)
    p.add_argument("--head",       type=str, default="proto", choices=["proto", "trx"])
    p.add_argument("--n_warmup",   type=int, default=10)
    p.add_argument("--n_runs",     type=int, default=100)
    p.add_argument("--device",     type=str, default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    N, K, Q = args.n_way, args.k_shot, args.n_query
    T, P, D = args.T, args.n_patches, args.input_dim
    H = args.hidden_dim
    is_spatio = P > 0

    # -- Build model ----------------------------------------------------------
    if is_spatio:
        model = SpatioTemporalTRXNet(
            input_dim   = D,
            hidden_dim  = H,
            n_phases    = args.n_phases,
            n_heads     = 4,
            max_patches = max(64, P),
            n_patches   = P,
            dropout     = 0.0,   # disable dropout for profiling
            head_type   = args.head,
        ).to(device)
        print(f"Model: SpatioTemporalTRXNet  P={P}  H={H}  head={args.head}")
    else:
        model = PhaseAwareProtoNet(
            input_dim   = D,
            hidden_dim  = H,
            n_phases    = args.n_phases,
            dropout     = 0.0,
        ).to(device)
        print(f"Model: PhaseAwareProtoNet  H={H}")

    if args.ckpt_path is not None and os.path.exists(args.ckpt_path):
        ckpt  = torch.load(args.ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        # auto-detect input_dim from checkpoint
        proj_w = state.get("encoder.proj.0.weight")
        if proj_w is not None:
            ckpt_D = proj_w.shape[1]
            if ckpt_D != D:
                print(f"  [auto] input_dim: {D} -> {ckpt_D} (from checkpoint)")
                D = ckpt_D
                args.input_dim = D

        # auto-detect model type: SpatioTemporalTRXNet uses "encoder.transformer"
        # PhaseAwareProtoNet uses "encoder.temporal_transformer" + "temperature"
        has_spatial_pe = "encoder.spatial_pe" in state
        has_temperature = "temperature" in state
        if has_temperature and not has_spatial_pe:
            # legacy PhaseAwareProtoNet checkpoint
            is_spatio = False
            P = 0
            args.n_patches = 0
            print("  [auto] detected PhaseAwareProtoNet checkpoint")
        else:
            is_spatio = True
            print("  [auto] detected SpatioTemporalTRXNet checkpoint")

        # rebuild model with correct dims
        if is_spatio:
            model = SpatioTemporalTRXNet(
                input_dim=D, hidden_dim=H, n_phases=args.n_phases,
                n_heads=4, max_patches=max(64, P), n_patches=P,
                dropout=0.0, head_type=args.head,
            ).to(device)
        else:
            model = PhaseAwareProtoNet(
                input_dim=D, hidden_dim=H, n_phases=args.n_phases, dropout=0.0,
            ).to(device)
        model.eval()

        # rebuild dummy tensors with correct dims
        if is_spatio:
            sup_f = torch.randn(N * K, T, P, D, device=device)
            qry_f = torch.randn(Q,     T, P, D, device=device)
        else:
            sup_f = torch.randn(N * K, T, D, device=device)
            qry_f = torch.randn(Q,     T, D, device=device)
        sup_m = torch.ones(N * K, T, device=device)
        qry_m = torch.ones(Q,     T, device=device)
        print(f"Dummy input: sup={tuple(sup_f.shape)}  qry={tuple(qry_f.shape)}")

        model.load_state_dict(state)
        print(f"Loaded: {args.ckpt_path}  (epoch={ckpt.get('epoch','?')})")
    else:
        print("Using random weights (no checkpoint provided or found)")

    model.eval()

    # -- Build dummy episode (use auto-detected D) ----------------------------─
    if is_spatio:
        sup_f = torch.randn(N * K, T, P, D, device=device)
        qry_f = torch.randn(Q,     T, P, D, device=device)
    else:
        sup_f = torch.randn(N * K, T, D, device=device)
        qry_f = torch.randn(Q,     T, D, device=device)
    sup_m = torch.ones(N * K, T, device=device)
    qry_m = torch.ones(Q,     T, device=device)
    print(f"Dummy input: sup={tuple(sup_f.shape)}  qry={tuple(qry_f.shape)}")

    # -- Parameter count ------------------------------------------------------─
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters : {total_params:,}  ({total_params/1e6:.3f} M)")
    print(f"Trainable  : {trainable:,}  ({trainable/1e6:.3f} M)")

    # -- FLOPs ----------------------------------------------------------------─
    print("\nMeasuring FLOPs ...")
    flops, _, flop_backend = measure_flops(
        model, sup_f, sup_m, qry_f, qry_m, N, K, device
    )
    if flops is not None:
        print(f"FLOPs/episode : {flops/1e9:.4f} GFLOPs  (backend={flop_backend})")
    else:
        print("FLOPs: unavailable (pip install thop or fvcore)")
        print("  pip install thop   OR   pip install fvcore")

    # -- Latency & memory ----------------------------------------------------─
    print(f"\nBenchmarking latency ({args.n_warmup} warm-up + {args.n_runs} runs) ...")
    mean_ms, std_ms, peak_mb = benchmark_latency(
        model, sup_f, sup_m, qry_f, qry_m, N, K, device,
        n_warmup=args.n_warmup, n_runs=args.n_runs,
    )

    # -- Summary --------------------------------------------------------------─
    print()
    print("=" * 60)
    print("  Engineering Metrics Summary")
    print("=" * 60)
    print(f"  Config  : {N}-way {K}-shot  Q={Q}  T={T}  P={P}  D={D}")
    print(f"  Params      : {total_params/1e6:.3f} M")
    if flops is not None:
        print(f"  FLOPs       : {flops/1e9:.4f} GFLOPs/episode")
    print(f"  Latency    : {mean_ms:.2f} +/- {std_ms:.2f} ms/episode")
    print(f"  Throughput      : {1000/mean_ms:.1f} episodes/s")
    if device.type == "cuda":
        print(f"  Peak GPU mem    : {peak_mb:.1f} MB")
    print("=" * 60)

    # -- Theoretical complexity ------------------------------------------------
    print_complexity(args, is_spatio)


if __name__ == "__main__":
    main()
