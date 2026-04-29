# =============================================================================
# ablation_experiments.ps1
#
# Two-phase orthogonal ablation experiment for SVD FSAR:
#
#   Phase 1 – Layer-group ablation (diffusion_step = 23 fixed)
#     Shallow   : layers [2,3,4]
#     Mid       : layers [6,7,8]
#     Deep      : layers [11,12,13]
#     Wide-span : layers [3,7,12]
#
#   Phase 2 – Noise-step ablation (layer group fixed to best from Phase 1)
#     Noise-23  : diffusion_step = 23
#     Noise-15  : diffusion_step = 15
#     Noise-5   : diffusion_step = 5
#
# Each experiment: Extract → Train → Test → log acc to results.csv
#
# Usage (run from generative-models/):
#   pwsh scripts/ablation_experiments.ps1
#   pwsh scripts/ablation_experiments.ps1 -BestLayerExp mid_678
# =============================================================================

param(
    [string]$BestLayerExp = "auto",   # override: set Phase-2 layer group manually
    [string]$DataSplit    = "meta_train",
    [int]   $NumEpochs   = 30,
    [int]   $BatchSize   = 4,
    [float] $LR          = 1e-4,
    [int]   $Seed        = 42,
    [string]$Device      = "cuda",
    [string]$AblationDir = "results/ablation"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helper: run a command and exit on failure
# ---------------------------------------------------------------------------
function Invoke-Step {
    param([string]$Desc, [string]$Cmd)
    Write-Host "`n[STEP] $Desc" -ForegroundColor Cyan
    Write-Host "  $Cmd" -ForegroundColor DarkGray
    Invoke-Expression $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Step failed: $Desc"
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Helper: extract best experiment name from results CSV
# ---------------------------------------------------------------------------
function Get-BestExp {
    param([string]$CsvPath)
    if (-not (Test-Path $CsvPath)) { return $null }
    $rows  = Import-Csv $CsvPath
    $best  = $rows | Sort-Object { [double]$_.test_acc } -Descending | Select-Object -First 1
    return $best.exp_name
}

# ---------------------------------------------------------------------------
# Result log
# ---------------------------------------------------------------------------
$null = New-Item -ItemType Directory -Force -Path $AblationDir
$ResultsCsv = "$AblationDir/results.csv"
if (-not (Test-Path $ResultsCsv)) {
    "exp_name,phase,layers,diffusion_step,test_acc" | Out-File -FilePath $ResultsCsv -Encoding utf8
}

# ---------------------------------------------------------------------------
# Per-experiment pipeline
# ---------------------------------------------------------------------------
function Run-Experiment {
    param(
        [string]$ExpName,
        [string]$Phase,
        [string]$LayerArg,   # e.g. "[2,3,4]"
        [int]   $DiffStep
    )

    $ExpDir   = "$AblationDir/$ExpName"
    $FeatsDir = "$ExpDir/feats"
    $TrainDir = "$ExpDir/train"
    $TestCsv  = "$ExpDir/test_results.csv"

    Write-Host "`n============================================" -ForegroundColor Yellow
    Write-Host "  Experiment : $ExpName" -ForegroundColor Yellow
    Write-Host "  Phase      : $Phase"
    Write-Host "  Layers     : $LayerArg"
    Write-Host "  Diff step  : $DiffStep"
    Write-Host "============================================" -ForegroundColor Yellow

    # ── Step 1: Feature Extraction ──────────────────────────────────────────
    if (-not (Test-Path $FeatsDir)) {
        Invoke-Step "Extract SVD features ($ExpName)" `
            "python scripts/generate_svd_maps.py ``
                --dataset ssv2_episodic ``
                --split $DataSplit ``
                --layer_idxs $LayerArg ``
                --diffusion_step $DiffStep ``
                --add_noise True ``
                --seed $Seed ``
                --output_folder $AblationDir ``
                --exp_name $ExpName ``
                --device $Device"
    } else {
        Write-Host "  [skip] Features already exist: $FeatsDir" -ForegroundColor DarkGray
    }

    # ── Step 2: Train FSAR model ────────────────────────────────────────────
    $BestCkpt = "$TrainDir/best_model.pth"
    if (-not (Test-Path $BestCkpt)) {
        Invoke-Step "Train PhaseAwareProtoNet ($ExpName)" `
            "python scripts/train_fsar.py ``
                --dry_run_feats $FeatsDir ``
                --hidden_dim 512 ``
                --n_phases 3 ``
                --num_epochs $NumEpochs ``
                --batch_size $BatchSize ``
                --lr $LR ``
                --seed $Seed ``
                --num_workers 0 ``
                --save_dir $TrainDir ``
                --device $Device"
    } else {
        Write-Host "  [skip] Checkpoint already exists: $BestCkpt" -ForegroundColor DarkGray
    }

    # ── Step 3: Evaluate ────────────────────────────────────────────────────
    Invoke-Step "Test ($ExpName)" `
        "python scripts/test_fsar.py ``
            --ckpt_path $BestCkpt ``
            --dry_run_feats $FeatsDir ``
            --dry_run_split test ``
            --batch_size $BatchSize ``
            --num_workers 0 ``
            --seed $Seed ``
            --device $Device ``
            --output_csv $TestCsv"

    # ── Parse accuracy from CSV ─────────────────────────────────────────────
    $TestAcc = "N/A"
    if (Test-Path $TestCsv) {
        $rows    = Import-Csv $TestCsv
        $accVals = $rows | ForEach-Object { [double]$_.accuracy }
        $mean    = ($accVals | Measure-Object -Average).Average
        $TestAcc = [math]::Round($mean * 100, 2)
        Write-Host "`n  >> $ExpName  test_acc = $TestAcc %" -ForegroundColor Green
    }

    # Append to global results CSV
    "$ExpName,$Phase,$LayerArg,$DiffStep,$TestAcc" | `
        Out-File -FilePath $ResultsCsv -Append -Encoding utf8

    return $TestAcc
}

# =============================================================================
# Phase 1 — Layer-group ablation  (diffusion_step = 23 fixed)
# =============================================================================
Write-Host "`n#####  PHASE 1: Layer-group ablation  #####" -ForegroundColor Magenta

$Phase1Configs = @(
    @{ Name = "shallow_234";   Layers = "[2,3,4]"    },
    @{ Name = "mid_678";       Layers = "[6,7,8]"    },
    @{ Name = "deep_111213";   Layers = "[11,12,13]" },
    @{ Name = "wide_3712";     Layers = "[3,7,12]"   }
)

foreach ($cfg in $Phase1Configs) {
    Run-Experiment `
        -ExpName    $cfg.Name `
        -Phase      "layer_ablation" `
        -LayerArg   $cfg.Layers `
        -DiffStep   23
}

# =============================================================================
# Phase 2 — Noise-step ablation  (best layer group fixed)
# =============================================================================
Write-Host "`n#####  PHASE 2: Noise-step ablation  #####" -ForegroundColor Magenta

# Determine best layer group from Phase 1
if ($BestLayerExp -eq "auto") {
    $BestLayerExp = Get-BestExp -CsvPath $ResultsCsv
    if ($null -eq $BestLayerExp) {
        Write-Warning "Could not determine best experiment; defaulting to 'mid_678'."
        $BestLayerExp = "mid_678"
    }
}

# Look up the layer string for the best experiment
$BestLayerStr = ($Phase1Configs | Where-Object { $_.Name -eq $BestLayerExp }).Layers
if (-not $BestLayerStr) {
    Write-Warning "Layer string for '$BestLayerExp' not found; using '[6,7,8]'."
    $BestLayerStr = "[6,7,8]"
}

Write-Host "  Best layer group: $BestLayerExp  ($BestLayerStr)" -ForegroundColor Green

$Phase2DiffSteps = @(23, 15, 5)

foreach ($step in $Phase2DiffSteps) {
    $expName = "${BestLayerExp}_step${step}"
    # step=23 result reuses Phase-1 features (already extracted); avoid re-extraction
    $srcFeats = if ($step -eq 23) { "$AblationDir/$BestLayerExp/feats" } `
                else              { "$AblationDir/$expName/feats" }

    if ($step -ne 23) {
        # Need to re-extract with different noise level
        Run-Experiment `
            -ExpName   $expName `
            -Phase     "noise_ablation" `
            -LayerArg  $BestLayerStr `
            -DiffStep  $step
    } else {
        # step=23 already done in Phase 1 — just log again under new exp name
        Write-Host "`n  [reuse] step=23 result from Phase 1 ($BestLayerExp)" -ForegroundColor DarkGray
        if (Test-Path $ResultsCsv) {
            $row = Import-Csv $ResultsCsv | Where-Object { $_.exp_name -eq $BestLayerExp }
            if ($row) {
                "$expName,noise_ablation,$BestLayerStr,$step,$($row.test_acc)" | `
                    Out-File -FilePath $ResultsCsv -Append -Encoding utf8
            }
        }
    }
}

# =============================================================================
# Summary
# =============================================================================
Write-Host "`n===================  ABLATION SUMMARY  ===================" -ForegroundColor Yellow
if (Test-Path $ResultsCsv) {
    Import-Csv $ResultsCsv | Format-Table -AutoSize
}
Write-Host "Full results saved to: $ResultsCsv" -ForegroundColor Green
