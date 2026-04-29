@echo off
setlocal enabledelayedexpansion

echo =========================================
echo ActionDiff Windows Setup (Relative Paths)
echo =========================================

REM ==== User config ====
set ENV_NAME=svdf
set PROJECT_ROOT=.
set GEN_ROOT=.\generative-models

REM ==== Check conda ====
where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] conda not found in PATH.
    echo Please open Anaconda Prompt and run this script there.
    pause
    exit /b 1
)

REM ==== Activate env ====
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env: %ENV_NAME%
    pause
    exit /b 1
)

echo [OK] Activated conda env: %ENV_NAME%

REM ==== Go to project root ====
cd /d %PROJECT_ROOT%
if errorlevel 1 (
    echo [ERROR] Cannot cd to project root.
    pause
    exit /b 1
)

echo [STEP] Upgrade pip tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed upgrading pip tools
    pause
    exit /b 1
)

echo [STEP] Install ActionDiff root requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed installing root requirements
    pause
    exit /b 1
)

echo [STEP] Enter generative-models...
cd /d %GEN_ROOT%
if errorlevel 1 (
    echo [ERROR] Cannot cd to .\generative-models
    pause
    exit /b 1
)

echo [STEP] Install generative-models package...
pip install .
if errorlevel 1 (
    echo [ERROR] Failed pip install .
    pause
    exit /b 1
)

echo [STEP] Install Windows-friendly dependencies...
pip install ^
black==23.7.0 ^
chardet==5.1.0 ^
einops>=0.6.1 ^
fairscale>=0.4.13 ^
fire>=0.5.0 ^
fsspec>=2023.6.0 ^
imageio[ffmpeg] ^
imageio[pyav] ^
invisible-watermark>=0.2.0 ^
kornia==0.6.9 ^
matplotlib>=3.7.2 ^
natsort>=8.4.0 ^
ninja>=1.11.1 ^
numpy==1.26.4 ^
omegaconf>=2.3.0 ^
onnxruntime ^
open-clip-torch>=2.20.0 ^
opencv-python==4.6.0.66 ^
pandas>=2.0.3 ^
pillow>=9.5.0 ^
pudb>=2022.1.3 ^
pytorch-lightning==2.0.1 ^
pyyaml>=6.0.1 ^
rembg ^
scipy>=1.10.1 ^
streamlit>=0.73.1 ^
streamlit-keyup==0.2.0 ^
tensorboardx==2.6 ^
timm>=0.9.2 ^
tokenizers==0.12.1 ^
torchdata==0.6.1 ^
torchmetrics>=1.0.1 ^
tqdm>=4.65.0 ^
transformers==4.19.1 ^
urllib3<1.27,>=1.25.4 ^
wandb>=0.15.6 ^
webdataset>=0.2.33 ^
wheel>=0.41.0 ^
gradio
if errorlevel 1 (
    echo [ERROR] Failed installing main dependency bundle
    pause
    exit /b 1
)

echo [STEP] Install CLIP...
pip install git+https://github.com/openai/CLIP.git
if errorlevel 1 (
    echo [ERROR] Failed installing CLIP
    pause
    exit /b 1
)

echo [STEP] Install datapipelines...
pip install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
if errorlevel 1 (
    echo [ERROR] Failed installing datapipelines
    pause
    exit /b 1
)

echo [STEP] Fix av version for compatibility...
pip uninstall -y av
pip install av==13.0.0
if errorlevel 1 (
    echo [WARNING] Failed to pin av==13.0.0. Continuing...
)

echo [STEP] Final sanity checks...
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"
python -c "import torchvision, cv2, omegaconf, pytorch_lightning, transformers, open_clip, av; print('basic deps ok')"
python -c "import sgm; print('sgm ok')"

echo.
echo =========================================
echo Setup finished.
echo Put SVD / SVD-XT checkpoints here:
echo .\generative-models\checkpoints
echo Dataset root can be:
echo .\ssv2
echo =========================================
pause