# The environment for running experiments on a Windows machine with an NVIDIA card (written for Windows 10,
# RTX 4060 Ti 8 GB; not yet run on one - the first run is the test).
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1 -SkipLlm     # CPU work only (RL, search)
#
# It checks what is installed, makes .venv in the repository, installs the project with the RL and wandb extras,
# the CUDA build of torch, and llama-cpp-python with GPU offload, and then checks that the model would really run on
# the card - failing loudly when it would run on the CPU, as scripts/install_llama_gpu.sh does on Kaggle. At the
# end it prints the figures that decide what goes where: CPU threads and free video memory.
param(
    [switch]$SkipLlm,
    [string]$Cuda = "cu124"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Need($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) { throw "$name not found. $hint" }
}
function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "failed: $exe $($arguments -join ' ')" }
}

Need git "Install Git for Windows."
Need python "Install Python 3.12 from python.org and tick 'Add to PATH'."
$version = (python -c "import sys; print('%d.%d' % sys.version_info[:2])")
if ([version]$version -lt [version]"3.12") { throw "Python $version found; the project needs 3.12 or newer." }

if (-not $SkipLlm) {
    Need nvidia-smi "Install the NVIDIA driver (the CUDA toolkit is not needed for the prebuilt wheels)."
    Run nvidia-smi @("--query-gpu=name,memory.total,driver_version", "--format=csv")
}

if (-not (Test-Path .venv)) { Run python @("-m", "venv", ".venv") }
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
Run $py @("-m", "pip", "install", "-q", "-U", "pip")

# torch first, from the CUDA index, so that the extras below do not pull the CPU build in its place
if ($SkipLlm) {
    Run $py @("-m", "pip", "install", "-q", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
} else {
    Run $py @("-m", "pip", "install", "-q", "torch", "--index-url", "https://download.pytorch.org/whl/$Cuda")
}
Run $py @("-m", "pip", "install", "-q", "-e", ".[rl,logging,test]", "huggingface_hub", "kaggle")

if (-not $SkipLlm) {
    # the prebuilt CUDA wheel; a source build on Windows needs Visual Studio and the CUDA toolkit, so it is not tried
    Run $py @("-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", "llama-cpp-python",
              "--only-binary=:all:", "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/$Cuda")
    Run $py @("-c", "from llama_cpp import llama_cpp; ok = bool(llama_cpp.llama_supports_gpu_offload()); print('llama.cpp gpu offload:', ok); raise SystemExit(0 if ok else 3)")
    Run $py @("-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")
}

Write-Host ""
Write-Host "CPU threads:" ([Environment]::ProcessorCount)
if (-not $SkipLlm) { Run nvidia-smi @("--query-gpu=memory.free", "--format=csv,noheader") }
Write-Host "Ready. Use .venv\Scripts\python.exe for the scripts; run 'claude remote-control' in this folder to hand the session over."
