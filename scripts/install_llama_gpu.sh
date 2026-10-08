#!/usr/bin/env bash
# llama-cpp-python that runs on the GPU of a Kaggle notebook, or a failure that says it does not.
#
# The prebuilt CUDA wheels are for Python 3.10-3.12; a notebook run through the API has 3.13, where `pip install
# llama-cpp-python` quietly builds a CPU-only one and every model runs at CPU speed (a 2B model took 12 s a call).
# So: the prebuilt wheel if there is one, otherwise a build with CUDA for this card, and then a check that the
# result offloads to the GPU, failing the notebook when it does not rather than letting a job run on the CPU.
set -e
pip install -q huggingface_hub
python -c "import sys; print('python', sys.version.split()[0])"
if pip install -q "llama-cpp-python[server]" --only-binary=:all: \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124; then
    echo "prebuilt CUDA wheel installed"
else
    arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    echo "no prebuilt wheel for this Python: building llama-cpp-python with CUDA for sm_${arch}"
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${arch}" FORCE_CMAKE=1 \
        pip install -q --no-cache-dir "llama-cpp-python[server]"
fi
python -c "
from llama_cpp import llama_cpp
ok = bool(llama_cpp.llama_supports_gpu_offload())
print('llama.cpp gpu offload:', ok)
raise SystemExit(0 if ok else 3)"
