# GPU-capable image (also runs on CPU). Requires NVIDIA Container Toolkit /
# Docker Desktop WSL2 GPU support for CUDA.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /workspace
COPY pyproject.toml ./
RUN pip install --no-cache-dir numpy pillow pyyaml pytest torchvision==0.20.1
COPY . .
RUN pip install --no-cache-dir -e .
ENV PYTHONUNBUFFERED=1
CMD ["bash"]
