FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Seoul

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget vim build-essential cmake pkg-config \
    libgl1-mesa-glx libglib2.0-0 libxext6 libsm6 libxrender1 \
    python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm /tmp/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

WORKDIR /workspace
