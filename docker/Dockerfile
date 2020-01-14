# Re-use from Nvidia's CUDA Dockerfile
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
# Python Version
ARG PYTHON_VERSION=3.7
# Python essentials
RUN apt-get update &&\
    apt-get install -y --no-install-recommends python3 python3-dev \
      python3-setuptools gcc libtinfo-dev zlib1g-dev clang python3-pip llvm \
      llvm-dev build-essential cmake git curl vim ca-certificates libjpeg-dev \
      libpng-dev &&\
    rm -rf /var/lib/apt/lists/*
# Install conda and packages.
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch && \
     /opt/conda/bin/conda install -y numba matplotlib pandas scipy && \
     #/opt/conda/bin/conda install -y pyculib -c anaconda && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
