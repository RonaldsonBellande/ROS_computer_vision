ARG ML_ARCHITECTURE_VERSION=latest

FROM ubuntu:20.04 as base_build
FROM nvidia/cuda:11.2.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHON_VERSION="3.8"
ENV CUDNN_VERSION=8.1.0.77
ENV TF_TENSORRT_VERSION=7.2.2
ENV CUDA=11.2
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

ARG ML_ARCHITECTURE_VERSION_GIT_BRANCH=master
ARG ML_ARCHITECTURE_VERSION_GIT_COMMIT=HEAD

LABEL maintainer=ronaldsonbellande@gmail.com
LABEL ml_architecture_github_branchtag=${ML_ARCHITECTURE_VERSION_GIT_BRANCH}
LABEL ml_architecture_github_commit=${ML_ARCHITECTURE_VERSION_GIT_COMMIT}

# Ubuntu setup
RUN apt-get update -y
RUN apt-get upgrade -y

# RUN workspace and sourcing
WORKDIR ./
COPY requirements.txt .

# Install dependencies for system
RUN apt-get update && apt-get install -y --no-install-recommends <system_requirements.txt && \
  apt-get upgrade -y && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install python 3.8 and make primary 
RUN add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && apt-get install -y \
  python3.8 python3.8-dev python3-pip python3.8-venv && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Pip install 
RUN pip3 install --upgrade pip

# Install python libraries
RUN pip --no-cache-dir install -r requirements.txt

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
  apt-get update && apt-get install -y --no-install-recommends \
  cuda-nvrtc-${CUDA/./-} \
  libcudnn8=${CUDNN_VERSION}-1+cuda${CUDA} \
  -r cuda_requirements.txt

# We don't install libnvinfer-dev since we don't need to build against TensorRT
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub && \
  echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /"  > /etc/apt/sources.list.d/tensorRT.list && \
  apt-get update && \
  apt-get install -y --no-install-recommends libnvinfer7=${TF_TENSORRT_VERSION}-1+cuda11.0 \
  libnvinfer-plugin7=${TF_TENSORRT_VERSION}-1+cuda11.0 \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*;

