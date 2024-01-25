FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python-is-python3 \
    libsm6 \
    vim \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/
# install requirements file
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
ENV DEBIAN_FRONTEND=noninteractive
#copy model weights
#COPY models models
#copy model source code
COPY main.py .
COPY helpers.py .
COPY src src
ENV CUDA_MODULE_LOADING LAZY
ENV LOG_VERBOSE 0
# fix triton in colab
CMD python3 main.py