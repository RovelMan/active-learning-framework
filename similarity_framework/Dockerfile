FROM nvidia/cuda:9.0-base-ubuntu16.04

WORKDIR /workspace
RUN mkdir -p /root/.ssh

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        curl \
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install git python3-pip libsm6 libxext6 libxrender-dev -y
RUN pip3 install --upgrade pip

RUN pip3 install torch==1.0.0 torchvision==0.2.1 opencv-python numpy==1.14.3 scipy==1.0.1 scikit-image==0.13.0 tqdm==4.28.1 IPython[all]


# For CUDA profiling, TensorFlow requires CUPTI. Maybe PyTorch needs this too.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# IPython
EXPOSE 8888

# FROM nvcr.io/nvidia/tensorflow:18.12-py3

# RUN apt-get update
# RUN apt-get install git python3-pip libsm6 libxext6 libxrender-dev -y
# RUN pip3 install --upgrade pip

# RUN pip3 install torch==0.4.0 torchvision==0.2.1 opencv-python numpy==1.14.3 scipy==1.0.1 scikit-image==0.13.0 tqdm==4.28.1 IPython[all]
