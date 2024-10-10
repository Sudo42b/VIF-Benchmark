######### 1. NGC의 Tensorflow 공식 이미지
ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:20.12-tf1-py3
FROM $BASE_IMAGE

# Set environment variables for Python and CUDA
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 필수 설치요소들 설치
######### 2. GPG Key 변경
# Install dependencies
RUN  apt-key del 7fa2af80 \
&& apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
&& apt-get update -y \
&& apt-get upgrade -y \
&& apt-get -y install build-essential \
&& apt-get -y install libgl1-mesa-glx\
&& apt-get -y install libglib2.0-0 \  
&& apt-get -y install --no-install-recommends \
    wget \
    git \
    curl \
    unzip \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*


# Install Miniconda to manage Python environments
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean --all --yes

# Create and activate the 'timer' environment
COPY timer.yml /tmp/timer.yml
RUN /opt/conda/bin/conda env create -f /tmp/timer.yml

# Set the default environment when running the container
RUN echo "source activate timer" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=timer
ENV PATH=/opt/conda/envs/timer/bin:$PATH


# Activate 'timer' environment and install TensorFlow 1.15 (with CUDA 10.0)
# RUN /bin/bash -c "source activate timer && conda install tensorflow-gpu=1.15 cudatoolkit=12.1 -y"

# Activate 'timer' environment and install compatible PyTorch (CUDA 10.0)
# RUN /bin/bash -c "source activate timer && conda install pytorch torchvision torchaudio cudatoolkit=12.1 -c pytorch -y"
# RUN /bin/bash -c "source activate timer && conda install pytorch torchvision torchaudio cudatoolkit=11 -c pytorch -y"

# Ensure the NVIDIA libraries are available
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Set the working directory
WORKDIR /workspace

# Expose Jupyter notebook and tensorboard ports
EXPOSE 8888 6006

# Run a shell by default
CMD ["/bin/bash"]

#Refer: https://sseongju1.tistory.com/62
#docker build -t tf0 -f Dockerfile .
#docker run -d -it --gpus all -p 8888:8888 -v $PWD:/workspace/VIF b105102bbd28 /bin/bash
#docker run -it --gpus all -p 8888:8888 -v $PWD:/workspace -w /workspace b105102bbd28 /bin/bash
#  b105102bbd28
# 'CONTAINER ID def94daa163a, NAMES romantic_lumiere' 실행
#docker exec -it b105102bbd28 /bin/bash
#conda env update --file timer.yml --prune

