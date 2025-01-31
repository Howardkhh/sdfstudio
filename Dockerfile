# Define base image.
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Variables used at build time.
## CUDA architectures, required by Colmap and tiny-cuda-nn.
## NOTE: All commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Asia/Taipei
## CUDA architectures, required by tiny-cuda-nn.
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages.
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python-is-python3 \
    python3.8-dev \
    python3-pip \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install GLOG (required by ceres).
RUN git clone --branch v0.6.0 https://github.com/google/glog.git --single-branch && \
    cd glog && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install && \
    cd ../.. && \
    rm -r glog
# Add glog path to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# Install Ceres-solver (required by colmap).
RUN git clone --branch 2.1.0 https://ceres-solver.googlesource.com/ceres-solver.git --single-branch && \
    cd ceres-solver && \
    git checkout $(git describe --tags) && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j && \
    make install && \
    cd ../.. && \
    rm -r ceres-solver

# Install colmap.
RUN git clone --branch 3.7 https://github.com/colmap/colmap.git --single-branch && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -DCUDA_ENABLED=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    make -j && \
    make install && \
    cd ../.. && \
    rm -r colmap
    
# Create non root user and setup environment.
# RUN groupadd -g 1002 vglusers
# RUN useradd -m -d /home/user -g root -G sudo,vglusers -u 1000 user

# Switch to new uer and workdir.
# USER 1000
# RUN mkdir /home/user
# WORKDIR /home/user

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

# Upgrade pip and install packages.
RUN python3.8 -m pip install --upgrade pip setuptools pathtools promise
# Install pytorch and submodules.
RUN python3.8 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# Install tynyCUDNN.
RUN python3.8 -m pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch
# Insatll hloc
RUN git clone --recursive https://github.com/cvg/Hierarchical-Localization/ && \
    cd Hierarchical-Localization/ && \
    python -m pip install . && \
    cp -r third_party /usr/local/lib/python3.8/dist-packages/ && \
    cd ..

# Copy nerfstudio folder and give ownership to user.
ARG USER=howardkhh
ARG USER_ID=1014
ARG STUDENT_GROUP_ID=1001
RUN groupadd -g $STUDENT_GROUP_ID student
RUN useradd -u $USER_ID -g student -ms /bin/bash $USER
RUN groupadd -g 1002 vglusers
RUN usermod -aG vglusers $USER
USER $USER
ADD . /home/$USER/sdfstudio
USER root

# Install nerfstudio dependencies.
RUN cd /home/$USER/sdfstudio && \
    python3.8 -m pip install -e . && \
    cd ..

RUN chown -Rh $USER:student /home/$USER/sdfstudio
USER $USER
WORKDIR /home/$USER/sdfstudio

# Install nerfstudio cli auto completion and enter shell if no command was provided.
CMD ns-install-cli --mode install && /bin/bash

