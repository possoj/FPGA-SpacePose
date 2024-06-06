# Copyright (c) 2022, Julien Posso
# Copyright (c) 2021, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This dockerfile is mainly inspired from FINN dockerfile:
# https://github.com/possoj/finn-temp/blob/main/docker/Dockerfile.finn
# But a lot of the entrypoint was moved to the dockerfile to launch the container much faster and improve reproducibility
# Also removed unnecessary environment variables

FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
LABEL authors="Julien Posso"
LABEL description="Docker image for compiling and deploying Pytorch DNNs with FINN"

ARG VITIS_VERSION=2022.1
# Directory where VITIS is installed in the container (MUST BE THE SAME OUTSIDE THE CONTAINER)
ARG XILINX_PATH=/tools/Xilinx
# Directory where FINN will be installed in the container
ARG FINN_PATH=/tools/finn
# Directory where the project will be mounted inside the container
ARG PROJECT_ROOT=/workspace/pose_estimation/FPGA-SpacePose
# FINN version (release version https://github.com/Xilinx/finn/releases)
ARG FINN_VERSION=v0.9

ENV XRT_DEB_VERSION="xrt_202210.2.13.466_18.04-amd64-xrt"
ENV FINN_ROOT=$FINN_PATH
ENV PROJECT_ROOT=$PROJECT_ROOT

WORKDIR $PROJECT_ROOT

# some Vitis deps require a timezone to be specified, which hangs in Docker
# use workaround from https://grigorkh.medium.com/fix-tzdata-hangs-docker-image-build-cdb52cc3360d
ENV TZ="America/Montreal"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libc6-dev-i386 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    nano \
    zsh \
    rsync \
    git \
    openssh-client \
    sshpass \
    wget \
    sudo \
    unzip \
    zip \
    locales \
    lsb-core

RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config
RUN locale-gen "en_US.UTF-8"

# install XRT
RUN wget https://www.xilinx.com/bin/public/openDownload?filename=$XRT_DEB_VERSION.deb -O /tmp/$XRT_DEB_VERSION.deb  \
    && apt install -y /tmp/$XRT_DEB_VERSION.deb \
    && rm /tmp/$XRT_DEB_VERSION.deb

# install Verilator from source to get the right version
RUN apt-get install -y git perl python3 make autoconf g++ flex bison ccache libgoogle-perftools-dev numactl perl-doc  \
    libfl2 libfl-dev zlibc zlib1g zlib1g-dev
RUN git clone https://github.com/verilator/verilator
RUN cd verilator && \
    git checkout v4.224 && \
    autoconf && \
    ./configure && \
    make -j4 && \
    make install

# extra Python package dependencies (for testing and interaction)
RUN pip install pygments==2.4.1 \
    && pip install ipykernel==5.5.5 \
    && pip install --ignore-installed jupyter==1.0.0 \
    && pip install markupsafe==2.0.1 \
    && pip install matplotlib==3.3.1 --ignore-installed \
    && pip install pytest-dependency==0.5.1 \
    && pip install sphinx==5.0.2 \
    && pip install sphinx_rtd_theme==0.5.0 \
    && pip install pytest-xdist[setproctitle]==2.4.0 \
    && pip install pytest-parallel==0.1.0 \
    && pip install "netron>=5.0.0" \
    && pip install pandas==1.1.5 \
    && pip install scikit-learn==0.24.1 \
    && pip install tqdm==4.31.1 \
    && pip install deap==1.3.1 \
    && pip install mip==1.13.0 \
    && pip install networkx==2.8 \
    && pip install future-annotations==1.0.0 \
    && pip install dependencies==2.0.1 \
    && pip install tokenize-rt==4.2.1 \
    && pip install tclwrapper==0.0.1 \
    && pip install psutil==5.9.1

# Fetch and install FINN
# RUN git clone --recursive https://github.com/Xilinx/finn.git $FINN_ROOT
# Specific version of FINN
RUN git clone --depth 1 --branch $FINN_VERSION --single-branch https://github.com/Xilinx/finn.git $FINN_ROOT
RUN pip install -r $FINN_ROOT/requirements.txt

# Tentative to make finn-experimental work. But do not work because of the dash? Not working with the official FINN docker neither...
# mv $FINN_ROOT/deps/finn-experimental $FINN_ROOT/deps/finn_experimental \

# Fetch repos and install
RUN $FINN_ROOT/fetch-repos.sh \
    && pip install -e $FINN_ROOT/deps/qonnx \
    && pip install -e $FINN_ROOT/deps/finn-experimental \
    && pip install -e $FINN_ROOT/deps/brevitas \
    && pip install -e $FINN_ROOT/deps/pyverilator \
    && pip install -e $FINN_ROOT

# Specific to Spacecraft Pose Estimation project
RUN pip install xlsxwriter==3.0.3 pandas==1.2.3 paramiko==3.1.0 yacs==0.1.8 opencv-python==4.5.5.64

ENV VIVADO_IP_CACHE "/tmp/vivado_ip_cache"

# Ju environement variables
ENV VITIS_PATH=$XILINX_PATH/Vitis/$VITIS_VERSION
ENV HLS_PATH=$XILINX_PATH/Vitis_HLS/$VITIS_VERSION
ENV VIVADO_PATH=$XILINX_PATH/Vivado/$VITIS_VERSION
ENV FINN_BUILD_DIR=$PROJECT_ROOT/finn_build
ENV XILINX_XRT=/opt/xilinx/xrt
ENV OHMYXILINX=$FINN_ROOT/deps/oh-my-xilinx
ENV PATH=$PATH:$OHMYXILINX
ENV NUM_DEFAULT_WORKERS=16

# Don't understand exactly why I need to put this at the end in order to work. Still not working most of the time (non deterministic)...
RUN pip install -e git+https://github.com/fbcotter/dataset_loading.git@0.0.4#egg=dataset_loading

# Generate ssh keys
RUN ssh-keygen -f /root/.ssh/id_rsa -P "" -C pose_finn

# Whatever the $PROJECT_ROOT, thanks to the symbolic link we can call the entry point
RUN ln -s $PROJECT_ROOT/setup/build/finn_entrypoint.sh /workspace/finn_entrypoint.sh
ENTRYPOINT ["/bin/bash", "/workspace/finn_entrypoint.sh"]
CMD ["bash"]
