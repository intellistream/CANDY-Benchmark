# Base image with CUDA 11.7 and Python 3.10
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ENV USER=candy
ENV PASSWD=candy
# Install essential packages and dependencies
RUN apt-get update && apt-get install -y --allow-change-held-packages \
    sudo \
    build-essential \
    cmake \
    curl \
    unzip \
    sudo \
    liblapack-dev \
    libblas-dev \
    libboost-dev \
    rsync \
    nano \
    libcudnn8 \
    libcudnn8-dev \
    graphviz \
    python3 \
    python3-pip \
    openssh-server \
    gdb \
    liblog4cxx-dev \
    git \
    libcupti-dev \
    && apt-get clean 


# Install Python dependencies
RUN pip3 install torch matplotlib pandas==2.0.0
# 添加用户并配置密码/Add user and config passwd
RUN useradd -m ${USER} && yes ${PASSWD} | passwd ${USER}
RUN usermod -G sudo ${USER}
# Please use this folder for your clion project
RUN mkdir /home/${USER}/candy
RUN chown -R ${USER} /home/${USER}/candy
# Configure SSH
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Set up workspace directory, only for root purpose of debuging
WORKDIR /workspace

# Default command to run SSH server
CMD ["/usr/sbin/sshd", "-D"]

