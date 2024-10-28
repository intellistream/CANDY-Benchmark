#!/bin/bash

get_cuda_version() {
    # Check if nvcc is available
    if command -v /usr/local/cuda/bin/nvcc &> /dev/null; then
        # Use nvcc to extract the version number
        cuda_version=$(/usr/local/cuda/bin/nvcc --version | grep "release" | grep -oP 'release \K[0-9]+\.[0-9]+')
        echo "CUDA is already installed. Version: $cuda_version"
        return 0
    else
        echo "CUDA is not installed or nvcc is not in your PATH, trying to install..."
        sudo apt install -y wget
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get install -y nvidia-gds-12-5 cuda-12-5 cuda-toolkit-12-5
        echo "Please go back to start.sh after the following reboot"
        sudo reboot
        return 1
    fi
}

# Check if there is an NVIDIA GPU installed using nvidia-smi
if nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU is installed."
    get_cuda_version
else
    echo "No NVIDIA GPU detected, going to the CPU-only setup."
    ./startWithoutCuda.sh
    exit 1
fi
