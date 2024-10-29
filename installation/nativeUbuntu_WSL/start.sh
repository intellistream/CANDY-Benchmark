#!/bin/bash
# Please make sure cuda is fully installed at /usr/local/cuda !!!!
# Function to get the major and minor version of CUDA using nvcc
get_cuda_version() {
    # Check if nvcc is available
    if command -v /usr/local/cuda/bin/nvcc &> /dev/null; then
        # Use nvcc to extract the version number
        cuda_version=$(/usr/local/cuda/bin/nvcc --version | grep "release" | grep -oP 'release \K[0-9]+\.[0-9]+')
    else
        echo "CUDA is not installed or nvcc is not in your PATH, I will try to install first."
        ./tryInstallCuda.sh
        exit 1
    fi
    echo $cuda_version
}

# Extract the major and minor version of CUDA
cuda_version=$(get_cuda_version)
echo "First, make sure you have sudo"
sudo ls
echo "Detected CUDA Version: $cuda_version"

# Replace dots with hyphens for the package versioning format used by Ubuntu packages
package_version=${cuda_version//./-}

# Formulate the package name
libcublas_package="libcublas-$package_version"

# Install the corresponding libcublas package
echo "Installing $libcublas_package..."
sudo apt-get update
sudo apt-get install -y $libcublas_package

if [ $? -eq 0 ]; then
    echo "$libcublas_package installation successful."
else
    echo "Failed to install $libcublas_package."
fi

echo "Installing others..."
sudo apt install -y liblapack-dev libblas-dev
sudo apt-get install -y graphviz libboost-all-dev swig
sudo apt-get install -y libcudnn8 libcudnn8-dev
pip install matplotlib pandas==2.0.0
pip install torch