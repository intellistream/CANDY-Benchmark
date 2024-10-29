#sudo apt-get install curl -y #If curl is not available, uncomment and download it.

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."

    # Download and install Docker
    curl -fsSL https://get.docker.com | sh

    # Enable and start Docker service
    sudo systemctl --now enable docker

    echo "Docker installation completed."
else
    echo "Docker is already installed."
fi

# Check if CUDA is installed
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Installing CUDA..."

    # Add NVIDIA package repository
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    wget https://developer.download.nvidia.com/compute/cuda/keys/NVIDIA-GPG-KEY
    sudo apt-key add NVIDIA-GPG-KEY
    sudo sh -c "echo 'deb https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/ /' > /etc/apt/sources.list.d/cuda.list"

    # Update the package lists
    sudo apt-get update

    # Install CUDA (modify the version if needed)
    sudo apt-get install -y cuda

    echo "CUDA installation completed."
else
    echo "CUDA is already installed."
fi

# Check if nvidia-docker2 is installed
if ! dpkg -l | grep -q nvidia-docker2; then
    echo "NVIDIA Container Toolkit not found. Installing..."

    # Get distribution information
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)

    # Add NVIDIA GPG key and repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    # Update the package lists
    sudo apt-get update

    # Install NVIDIA Docker
    sudo apt-get install -y nvidia-docker2

    # Configure runtime
    sudo nvidia-ctk runtime configure --runtime=docker

    # Restart Docker service
    sudo systemctl restart docker

    echo "NVIDIA Container Toolkit installation completed."
else
    echo "NVIDIA Container Toolkit is already installed."
fi