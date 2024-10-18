# CANDY

A library and benchmark suite for Approximate Nearest Neighbor Search (ANNS). This project is compatible with LibTorch.

## Table of Contents

- [Quick Start Guide](#quick-start-guide)
  - [Docker Support](#docker-support)
  - [Build Without Docker](#build-without-docker)
    - [Build with CUDA Support](#build-with-cuda-support)
    - [Build without CUDA (CPU-Only Version)](#build-without-cuda-cpu-only-version)
  - [Installing PyCANDY](#installing-pycandy)
  - [CLion Configuration](#clion-configuration)
- [Evaluation Scripts](#evaluation-scripts)
- [Additional Information](#additional-information)
---

## Quick Start Guide

### Docker Support

We provide Docker support to simplify the setup process.

1. **Navigate to the `./docker` directory:**

   ```shell
   cd ./docker
   ```

2. **Build and start the Docker container:**

   ```shell
   ./start.sh
   ```

   This script will build the Docker container and start it.

3. **Inside the Docker container, run the build script to install dependencies and build the project:**

  - **With CUDA support:**

    ```shell
    ./buildWithCuda.sh
    ```

  - **Without CUDA (CPU-only version):**

    ```shell
    ./buildCPUOnly.sh
    ```

### Build Without Docker

If you prefer to build without Docker, follow these steps.

#### Build with CUDA Support

To build CANDY and PyCANDY with CUDA support:

```shell
./buildWithCuda.sh
```

#### Build without CUDA (CPU-Only Version)

For a CPU-only version:

```shell
./buildCPUOnly.sh
```

These scripts will install dependencies and build the project.

### Installing PyCANDY

After building, you can install PyCANDY to your default Python environment:

```shell
python3 setup.py install --user
```

### CLion Configuration

When developing in CLion, you must manually configure:

1. **CMake Prefix Path:**


### Requires BLAS, LAPACK, boost and swig

```shell
sudo apt install liblapack-dev libblas-dev libboost-all-dev swig
```

  - Run the following command in your terminal to get the CMake prefix path:

    ```shell
    python3 -c 'import torch; print(torch.utils.cmake_prefix_path)'
    ```


  - Copy the output path and set it in CLion's CMake settings as:

    ```
    -DCMAKE_PREFIX_PATH=<output_path>
    ```

2. **Environment Variable `CUDACXX`:**

  - Manually set the environment variable `CUDACXX` to:

    ```
    /usr/local/cuda/bin/nvcc
    ```

## Evaluation Scripts

Evaluation scripts are located under `benchmark/scripts`.

To run an evaluation (e.g., scanning the number of elements in matrix A's row):

```shell
cd build/benchmark/scripts/scanARow
sudo ls  # Required for perf events
python3 drawTogether.py
cd ../figures
```

Figures will be generated in the `figures` directory.

---

## Additional Information

<details>
<summary><strong>Click to Expand</strong></summary>

### Table of Contents

- [Extra CMake Options](#extra-cmake-options)
- [Manual Build Instructions](#manual-build-instructions)
  - [Requirements](#requirements)
  - [Build Steps](#build-steps)
  - [CLion Build Tips](#clion-build-tips)
- [CUDA Installation (Optional)](#cuda-installation-optional)
  - [Install CUDA (if using CUDA-based Torch)](#install-cuda-if-using-cuda-based-torch)
  - [CUDA on Jetson Devices](#cuda-on-jetson-devices)
- [Torch Installation](#torch-installation)
  - [Install Python and Pip](#install-python-and-pip)
  - [Install PyTorch](#install-pytorch)
- [PAPI Support (Optional)](#papi-support-optional)
  - [Build PAPI](#build-papi)
  - [Verify PAPI Installation](#verify-papi-installation)
  - [Enable PAPI in CANDY](#enable-papi-in-candy)
- [Distributed CANDY with Ray (Optional)](#distributed-candy-with-ray-optional)
  - [Build with Ray Support](#build-with-ray-support)
  - [Running with Ray](#running-with-ray)
  - [Ray Dashboard (Optional)](#ray-dashboard-optional)
- [Local Documentation Generation (Optional)](#local-documentation-generation-optional)
  - [Install Required Packages](#install-required-packages)
  - [Generate Documentation](#generate-documentation)
    - [Accessing Documentation](#accessing-documentation)
- [Known Issues](#known-issues)

---

### Extra CMake Options

You can set additional CMake options using `cmake -D<option>=ON/OFF`:

- `ENABLE_PAPI` (OFF by default)
  - Enables PAPI-based performance tools.
  - **Setup**:
    - Navigate to the `thirdparty` directory.
    - Run `installPAPI.sh` to enable PAPI support.
    - Alternatively, set `REBUILD_PAPI` to `ON`.
- `ENABLE_HDF5` (OFF by default)
  - Enables loading data from HDF5 files.
  - The HDF5 source code is included; no extra dependency is required.
- `ENABLE_PYBIND` (OFF by default)
  - Enables building Python bindings (PyCANDY).
  - Ensure the `pybind11` source code in the `thirdparty` folder is complete.

### Manual Build Instructions

#### Requirements

- **Compiler**: G++11 or newer.
  - The default `gcc/g++` version on Ubuntu 22.04 (Jammy) is sufficient.
- **BLAS and LAPACK**:
  ```shell
  sudo apt install liblapack-dev libblas-dev
  ```
- **Graphviz (Optional)**:
  ```shell
  sudo apt-get install graphviz
  pip install torchviz
  ```

#### Build Steps

1. **Set the CUDA Compiler Path** (if using CUDA):

   ```shell
   export CUDACXX=/usr/local/cuda/bin/nvcc
   ```

2. **Create Build Directory**:

   ```shell
   mkdir build && cd build
   ```

3. **Configure CMake**:

   ```shell
   cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch; print(torch.utils.cmake_prefix_path)'` ..
   ```

4. **Build the Project**:

   ```shell
   make
   ```

**For Debug Build**:

```shell
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=`python3 -c 'import torch; print(torch.utils.cmake_prefix_path)'` ..
make
```

#### CLion Build Tips

- Manually retrieve the CMake prefix path:

  ```shell
  python3 -c 'import torch; print(torch.utils.cmake_prefix_path)'
  ```

- Set the `-DCMAKE_PREFIX_PATH` in CLion's CMake settings.
- Set the environment variable `CUDACXX` to `/usr/local/cuda/bin/nvcc` in CLion.

### CUDA Installation (Optional)

#### Install CUDA (if using CUDA-based Torch)

Refer to the [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu) for more details.

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda
sudo apt-get install nvidia-gds
sudo apt-get install libcudnn8 libcudnn8-dev libcublas-11-7
```

**Note**: Ensure CUDA is installed before installing CUDA-based Torch. Reboot your system after installation.

#### CUDA on Jetson Devices

- No need to install CUDA if using a pre-built JetPack on Jetson.
- Ensure `libcudnn8` and `libcublas` are installed:

  ```shell
  sudo apt-get install libcudnn8 libcudnn8-dev libcublas-*
  ```

### Torch Installation

Refer to the [PyTorch Get Started Guide](https://pytorch.org/get-started/locally/) for more details.

#### Install Python and Pip

```shell
sudo apt-get install python3 python3-pip
```

#### Install PyTorch

- **With CUDA**:

  ```shell
  pip3 install torch==2.4.0 torchvision torchaudio
  ```

- **Without CUDA**:

  ```shell
  pip3 install --ignore-installed torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

**Note**: Conflict between `torch2.4.0+cpu` and `torchaudio+cpu` may occur with Python versions > 3.10.

### PAPI Support (Optional)

PAPI provides a consistent interface for collecting performance counter information.

#### Build PAPI

- Navigate to the `thirdparty` directory.
- Run `installPAPI.sh`.
- PAPI will be compiled and installed in `thirdparty/papi_build`.

#### Verify PAPI Installation

- Navigate to `thirdparty/papi_build/bin`.
- Run `sudo ./papi_avail` to check available events.
- Run `./papi_native_avail` to view native events.

#### Enable PAPI in CANDY

- Set `-DENABLE_PAPI=ON` when configuring CMake.
- Add the following to your top-level config file:

  ```
  usePAPI,1,U64
  perfUseExternalList,1,U64
  ```

- To specify custom event lists, set:

  ```
  perfListSrc,<path_to_your_list>,String
  ```

- Edit `perfLists/perfList.csv` in your build directory to include desired events.

### Distributed CANDY with Ray (Optional)

#### Build with Ray Support

1. **Install Ray**:

   ```shell
   pip install ray==2.8.1 ray-cpp==2.8.1
   ```

2. **Get Ray Library Path**:

   ```shell
   ray cpp --show-library-path
   ```

3. **Set `RAYPATH` Environment Variable**:

   ```shell
   export RAYPATH=<ray_library_path>
   ```

4. **Configure CMake**:

   ```shell
   cmake -DENABLE_RAY=ON ..
   ```

#### Running with Ray

- **Start the Head Node**:

  ```shell
  ray start --head
  ```

- **Start Worker Nodes**:

  ```shell
  ray start --address <head_node_ip>:6379 --node-ip-address <worker_node_ip>
  ```

- **Run the Program**:

  ```shell
  export RAY_ADDRESS=<head_node_ip>:6379
  ./<your_program_with_ray_support>
  ```

**Notes**:

- Ensure the file paths and dependencies are identical across all nodes.
- For different architectures, recompile the source code on each node.
- `torch::Tensor` may not be serializable; consider using `std::vector<float>` instead.

#### Ray Dashboard (Optional)

Refer to the [Ray Observability Guide](https://docs.ray.io/en/latest/ray-observability/getting-started.html#observability-getting-started) to set up a dashboard.

### Local Documentation Generation (Optional)

#### Install Required Packages

```shell
sudo apt-get install doxygen graphviz
sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra
```

#### Generate Documentation

```shell
./genDoc.SH
```

##### Accessing Documentation

- **HTML Pages**: Located in `doc/html/index.html`.
- **PDF Manual**: Found at `refman.pdf` in the root directory.

### Known Issues

- Conflicts may occur with certain versions of PyTorch and Python.

</details>