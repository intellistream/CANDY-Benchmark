# CANDY

A Lib and benchmark for AKNN. This project is compatable with
libtorch

## Extra Cmake options (set by cmake -Dxxx=ON/OFF)

- ENABLE_PAPI, this will enable PAPI-based perf tools (OFF by default)
    - you need first cd to thirdparty and run installPAPI.sh to enable PAPI support, or also set REBUILD_PAPI to ON
    - not used in C++ benchmark currently
- ENABLE_HDF5, this will enable you to load data from HDF5 (OFF by default)
    - we have included the source code of hdf5 lib, no extra dependency
- ENABLE_PYBIND, this will enable you to make python binds, i.e., PyRAINA (OFF by default)
    - we have included the source code of pybind 11 in third party folder, please make sur it is complete

## One-Key build examples with auto solving of dependencies

- buildWithCuda.sh To build CANDY and PyCANDY with cuda support, make sure you have cuda installed before it
- buildCPUOnly.sh This is a CPU-only version
- After either one, you can run the following to add PyCANDY To your default python environment

```shell
 python3 setup.py install --user
```

### Where are the evaluation scripts

See build/benchmark/scripts/rerunAll.sh after one-key build

## Manual build

### Requires G++11

The default version of gcc/g++ on ubuntu 22.04 (jammy) is good enough.

#### For x64 ubuntu older than 21.10

run following first

```shell
sudo add-apt-repository 'deb http://mirrors.kernel.org/ubuntu jammy main universe'
sudo apt-get update
```

Then, install the default gcc/g++ of ubuntu22.04

```shell
sudo apt-get install gcc g++ cmake python3 python3-pip
```

#### For other architectures

Please manually edit your /etc/sources.list, and add a jammy source, then

```shell
sudo apt-get update
sudo apt-get install gcc g++ cmake 
```

Please invalidate the jammy source after installing the gcc/g++ from jammy, as some packs from
jammy may crash down your older version

#### WARNNING

Please do not install the python3 from jammy!!! Keep it raw is safer!!!

### Requires BLAS and LAPACK

```shell
sudo apt install liblapack-dev libblas-dev
```

### (Optional) Install graphviz

```shell
sudo apt-get install graphviz
pip install torchviz
```

### Requires Torch

You may refer to https://pytorch.org/get-started/locally/ for mor details, following are the minimal requirements
DO NOT USE CONADA!!!!!

#### (Optional) Cuda-based torch

Note:

- useCuda config is only valid when cuda is installed
- this branch only allows blackbox call on torch-cuda functions!
  You may wish to install cuda for faster pre-training on models, following is a reference procedure. Please refer
  to https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda
sudo apt-get install nvidia-gds
sudo apt-get install libcudnn8 libcudnn8-dev libcublas-11-7
```

The libcublas depends on your specific version.
Then you may have to reboot for enabling cuda.

DO INSTALL CUDA BEFORE INSTALL CUDA-BASED TORCH!!!

##### Cuda on Jetson

There is no need to install cuda if you use a pre-build jetpack on jetson. It will neither work,:(
Instead, please only check your libcudnn8 and libcublas

```shell
sudo apt-get install libcudnn8 libcudnn8-dev libcublas-*
```

#### (Required) Install pytorch (should install separately)

```shell
sudo apt-get install python3 python3-pip
```

(w/ CUDA):
(Please make all cuda dependencies installed before pytorch!!!)

```shell
pip3 install torch==1.13.0 torchvision torchaudio
```

(w/o CUDA)

```shell
pip3 install --ignore-installed torch==1.13.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

*Note: Conflict between torch1.13.0+cpu and torchaudio+cpu may occur under python version > 3.10*

### (Optional) Requires SPDK (contained source in this repo as third party)

The Storage Performance Development Kit (SPDK) provides a set of tools and libraries for writing high performance,
scalable, user-mode storage applications. It achieves high performance by moving all of the necessary drivers into
userspace and operating in a polled mode instead of relying on interrupts, which avoids kernel context switches and
eliminates interrupt handling overhead.

#### How to build SPDK

- cd to thirdparty and run cloneSPDK.sh, SPDK will be compiled and installed at default OS path
    - Alternatively, set -DREBUILD_SPDK=ON to let CMAKE do this for you

#### How to use SPDK in CANDY

- set -DENABLE_SPDK=ON in cmake CANDY

#### Known issues

isa-l is disabled, as it conflicts with some other c++ libs

### (optional) Requires PAPI (contained source in this repo as third party)

PAPI is a consistent interface and methodology for collecting performance counter information from various hardware and
software components: https://icl.utk.edu/papi/.
, CANDY includes it in thirdparty/papi_7_0_1.

#### How to build PAPI

- cd to thirdparty and run installPAPI.sh, PAPI will be compiled and installed in thirdparty/papi_build

#### How to verify if PAPI works on my machine

- cd to thirdparty/papi_build/bin , and run papi_avail by sudo, there should be at least one event avaliable
- the run papi_native_avail, the printed tags are valid native events.
- please report to PAPI authors if you find your machine not supported

#### How to use PAPI in CANDY

- set -DENABLE_PAPI=ON in cmake CANDY
- in your top config file, add two config options:
    - usePAPI,1,U64
    - perfUseExternalList,1,U64
    - if you want to change the file to event lists, please also set the following:
        - perfListSrc,<the path to your list>,String
- edit the perfLists/perfList.csv in your BINARY build path of benchmark (or your own list path), use the following
  format
    - <the event name tag you want CANDY to display>, <The inline PAPI tags from papi_native_avail/papi_avail>,
      String
- please note that papi has a limitation of events due to hardware constraints, so only put 2~3 in each run

### Build steps

(CUDA-related is only necessary if your pytorch has cuda, but it's harmless if you don't have cuda.)

#### Build in shell

```shell
export CUDACXX=/usr/local/cuda/bin/nvcc
mkdir build && cd build
```

Build for release by default:

```shell
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make 
```

Or you can build with debug mode to debug cpp dynamic lib:

```shell
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make 
```

#### Tips for build in Clion

There are bugs in the built-in cmake of Clion, so you can not run
-DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`.
Following may help:

- Please run 'import torch;print(torch.utils.cmake_prefix_path)' manually first, and copy the path
- Paste the path to -DCMAKE_PREFIX_PATH=
- Manually set the environment variable CUDACXX as "/usr/local/cuda/bin/nvcc" in Clion's cmake settings

### (Optional) Distributed CANDY with Ray

#### To build

1. Install ray by

```shell
pip install ray==2.8.1 ray-cpp==2.8.1
```

Get the installation path by

```shell
 ray cpp --show-library-path
```

*Note: use sudo before pip if does not work

2. Run the following before building me

```shell
 export RAYPATH= <The lib path printed in setp 1>
```

3. Add the following Cmake options in cmake me

```shell
-DENABLE_RAY=ON
```

4. Go to https://docs.ray.io/en/latest/ray-observability/getting-started.html#observability-getting-started if you need
   a dashboard

#### To run

* Please set up the cluster before run the program
    * To start the head node
   ```shell
   ray start --head
   ``` 
    * To start the other nodes
   ```shell
   ray start --address <ip of head node>:6379 --node-ip-address <ip of this node>
   ``` 
  or

   ```shell
   ray start --address <ip of head node>:6379
   ``` 

    * To run the program

   ```shell
   export RAY_ADDRESS=<ip:6379>
   ./<a program with ray support>
   ``` 	

* Please make sure the file path of built progarm, other dependency like torch, is totally the same for all computers in
  the cluster
* For different arch, please recompile from source code, but keep the path, name of the *.so and binary the same
* torch::Tensor seems to be unable to be packed as remote args (both in and out), please convert to std::vector<float>

#### Known issues

Does not work with python

### Local generation of the documents

You can also re-generate them locally, if you have the doxygen and graphviz. Following are how to install them in ubuntu
21.10/22.04

```shell
sudo apt-get install doxygen
sudo apt-get install graphviz
sudo apt-get install texlive-latex-base
sudo apt-get install texlive-fonts-recommended
sudo apt-get install texlive-fonts-extra
sudo apt-get install texlive-latex-extra
```

Then, you can do

```shell
./genDoc.SH
```

#### HTML pages

To get the documents in doc/html folder, and start at index.html

#### pdf manual

To find the refman.pdf at the root

## Evaluation scripts

They are place under benchmark.scripts, for instance, the following allows to scan the number of elements in
the matrix A's row
(Assuming you are at the root dir of this project, and everything is built under build folder)

```shell
cd build/benchmark/scripts
cd scanARow #enter what you want to evaluate
sudo ls # run an ls to make it usable, as perf events requires sudo
python3 drawTogether.py # no sudo here, sudo is insider *.py
cd ../figures
```

You will find the figures then.

## Known issues
### CUDA and torch
For torch>=2.0, the header may require you to only use c++17 for tensor, please do either the following such that the nvcc works when include <torch.h>:
- Downgrade torch into 1.13.0
- Upgrade CUDA to 12.5 or above
### How to run SPDK-related functions without sudo/root
  ```shell
   sudo setcap all+ep <your_app_name>
   ./<your_app_name>
   ```


