# FindTorch.cmake
# Locate the Torch library
# This module defines
#  TORCH_FOUND        - system has Torch
#  TORCH_INCLUDE_DIRS - the Torch include directories
#  TORCH_LIBRARIES    - link these to use Torch
#  TORCH_VERSION      - the Torch version found

find_path(TORCH_INCLUDE_DIR
        NAMES torch/torch.h
        HINTS ${Torch_DIR} ENV Torch_DIR
        PATH_SUFFIXES include)

find_library(TORCH_LIBRARY
        NAMES torch
        HINTS ${Torch_DIR} ENV Torch_DIR
        PATH_SUFFIXES lib)

find_library(TORCH_PYTHON_LIBRARY
        NAMES torch_python
        HINTS ${Torch_DIR} ENV Torch_DIR
        PATH_SUFFIXES lib)

if (TORCH_INCLUDE_DIR AND TORCH_LIBRARY AND TORCH_PYTHON_LIBRARY)
    set(TORCH_FOUND TRUE)
    set(TORCH_LIBRARIES ${TORCH_LIBRARY} ${TORCH_PYTHON_LIBRARY})
    set(TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIR})
else()
    set(TORCH_FOUND FALSE)
endif()

# Optional: Get the PyTorch version
if (TORCH_FOUND)
    execute_process(
            COMMAND python3 -c "import torch; print(torch.__version__)"
            OUTPUT_VARIABLE TORCH_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

mark_as_advanced(TORCH_INCLUDE_DIR TORCH_LIBRARY TORCH_PYTHON_LIBRARY)
