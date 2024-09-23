# FindTorch.cmake
# Locate the Torch library using Python

# Use Python to locate the torch CMake prefix path if Torch_DIR is not already set
if (NOT Torch_DIR)
    execute_process(
            COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE Torch_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

# Check if the required directories exist for Torch headers and libraries
find_path(TORCH_INCLUDE_DIR
        NAMES torch/torch.h
        HINTS ${Torch_DIR} ENV Torch_DIR
        PATH_SUFFIXES include)

find_library(TORCH_LIBRARY
        NAMES torch
        HINTS ${Torch_DIR} ENV Torch_DIR
        PATH_SUFFIXES lib)

# If both the include directory and library are found, mark Torch as found
if (TORCH_INCLUDE_DIR AND TORCH_LIBRARY)
    set(TORCH_FOUND TRUE)
    set(TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIR})
    set(TORCH_LIBRARIES ${TORCH_LIBRARY})
else()
    set(TORCH_FOUND FALSE)
endif()

# Output the found paths for debugging purposes
message(STATUS "Torch include directory: ${TORCH_INCLUDE_DIR}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

mark_as_advanced(TORCH_INCLUDE_DIR TORCH_LIBRARY)
