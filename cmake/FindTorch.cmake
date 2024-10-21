# Run the Python command and capture its output
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE PYTHON_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Append the Python command output to CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${PYTHON_OUTPUT}")