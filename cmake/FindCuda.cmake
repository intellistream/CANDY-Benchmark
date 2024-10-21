# Function to find valid CUDA compilers within a specific version range
function(find_valid_cuda MIN_CUDA_VERSION MAX_CUDA_VERSION)
    # List of candidate CUDA base directories
    if (EXISTS "/usr/bin/nvcc")
        set(CUDA_PATHS "/usr/bin")
    else()
        file(GLOB CUDA_PATHS /usr/local/cuda-*/bin)
    endif()

    set(VALID_CUDA_FOUND FALSE)

    # Loop over each candidate CUDA path
    foreach(CUDA_PATH ${CUDA_PATHS})
        # Search for nvcc in the current CUDA path
        set(CUDA_COMPILER_PATH ${CUDA_PATH}/nvcc)
        if (EXISTS ${CUDA_COMPILER_PATH})
            # Run nvcc --version to get the version output
            execute_process(
                    COMMAND ${CUDA_COMPILER_PATH} --version
                    OUTPUT_VARIABLE CUDA_VERSION_OUTPUT
                    OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            # Extract the version number from the output (e.g., "release 12.5")
            string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" _ ${CUDA_VERSION_OUTPUT})
            set(CUDA_VERSION ${CMAKE_MATCH_1})

            # Check if the CUDA version is within the provided range
            if (CUDA_VERSION VERSION_GREATER_EQUAL ${MIN_CUDA_VERSION} AND
                    CUDA_VERSION VERSION_LESS_EQUAL ${MAX_CUDA_VERSION})
                message(STATUS "Found valid CUDA compiler: ${CUDA_COMPILER_PATH}")
                message(STATUS "CUDA version: ${CUDA_VERSION}")
                set(VALID_CUDA_FOUND TRUE)
                break()  # Stop searching if a valid version is found
            else()
                message(WARNING "CUDA version ${CUDA_VERSION} found at ${CUDA_COMPILER_PATH} is outside the allowed range (${MIN_CUDA_VERSION} - ${MAX_CUDA_VERSION}), ignoring this path.")
            endif()
        else()
            message(STATUS "No CUDA compiler found in path: ${CUDA_PATH}")
        endif()
    endforeach()

    # If no valid CUDA was found, print a warning
    if (NOT VALID_CUDA_FOUND)
        message(WARNING "No valid CUDA compiler found in the range ${MIN_CUDA_VERSION} - ${MAX_CUDA_VERSION}. I don't think you can use CUDA...")
    else()
        set(ENV{CUDACXX} ${CUDA_COMPILER_PATH})
        message(STATUS "Finally using CUDA compiler at $ENV{CUDACXX}")
    endif()
endfunction()

# Call the function to find valid CUDA compilers with a specific version range
find_valid_cuda("11.0" "12.5")
