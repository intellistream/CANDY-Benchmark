
cmake_minimum_required(VERSION 3.14)
project(CANDY CXX)
include (cmake/FindCuda.cmake)
include (cmake/FindTorch.cmake)
find_package(Torch REQUIRED)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_RULE_MESSAGES OFF)
# Custom CMake find instructions and macros
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(cmake/macros.cmake)
include(cmake/default.cmake)


# C++ Standard
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#gcc 10 g++10
# First compile faiss before anything else
#set(CMAKE_CXX_FLAGS "-fno-openmp")
#test avx2

add_subdirectory(thirdparty/faiss)

# Set specific options for Faiss compilation
#target_compile_options(faiss PRIVATE "-fno-openmp")
set(LIBRARIES ${LIBRARIES} faiss)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(LIBRARIES ${LIBRARIES} ${TORCH_LIBRARIES})
# Set Optimization Flags
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Werror=return-type -Wno-interference-size")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DNO_RACE_CHECK -DCANDY_DEBUG_MODE=1")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-ignored-qualifiers -Wno-sign-compare -O3")
set(PROJECT_BINARY_DIR_RAW ${PROJECT_BINARY_DIR})

# Valid values are "generic", "avx2", "avx512".

detect_avx512_support(AVX512_AVAILABLE)
# Use AVX-512 based on the result
if(AVX512_AVAILABLE)
    message(STATUS "AVX-512 support detected.")
    set(CANDY_AVX512 1)
    set(CANDY_AVX2 1)
else()
    message(STATUS "AVX-512 support NOT detected.")
    detect_avx2_support(AVX2_AVAILABLE)
    if(AVX2_AVAILABLE)
        message(STATUS "AVX-2 support detected.")
        set(CANDY_AVX2 1)
    else ()
        message(STATUS "AVX-2 support not detected.")
        set(CANDY_AVX2 0)
        set(CANDY_AVX512 0)
    endif ()
endif()
configure_file(
        "${PROJECT_SOURCE_DIR}/include/simd_config.h.in"
        "${PROJECT_BINARY_DIR}/include/simd_config.h"
)


#set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CUDA_FLAGS "-std=c++11")
option(ENABLE_OPENCL
        "Enable opencl support"
        OFF
)
option(ENABLE_RAY
        "Enable the distributed computing powered by ray"
        OFF
)
option(ENABLE_HDF5
        "Enable load data from hdf5 file"
        OFF
)
option(ENABLE_SPTAG
        "Integrate SPTAG LIBS"
        OFF
)
option(ENABLE_PYBIND
        "Enable original pybind and build CANDY python"
        OFF
)

#OPTIONAL OPENCL
if (NOT ENABLE_OPENCL)
    message(STATUS "I will NOT include opencl support ")
    set(CANDY_CL 0)
else ()
    set(CANDY_CL 1)
    message(STATUS "I will include opencl support, pls make sure you have corresponding libs")
    set(LIBRARIES ${LIBRARIES} OpenCL)
endif ()
configure_file(
        "${PROJECT_SOURCE_DIR}/include/opencl_config.h.in"
        "${PROJECT_BINARY_DIR}/include/opencl_config.h"
)
option(ENABLE_PAPI
        "Enable papi support, pls first compile papi or set REBUILD_PAPI to ON"
        OFF
)
# OPTIONAL PAPI
if (NOT ENABLE_PAPI)
    message(STATUS "I will NOT use PAPI ")
    set(CANDY_PAPI 0)
else ()
    set(CANDY_PAPI 1)
    message(STATUS "I will try to use PAPI for HW counters, pls make sure your arch supports it")
    option(REBUILD_PAPI
            "Rebuild the papi lib"
            OFF
    )
    if (REBUILD_PAPI)
        set(PAPISCRIPTPATH ../thirdparty)
        execute_process(COMMAND bash ${PAPISCRIPTPATH}/makeClean.sh WORKING_DIRECTORY ${PAPISCRIPTPATH})
        execute_process(COMMAND bash ${PAPISCRIPTPATH}/installPAPI.sh WORKING_DIRECTORY ${PAPISCRIPTPATH})
        message(STATUS "I have rebuilt PAPI")
    endif ()
    find_library(libPAPI libpapi.so ./thirdparty/papi_build/lib)
    message(STATUS "papiLibs= " ${libPAPI})
    set(LIBRARIES ${LIBRARIES} ${libPAPI})

endif ()
configure_file(
        "${PROJECT_SOURCE_DIR}/include/papi_config.h.in"
        "${PROJECT_BINARY_DIR}/include/papi_config.h"
)
#options of ray
if (NOT ENABLE_RAY)
    message(STATUS "I will NOT include ray or distributed features ")
    set(CANDY_RAY 0)
else ()
    set(CANDY_RAY 1)
    message(STATUS "I will include ray and distributed computing support, pls make sure you have corresponding libs")
    if (NOT DEFINED ENV{RAYPATH})
        set(ENV{RAYPATH} '/usr/lib')
        message(STATUS "auto set RAYPATH= " $ENV{RAYPATH})
    endif ()
    set(RAY_DIR $ENV{RAYPATH})
    set(LIBRARIES ${LIBRARIES} ${RAY_DIR}/libray_api.so)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rayHeader)
endif ()

configure_file(
        "${PROJECT_SOURCE_DIR}/include/ray_config.h.in"
        "${PROJECT_BINARY_DIR}/include/ray_config.h"
)
#set(CMAKE_CUDA_ARCHITECTURES 55)
# Set LOGGING_LEVEL Flag
#option of hdf5
if (NOT ENABLE_HDF5)
    message(STATUS "I will NOT include support for HDF5 file ")
    set(CANDY_HDF5 0)
else ()
    set(CANDY_HDF5 1)
    message(STATUS "I will include support for HDF5 file, please make sure hdf5 works")
    add_subdirectory(thirdparty/hdf5-1.14.3)
    set(LIBRARIES ${LIBRARIES} hdf5-static)
endif ()
configure_file(
        "${PROJECT_SOURCE_DIR}/include/hdf5_config.h.in"
        "${PROJECT_BINARY_DIR}/include/hdf5_config.h"
)

# Reset output directories for the rest of your project
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
#pytorch
#set(Torch_DIR "/home/tony/.local/lib/python3.10/site-packages/torch/share/cmake" )
# Log4cc
#find_package(Log4cxx REQUIRED)
#include_directories(${Log4cxx_INCLUDE_DIR})
#set(LIBRARIES ${LIBRARIES} ${Log4cxx_LIBRARY})


option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")


# Print all used include directories
message(STATUS "INCLUDE_DIRS:")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach (dir ${dirs})
    message(STATUS "  - ${dir}")
endforeach ()

message(STATUS " Libraries: ${LIBRARIES}")
#add_subdirectory(pytorchNN)
# Add Source Code


add_subdirectory(src)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

# Add Library
get_sources(CANDY_SOURCE_FILES)
get_headers(CANDY_HEADER_FILES)
add_library(CANDY SHARED ${CANDY_SOURCE_FILES} ${CANDY_HEADER_FILES} ${CMAKE_CURRENT_BINARY_DIR})

set_property(TARGET CANDY PROPERTY CXX_STANDARD 20)
target_include_directories(CANDY PUBLIC "include")
target_include_directories(CANDY PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(CANDY PUBLIC "thirdparty/papi_build/include")
target_include_directories(CANDY PUBLIC "thirdparty/")
target_include_directories(CANDY PUBLIC "thirdparty/SPTAG")
target_include_directories(CANDY PUBLIC "thirdparty/SPTAG/AnnService")
target_link_options(CANDY PUBLIC "-Wl,-rpath,./")
target_link_options(CANDY PUBLIC "-Wl,-rpath,./")
target_link_libraries(CANDY PUBLIC ${LIBRARIES})
install(TARGETS CANDY LIBRARY DESTINATION .)
#add_subdirectory(test)
#Add benchmarks with command
add_subdirectory(benchmark)
#Add tests (based on GTest suits)
#include(GoogleTest)

if (ENABLE_UNIT_TESTS)
    add_subdirectory(test)
endif ()

if (NOT ENABLE_PYBIND)
    message(STATUS "I will NOT build original python package PyCANDY")
    set(CANDY_PYBIND 0)
else ()
    message(STATUS "I will  build original python package PyCANDY")
    add_subdirectory(thirdparty/pybind11)
    pybind11_add_module(PyCANDY ${PROJECT_SOURCE_DIR}/src/PyCANDY.cpp)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
    target_link_libraries(PyCANDY PUBLIC ${LIBRARIES} CANDY ${TORCH_PYTHON_LIBRARY})
    install(TARGETS PyCANDY LIBRARY DESTINATION .)
    message(STATUS "Additional include torch_python: ${TORCH_PYTHON_LIBRARY}")
    set(CANDY_PYBIND 1)
    set_property(TARGET PyCANDY  PROPERTY CXX_STANDARD 20)
endif ()
configure_file(
        "${PROJECT_SOURCE_DIR}/include/pybind_config.h.in"
        "${PROJECT_BINARY_DIR}/include/pybind_config.h"
)
install(DIRECTORY "include" DESTINATION "/CANDY" COMPONENT CANDY)
# copy scripts
file(GLOB allCopyFiles "${PROJECT_SOURCE_DIR}/scripts/*")
file(COPY ${allCopyFiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/benchmark/scripts)
file(COPY ${allCopyFiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/benchmark/scripts)
message(STATUS "build path:" ${CMAKE_CURRENT_BINARY_DIR})
# copy files needed for real world matrix loader
set(source_directory "${CMAKE_SOURCE_DIR}/benchmark/datasets/")
set(destination_directory "${CMAKE_BINARY_DIR}/benchmark/datasets/")
file(COPY ${source_directory} DESTINATION ${destination_directory})
