cmake_minimum_required(VERSION 3.21)

project(puck)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_BINARY_DIR}/third_libs)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -w -g0 -Ofast -fPIC")
set(CMAKE_C_FLAGS "-std=c++11 -w -g0 -Ofast -fPIC")
#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-DFINTEGER=int)

add_compile_options(-fopenmp -lpthread -lcrypto -lrt -lgfortran -ldl -lz -fPIC -rdynamic)
add_compile_options(-Ofast -g -pipe -fPIC -march=native -w)

include(${PROJECT_SOURCE_DIR}/cmake/gflags.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/glog.cmake)
find_package(OpenMP REQUIRED)
find_package(MKL REQUIRED)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/output)
add_subdirectory(puck)
add_subdirectory(demo)
add_subdirectory(tools)




