# Copyright (C) 2021 by the IntelliStream team (https://github.com/intellistream)

cmake_minimum_required(VERSION 3.14)
project(SESAME CXX)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# Custom CMake find instructions and macros
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(cmake/macros.cmake)
include(cmake/default.cmake)
include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Optimization Flags
set(CMAKE_CXX_FLAGS "-g -std=c++20 -Wall -fconcepts-diagnostics-depth=2 -fopenmp -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -DNO_RACE_CHECK -DSESAME_DEBUG_MODE=1 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-ignored-qualifiers -Wno-sign-compare -O3 -DNDEBUG -flto=auto")

# Set LOGGING_LEVEL Flag
if (SESAME_LOGGING_LEVEL)
    get_log_level_value(SESAME_LOGGING_VALUE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSESAME_LOGGING_LEVEL=${SESAME_LOGGING_VALUE}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSESAME_LOGGING_LEVEL=${SESAME_LOGGING_VALUE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DSESAME_LOGGING_LEVEL=${SESAME_LOGGING_VALUE}")
else (SESAME_LOGGING_LEVEL)
    message("---Everything will be logged")
endif (SESAME_LOGGING_LEVEL)

message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")

option(ENABLE_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_TESTS}")

if (ENABLE_TESTS)
    #enable_testing()
    # Google Test
    #FetchContent_Declare(
     #       googletest
      #      GIT_REPOSITORY https://github.com/google/googletest.git
       #     GIT_TAG v1.14.0
    #)
    #set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    #FetchContent_MakeAvailable(googletest)
    #include(GoogleTest)
endif ()


# set(Boost_DEBUG 1)
#set(BOOST_INCLUDE_LIBRARIES timer lockfree)
#set(BOOST_ENABLE_CMAKE ON)
#FetchContent_Declare(
#  Boost
#  URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
#  URL_MD5 893b5203b862eb9bbd08553e24ff146a
#  DOWNLOAD_NO_EXTRACT FALSE
#  EXCLUDE_FROM_ALL
#)
#FetchContent_MakeAvailable(Boost)

set(GFLAGS_BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(GFLAGS_BUILD_gflags_nothreads_LIB ON)
set(GFLAGS_BUILD_gflags_LIB OFF CACHE BOOL "")
# for darwin
if (APPLE)
    set(GFLAGS_INTTYPES_FORMAT "C99" CACHE STRING "")
endif ()
#add_subdirectory(third_party/gflags)

find_package(OpenMP)

# Print all used include directories
message(STATUS "INCLUDE_DIRS:")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach (dir ${dirs})
    message(STATUS "  - ${dir}")
endforeach ()

# Add Source Code
add_subdirectory(src)

# Add Library
get_source_sesame(sesame_SOURCE_FILES)
get_header_sesame(sesame_HEADER_FILES)
add_library(sesame SHARED ${sesame_SOURCE_FILES} ${sesame_HEADER_FILES} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(sesame PUBLIC "include")
target_include_directories(sesame PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
#target_link_libraries(sesame PUBLIC ${LIBRARIES} pthread Boost::timer Boost::lockfree)
#target_link_libraries(sesame PRIVATE gflags)

#Add benchmarks with command
#add_subdirectory(benchmark)

#Add tests (based on GTest suits)
#add_subdirectory(test)

install(DIRECTORY "include" DESTINATION "/sesame" COMPONENT SESAME)

option(ENABLE_PYTHON "Enable Python bindings" OFF)
message(STATUS "Enable Python bindings: ${ENABLE_PYTHON}")
if (ENABLE_PYTHON)
    add_subdirectory(third_party/pybind11)
    pybind11_add_module(pysame python/src/Pysame.cpp)
    target_include_directories(pysame PRIVATE "include")
    target_link_libraries(pysame PUBLIC sesame)
    set_target_properties(pysame PROPERTIES 
        BUILD_RPATH "\$ORIGIN/"
        INSTALL_RPATH "\$ORIGIN/"
    )
endif ()
