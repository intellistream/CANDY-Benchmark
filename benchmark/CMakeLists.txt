# Copyright (C) 2021 by the INTELLI team (https://github.com/intellistream)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (BUILD_BENCHMARKS)
    message("---Building benchmarks ${source_dir}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark)
    include_directories("${source_dir}/include/")
    include_directories("include/")
    #add_executable(onlineInsert "src/OnlineInsert.cpp")
    #target_link_libraries(onlineInsert CANDYBENCH)
    #add_executable(offlineBuild "src/OfflineBuild.cpp")
    ##target_link_libraries(offlineBuild CANDYBENCH)
    #add_executable(multiRW "src/MultiRW.cpp")
    #target_link_libraries(multiRW CANDYBENCH)
    #add_executable(onlineInsert2 "src/OnlineInsertLSHAPG.cpp")
    #target_link_libraries(onlineInsert2 CANDYBENCH)
    #add_executable(onlineCUD "src/OnlineCUD.cpp")
    #target_link_libraries(onlineCUD CANDYBENCH)
    if (ENABLE_RAY)
        add_library(onlineInsertL SHARED "src/OnlineInsert.cpp")
        target_link_libraries(onlineInsertL CANDYBENCH)
        target_link_options(onlineInsert PRIVATE "-Wl,-rpath,./")
        target_link_options(onlineInsertL PRIVATE "-Wl,-rpath,./")
        #
        add_library(offlineBuildL SHARED "src/OfflineBuild.cpp")
        target_link_libraries(offlineBuildL CANDYBENCH)
        target_link_options(offlineBuild PRIVATE "-Wl,-rpath,./")
        target_link_options(offlineBuildL PRIVATE "-Wl,-rpath,./")

        add_library(multiRWL SHARED "src/MultiRW.cpp")
        target_link_libraries(multiRWL CANDYBENCH)
        target_link_options(multiRWL PRIVATE "-Wl,-rpath,./")
        target_link_options(multiRW PRIVATE "-Wl,-rpath,./")

        add_library(onlineCUDL SHARED "src/OnlineCUD.cpp")
        target_link_libraries(onlineCUDL CANDYBENCH)
        target_link_options(onlineCUDL PRIVATE "-Wl,-rpath,./")
        target_link_options(onlineCUDL PRIVATE "-Wl,-rpath,./")

    endif ()
    #the onlineInsert program

    #add_executable(step1AMME2ESaveMtxToPt "torchscripts/VQ/step1AMME2ESaveMtxToPt.cpp")
    #target_link_libraries(step1AMME2ESaveMtxToPt IntelliStream)
    #add_executable(step1PCACCASaveMtxToPt "torchscripts/VQ/step1PCACCASaveMtxToPt.cpp")
    #target_link_libraries(step1PCACCASaveMtxToPt IntelliStream)
    #add_executable(step3TestVQ "torchscripts/VQ/step3TestVQ.cpp")
    #target_link_libraries(step3TestVQ IntelliStream)
    #     add_executable(VQTrainSIFT "torchscripts/VQ/PCA/VQTrainSIFT.cpp")
    #     target_link_libraries(VQTrainSIFT IntelliStream)

    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/perfLists
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/pythonTest.py
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    #file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/torchscripts DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    #     set_target_properties(benchmarkPCA PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/scripts/PCA)
    #     set_target_properties(benchmarkCCA PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/scripts/CCA)
    #set_target_properties(step1AMME2ESaveMtxToPt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/torchscripts/VQ)
    #set_target_properties(step1PCACCASaveMtxToPt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/torchscripts/VQ)
    #set_target_properties(step3TestVQ PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/torchscripts/VQ)
    #     set_target_properties(VQTrainSIFT PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/scripts/PCA/results/pca_nonstreaming/blockLRA/1)

    file(COPY ${PROJECT_SOURCE_DIR}/src/CL
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
    message("---Done building benchmarks")
else (BUILD_BENCHMARKS)
    message("---Not building benchmarks!")
endif (BUILD_BENCHMARKS)
