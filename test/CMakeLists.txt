
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Werror=return-type ")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
enable_testing()
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/torchscripts DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/datasets
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/src/CL
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
macro(add_catch_test appName SOURCE_FILES SOURCE_LIBS)
    get_property(SOURCE_FILES_LOCAL GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP")
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
    add_executable(${appName} ${SOURCE_FILES})
    target_link_libraries(${appName} ${SOURCE_LIBS})
    target_link_options(${appName} PRIVATE "-Wl,-rpath,./")
    add_test(
            NAME catch_${appName}
            COMMAND $<TARGET_FILE:${appName}> --success
    )
endmacro()
macro(add_catch_test_ray appName SOURCE_FILES SOURCE_LIBS)
    get_property(SOURCE_FILES_LOCAL GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP")
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
    add_library(${appName} SHARED ${SOURCE_FILES})
    target_link_libraries(${appName} ${SOURCE_LIBS})
    target_link_options(${appName} PRIVATE "-Wl,-rpath,./")
    add_test(
            NAME catch_${appName}
            COMMAND $<TARGET_FILE:${appName}> --success
    )
endmacro()
add_catch_test(hnsw_test SystemTest/HNSWTest.cpp CANDYBENCH)
add_catch_test(cpp_test SystemTest/SimpleTest.cpp CANDYBENCH)
add_catch_test(flatIndex_test SystemTest/FlatIndexTest.cpp CANDYBENCH)
add_catch_test(flatAMMIPIndex_test SystemTest/FlatAMMIPIndexTest.cpp CANDYBENCH)
add_catch_test(flatAMMIPObjIndex_test SystemTest/FlatAMMIPObjIndexTest.cpp CANDYBENCH)
add_catch_test(ppIndex_test SystemTest/ParallelPartitionIndexTest.cpp CANDYBENCH)
add_catch_test(pqIndex_Test SystemTest/PQIndexTest.cpp CANDYBENCH)
add_catch_test(onlinePQ_test SystemTest/OnlinePQIndexTest.cpp CANDYBENCH)
add_catch_test(onlineIVFLSH_test SystemTest/OnlineIVFLSHIndexTest.cpp CANDYBENCH)
add_catch_test(yinYang_test SystemTest/YinYangGraphIndexTest.cpp CANDYBENCH)
add_catch_test(yinYangSimple_test SystemTest/YinYangGraphSimpleIndexTest.cpp CANDYBENCH)
add_catch_test(nnDescent_test SystemTest/NNDescentIndexTest.cpp CANDYBENCH)
add_catch_test(kdTree_test SystemTest/KdTreeTest.cpp CANDYBENCH)
add_catch_test(dpgIndex_test SystemTest/DPGIndexTest.cpp CANDYBENCH)
add_catch_test(lshAPGIndex_test SystemTest/LSHAPGIndexTest.cpp CANDYBENCH)
add_catch_test(gravistarIndex_test SystemTest/GravistarIndexTest.cpp CANDYBENCH)
add_catch_test(hnswTensor_test SystemTest/HNSWTensorTest.cpp CANDYBENCH)
if (ENABLE_SPTAG)
    add_catch_test(sptagIndex_test SystemTest/SPTAGIndexTest.cpp CANDYBENCH)
endif ()

if (ENABLE_OPENCL)
    add_catch_test(cl_test SystemTest/CLTest.cpp CANDYBENCH)
endif ()
if (ENABLE_RAY)
    add_catch_test(dp_test SystemTest/DistributedPartitionIndexTest.cpp CANDYBENCH)
    add_catch_test_ray(dp_testL SystemTest/DistributedPartitionIndexTest.cpp CANDYBENCH)
endif ()

