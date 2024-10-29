
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

add_catch_test(flatIndex_test SystemTest/FlatIndexTest.cpp CANDY)
add_catch_test(ppIndex_test SystemTest/ParallelPartitionIndexTest.cpp CANDY)


if (ENABLE_OPENCL)
    add_catch_test(cl_test SystemTest/CLTest.cpp CANDY)
endif ()
if (ENABLE_RAY)
    add_catch_test(dp_test SystemTest/DistributedPartitionIndexTest.cpp CANDY)
    add_catch_test_ray(dp_testL SystemTest/DistributedPartitionIndexTest.cpp CANDY)
endif ()

