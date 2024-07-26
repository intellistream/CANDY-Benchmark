function(build_spdk_libs THIRD_PARTY_ROOT)
    execute_process(COMMAND bash ${THIRD_PARTY_ROOT}/cloneSPDK.sh WORKING_DIRECTORY ${THIRD_PARTY_ROOT})
    message(STATUS "I have built SPDK")
endfunction()
function(find_spdk_paths SPDK_BUILD_DIR SPDK_INCLUDE_DIRS SPDK_LIBRARIES_OUT)
    # Find the SPDK library directory
    find_library(SPDK_LIB_NVME spdk*
            HINTS ${SPDK_BUILD_DIR}/lib
            NO_DEFAULT_PATH)
    find_library(SPDK_LIB_MAIN spdk
            HINTS ${SPDK_BUILD_DIR}/lib
            NO_DEFAULT_PATH)
    # Find pkg-config
    find_package(PkgConfig REQUIRED)
    set(ENV{PKG_CONFIG_PATH} ${SPDK_BUILD_DIR}/lib/pkgconfig)
    # Use pkg-config to find OpenSSL
    pkg_check_modules(OPENSSL REQUIRED openssl)
    # Use pkg-config to find SPDK
    # Set PKG_CONFIG_PATH to include SPDK .pc files
    #set(ENV{PKG_CONFIG_PATH} "${SPDK_BUILD_DIR}/lib/pkgconfig")
    #set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${CMAKE_SOURCE_DIR}/thirdparty/spdk/build/lib/pkgconfig")
    #pkg_check_modules(SPDK REQUIRED spdk)
    pkg_check_modules(SPDK REQUIRED spdk_nvme spdk_vmd)


        # Get the directory of the found library
        # Find OpenSSL
        find_package(OpenSSL REQUIRED)

        file(GLOB SPDK_SO_FILES "${SPDK_BUILD_DIR}/lib/*.so")
        file(GLOB DPDK_SO_FILES "${SPDK_BUILD_DIR}/../dpdk/build/lib/*.so")
        #set(SPDK_SO_FILES spdk_env_dpdk spdk_nvme)
        set(CRCPATH "${SPDK_BUILD_DIR}/lib/libspdk_spdk_tgt.so")
        set(UtilSub ${SPDK_BUILD_DIR}/lib/libspdk_util.a)
        message(CRC_LIB= ${CRCPATH})
        # Store the result in the specified variable
        set(${SPDK_LIBRARIES_OUT} ${SPDK_SO_FILES} ${DPDK_SO_FILES} ${OPENSSL_LIBRARIES} ${CRCPATH} ${ISAL_LIBRARY} PARENT_SCOPE)
        #message(SPDK_LIB= ${SPDK_LIBRARIES})

    # Find the SPDK include directory
    find_path(SPDK_INCLUDE_DIR spdk/env.h
            HINTS ${SPDK_BUILD_DIR}/include
            NO_DEFAULT_PATH)

    if (SPDK_INCLUDE_DIR)
        # Set the output variable to the found include directory
        set(${SPDK_INCLUDE_DIRS} ${SPDK_INCLUDE_DIR} PARENT_SCOPE)
    else ()
        message(FATAL_ERROR "SPDK include directory not found")
    endif ()
endfunction()