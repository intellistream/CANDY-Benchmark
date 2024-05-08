# Install script for directory: /home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/lib")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/tony/projects/CANDY/build/thirdparty/hdf5-1.14.3/src/H5FDsubfiling/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "headers" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/hdf5.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5api_adpt.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5encode.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5public.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Apublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5ACpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Cpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Dpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Epubgen.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Epublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5ESdevelop.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5ESpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Fpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDcore.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDdevelop.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDdirect.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDfamily.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDhdfs.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDlog.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDmirror.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDmpi.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDmpio.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDmulti.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDonion.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDros3.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDs3comms.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDsec2.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDsplitter.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDstdio.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDwindows.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDsubfiling/H5FDsubfiling.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5FDsubfiling/H5FDioc.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Gpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Idevelop.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Ipublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Ldevelop.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Lpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Mpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5MMpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Opublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Ppublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5PLextern.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5PLpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Rpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Spublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Tdevelop.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Tpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5TSdevelop.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5VLconnector.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5VLconnector_passthru.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5VLnative.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5VLpassthru.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5VLpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Zdevelop.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Zpublic.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5Epubgen.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5version.h"
    "/home/tony/projects/CANDY/thirdparty/hdf5-1.14.3/src/H5overflow.h"
    "/home/tony/projects/CANDY/build/thirdparty/hdf5-1.14.3/src/H5pubconf.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libraries" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/tony/projects/CANDY/build/thirdparty/hdf5-1.14.3/bin/libhdf5.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libraries" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/tony/projects/CANDY/build/thirdparty/hdf5-1.14.3/bin/libhdf5.so.310.3.0"
    "/home/tony/projects/CANDY/build/thirdparty/hdf5-1.14.3/bin/libhdf5.so.310"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libhdf5.so.310.3.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libhdf5.so.310"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libraries" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/tony/projects/CANDY/build/thirdparty/hdf5-1.14.3/bin/libhdf5.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libhdf5.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libhdf5.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libhdf5.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libraries" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/tony/projects/CANDY/build/thirdparty/hdf5-1.14.3/CMakeFiles/hdf5.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libraries" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "/home/tony/projects/CANDY/build/thirdparty/hdf5-1.14.3/CMakeFiles/h5cc")
endif()

