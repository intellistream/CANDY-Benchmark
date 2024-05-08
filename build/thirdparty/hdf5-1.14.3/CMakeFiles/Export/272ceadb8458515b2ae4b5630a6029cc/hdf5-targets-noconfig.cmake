#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hdf5-static" for configuration ""
set_property(TARGET hdf5-static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(hdf5-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libhdf5.a"
  )

list(APPEND _cmake_import_check_targets hdf5-static )
list(APPEND _cmake_import_check_files_for_hdf5-static "${_IMPORT_PREFIX}/lib/libhdf5.a" )

# Import target "hdf5-shared" for configuration ""
set_property(TARGET hdf5-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(hdf5-shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libhdf5.so.310.3.0"
  IMPORTED_SONAME_NOCONFIG "libhdf5.so.310"
  )

list(APPEND _cmake_import_check_targets hdf5-shared )
list(APPEND _cmake_import_check_files_for_hdf5-shared "${_IMPORT_PREFIX}/lib/libhdf5.so.310.3.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
