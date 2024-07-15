macro(get_log_level_value SESAME_LOGGING_VALUE)
    if (${SESAME_LOGGING_LEVEL} STREQUAL "TRACE")
        message("-- Log level is set to TRACE!")
        set(SESAME_LOGGING_VALUE 6)
    elseif (${SESAME_LOGGING_LEVEL} STREQUAL "DEBUG")
        message("-- Log level is set to DEBUG!")
        set(SESAME_LOGGING_VALUE 5)

    elseif (${SESAME_LOGGING_LEVEL} STREQUAL "INFO")
        message("-- Log level is set to INFO!")
        set(SESAME_LOGGING_VALUE 4)
    elseif (${SESAME_LOGGING_LEVEL} STREQUAL "WARN")
        message("-- Log level is set to WARN!")
        set(SESAME_LOGGING_VALUE 3)

    elseif (${SESAME_LOGGING_LEVEL} STREQUAL "ERROR")
        message("-- Log level is set to ERROR!")
        set(SESAME_LOGGING_VALUE 2)

    elseif (${SESAME_LOGGING_LEVEL} STREQUAL "FATAL_ERROR")
        message("-- Log level is set to FATAL_ERROR!")
        set(SESAME_LOGGING_VALUE 1)

    else ()
        message(WARNING "-- Could not set SESAME_LOGGING_VALUE as ${SESAME_LOGGING_LEVEL} did not equal any logging level!!!  Defaulting to debug!")
        set(SESAME_LOGGING_VALUE 5)
    endif ()
endmacro(get_log_level_value SESAME_LOGGING_VALUE)

macro(add_source PROP_NAME SOURCE_FILES)
    set(SOURCE_FILES_ABSOLUTE)
    foreach (it ${SOURCE_FILES})
        get_filename_component(ABSOLUTE_PATH ${it} ABSOLUTE)
        set(SOURCE_FILES_ABSOLUTE ${SOURCE_FILES_ABSOLUTE} ${ABSOLUTE_PATH})
    endforeach ()

    get_property(OLD_PROP_VAL GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP")
    set_property(GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP" ${SOURCE_FILES_ABSOLUTE} ${OLD_PROP_VAL})
endmacro()

macro(add_source_sesame)
    add_source(sesame "${ARGN}")
endmacro()


macro(get_source PROP_NAME SOURCE_FILES)
    get_property(SOURCE_FILES_LOCAL GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP")
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
endmacro()

macro(get_source_sesame SOURCE_FILES)
    get_source(sesame SOURCE_FILES_LOCAL)
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
endmacro()

macro(get_header_sesame HEADER_FILES)
    file(GLOB_RECURSE ${HEADER_FILES} "include/*.h" "include/*.hpp")
endmacro()