add_subdirectory(Utils)
add_subdirectory(Algorithms)
add_subdirectory(DataLoader)
if (DEFINED ENV{CANDY_CL})
    add_sources(
            CLContainer.cpp
    )
endif ()
include_directories("../include/")
