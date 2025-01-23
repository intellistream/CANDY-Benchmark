add_subdirectory(Utils)

add_subdirectory(CANDY)
#add_subdirectory(DataLoader)
#gcc options
if (DEFINED ENV{CANDY_CL})
    add_sources(
            CLContainer.cpp
    )
endif ()
add_sources(
        #torchBounds.cpp
)
include_directories("../include/")
