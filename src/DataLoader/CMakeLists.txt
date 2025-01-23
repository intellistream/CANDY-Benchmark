add_sources(
        AbstractDataLoader.cpp
        DataLoaderTable.cpp
        RandomDataLoader.cpp
        FVECSDataLoader.cpp
        ZipfDataLoader.cpp
        ExpFamilyDataLoader.cpp
        RBTDataLoader.cpp
)
if (ENABLE_HDF5)
    add_sources(HDF5DataLoader.cpp)
endif ()