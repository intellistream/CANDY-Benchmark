add_sources(
        AbstractIndex.cpp
        BucketedFlatIndex/BucketedFlatIndex.cpp
        BufferedCongestionDropIndex.cpp
        FlatIndex.cpp
        ParallelPartitionIndex.cpp
        IndexTable.cpp
        CongestionDropIndex.cpp
)
add_subdirectory(ParallelPartitionIndex)
add_subdirectory(CongestionDropIndex)
if (ENABLE_RAY)
    add_subdirectory(DistributedPartitionIndex)
    add_sources(DistributedPartitionIndex.cpp)
endif ()