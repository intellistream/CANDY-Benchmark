add_sources(
        CANDYObject.cpp
        AbstractIndex.cpp
        BucketedFlatIndex/BucketedFlatIndex.cpp
        BufferedCongestionDropIndex.cpp
        FlatIndex.cpp
        FlatAMMIPIndex.cpp
        FlatAMMIPObjIndex.cpp
        ParallelPartitionIndex.cpp
        OnlinePQIndex.cpp
        OnlineIVFLSHIndex.cpp
        OnlineIVFL2HIndex.cpp
        IndexTable.cpp
        PQIndex.cpp
        HNSWNaiveIndex.cpp
        CongestionDropIndex.cpp
        NNDescentIndex.cpp
)
add_subdirectory(ParallelPartitionIndex)
add_subdirectory(PQIndex)
add_subdirectory(OnlinePQIndex)
add_subdirectory(HNSWNaive)
add_subdirectory(CongestionDropIndex)
add_subdirectory(FlannIndex)
if (ENABLE_RAY)
    add_subdirectory(DistributedPartitionIndex)
    add_sources(DistributedPartitionIndex.cpp)
endif ()