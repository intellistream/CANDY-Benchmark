add_sources(
        CANDYObject.cpp
        AbstractIndex.cpp
        BucketedFlatIndex.cpp
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
        FaissIndex.cpp
        YinYangGraphIndex.cpp
        YinYangGraphSimpleIndex.cpp
        CongestionDropIndex.cpp
        NNDescentIndex.cpp
        FlannIndex.cpp
        DPGIndex.cpp
        Cuda/mm.cu
)
add_subdirectory(HashingModels)
add_subdirectory(ParallelPartitionIndex)
add_subdirectory(PQIndex)
add_subdirectory(OnlinePQIndex)
add_subdirectory(HNSWNaive)
add_subdirectory(YinYangGraphIndex)
add_subdirectory(CongestionDropIndex)
add_subdirectory(FlannIndex)
if (ENABLE_RAY)
    add_subdirectory(DistributedPartitionIndex)
    add_sources(DistributedPartitionIndex.cpp)
endif ()
if (ENABLE_SPDK)
    add_sources(
            FlatSSDGPUIndex.cpp
    )
    add_subdirectory(FlatSSDGPUIndex)
endif ()
if (CANDY_CUDA)
    message(WARNING "Cuda feature is experimental, and it is enabled")
    add_sources(
            Cuda/mm.cu
    )
endif ()