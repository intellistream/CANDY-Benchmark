//
// Created by tony on 25/05/23.
//

#include "Algorithms/BucketedFlatIndex/BucketedFlatIndex.h"
#include <Algorithms/BufferedCongestionDropIndex.h>
#include <Algorithms/CongestionDropIndex.h>
#include <Algorithms/KNNSearch.hpp>
#include <Algorithms/IndexTable.h>
#include <Algorithms/ParallelPartitionIndex.h>
#include <include/opencl_config.h>
#include <include/ray_config.h>
#if CANDY_CL == 1
//#include <CPPAlgos/CLMMCPPAlgo.h>
#endif
#if CANDY_RAY == 1
#include <Algorithms/DistributedPartitionIndex.h>
#endif
namespace CANDY {
CANDY::IndexTable::IndexTable() {
  indexMap["null"] = newAbstractIndex();
  indexMap["flat"] = newFlatIndex();
  indexMap["bucketedFlat"] = newBucketedFlatIndex();
  indexMap["parallelPartition"] = newParallelPartitionIndex();
  indexMap["congestionDrop"] = newCongestionDropIndex();
  indexMap["bufferedCongestionDrop"] = newBufferedCongestionDropIndex();
#if CANDY_CL == 1
  // indexMap["cl"] = newCLMMCPPAlgo();
#endif
#if CANDY_RAY == 1
  indexMap["distributedPartition"] = newDistributedPartitionIndex();
#endif
}
}  // namespace Algorithms
