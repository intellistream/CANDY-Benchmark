//
// Created by tony on 25/05/23.
//

#include "Algorithms/BucketedFlatIndex/BucketedFlatIndex.h"
#include <Algorithms/BufferedCongestionDropIndex.h>
#include <Algorithms/CongestionDropIndex.h>
#include <Algorithms/FlatAMMIPIndex.h>
#include <Algorithms/FlatAMMIPObjIndex.h>
#include <Algorithms/FlatIndex.h>
#include <Algorithms/HNSWNaiveIndex.h>
#include <Algorithms/IndexTable.h>
#include <Algorithms/NNDescentIndex.h>
#include <Algorithms/OnlineIVFL2HIndex.h>
#include <Algorithms/OnlineIVFLSHIndex.h>
#include <Algorithms/OnlinePQIndex.h>
#include <Algorithms/PQIndex.h>
#include <Algorithms/ParallelPartitionIndex.h>
#include <include/opencl_config.h>
#include <include/ray_config.h>
#include <include/sptag_config.h>
#if CANDY_CL == 1
//#include <CPPAlgos/CLMMCPPAlgo.h>
#endif
#if CANDY_RAY == 1
#include <Algorithms/DistributedPartitionIndex.h>
#endif
#if CANDY_SPTAG == 1
#include <Algorithms/SPTAGIndex.h>
#endif
namespace CANDY {
CANDY::IndexTable::IndexTable() {
  indexMap["null"] = newAbstractIndex();
  indexMap["flat"] = newFlatIndex();
  indexMap["flatAMMIP"] = newFlatAMMIPIndex();
  indexMap["flatAMMIPObj"] = newFlatAMMIPObjIndex();
  indexMap["bucketedFlat"] = newBucketedFlatIndex();
  indexMap["parallelPartition"] = newParallelPartitionIndex();
  indexMap["onlinePQ"] = newOnlinePQIndex();
  indexMap["onlineIVFLSH"] = newOnlineIVFLSHIndex();
  indexMap["onlineIVFL2H"] = newOnlineIVFL2HIndex();
  indexMap["PQ"] = newPQIndex();
  indexMap["HNSWNaive"] = newHNSWNaiveIndex();
  indexMap["NSW"] = newNSWIndex();
  indexMap["yinYang"] = newYinYangGraphIndex();
  indexMap["yinYangSimple"] = newYinYangGraphSimpleIndex();
  indexMap["congestionDrop"] = newCongestionDropIndex();
  indexMap["bufferedCongestionDrop"] = newBufferedCongestionDropIndex();
  indexMap["nnDescent"] = newNNDescentIndex();
  indexMap["LSHAPG"] = newLSHAPGIndex();
#if CANDY_CL == 1
  // indexMap["cl"] = newCLMMCPPAlgo();
#endif
#if CANDY_RAY == 1
  indexMap["distributedPartition"] = newDistributedPartitionIndex();
#endif
#if CANDY_SPTAG == 1
  indexMap["SPTAG"] = newSPTAGIndex();
#endif
}
}  // namespace Algorithms
