//
// Created by tony on 25/05/23.
//

#include <CANDY/BucketedFlatIndex.h>
#include <CANDY/BufferedCongestionDropIndex.h>
#include <CANDY/CongestionDropIndex.h>
#include <CANDY/DPGIndex.h>
#include <CANDY/FaissIndex.h>
#include <CANDY/FlannIndex.h>
#include <CANDY/FlatAMMIPIndex.h>
#include <CANDY/FlatAMMIPObjIndex.h>
#include <CANDY/FlatIndex.h>
#include <CANDY/HNSWNaiveIndex.h>
#include <CANDY/IndexTable.h>
#include <CANDY/NNDescentIndex.h>
#include <CANDY/OnlineIVFL2HIndex.h>
#include <CANDY/OnlineIVFLSHIndex.h>
#include <CANDY/OnlinePQIndex.h>
#include <CANDY/PQIndex.h>
#include <CANDY/ParallelPartitionIndex.h>
#include <CANDY/YinYangGraphIndex.h>
#include <CANDY/YinYangGraphSimpleIndex.h>
#include <include/opencl_config.h>
#include <include/ray_config.h>
#include <include/spdk_config.h>
#if CANDY_CL == 1
//#include <CPPAlgos/CLMMCPPAlgo.h>
#endif
#if CANDY_RAY == 1
#include <CANDY/DistributedPartitionIndex.h>
#endif
#if CANDY_SPDK == 1
#include <CANDY/FlatSSDGPUIndex.h>
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
  indexMap["faiss"] = newFaissIndex();
  indexMap["yinYang"] = newYinYangGraphIndex();
  indexMap["yinYangSimple"] = newYinYangGraphSimpleIndex();
  indexMap["congestionDrop"] = newCongestionDropIndex();
  indexMap["bufferedCongestionDrop"] = newBufferedCongestionDropIndex();
  indexMap["nnDescent"] = newNNDescentIndex();
  indexMap["Flann"] = newFlannIndex();
  indexMap["DPG"] = newDPGIndex();
#if CANDY_CL == 1
  // indexMap["cl"] = newCLMMCPPAlgo();
#endif
#if CANDY_RAY == 1
  indexMap["distributedPartition"] = newDistributedPartitionIndex();
#endif
#if CANDY_SPDK == 1
  indexMap["flatSSDGPU"] = newFlatSSDGPUIndex();
#endif
}
}  // namespace CANDY
