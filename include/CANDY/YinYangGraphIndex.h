/*! \file YinYangGraphIndex.h*/
//
// Created by tony on 04/01/24.
//

#ifndef CANDY_INCLUDE_CANDY_YINYANGGRAPHINDEX_H_
#define CANDY_INCLUDE_CANDY_YINYANGGRAPHINDEX_H_

#include <Utils/AbstractC20Thread.hpp>
#include <Utils/ConfigMap.hpp>
#include <memory>
#include <vector>
#include <Utils/IntelliTensorOP.hpp>
#include <faiss/IndexFlat.h>
#include <CANDY/AbstractIndex.h>
#include <CANDY/YinYangGraphIndex/YinYangGraph.h>
#include <CANDY/FlatIndex.h>
namespace CANDY {

/**
 * @ingroup  CANDY_lib_bottom The main body and interfaces of library function
 * @{
 */
class YinYangGraphIndex;
/**
 * @class YinYangGraphIndex CANDY/YinYangGraphIndex.h
 * @brief The class of indexing using a yinyang graph, store data as brutal force does, and preserve similarity in another tensor
 * @todo implement the delete and revise later
 * @note currently single thread, not yet on SSD
 * @note current heuristics
 * - gaurantee adjecent connectivity, p_{i,i+1}>0, p_{i-1,i}>0
 * - control the #edges as HNSW does, but simpler shrinking
 * - optional using attention function to insert, rather than using raw data
 * @note config parameters
 * - vecDim, the dimension of vectors, default 768, I64
 * - maxConnection, the max number of connections in the yinyang graph (for yang vertex of data), default 256, I64
 * - metricType, the type of AKNN metric, default L2, String
 * - cudaDevice, the cuda device for DCO, default -1 (none), I64
 * - DCOBatchSize, the batch size of internal distance comparison operation (DCO), default equal to -1, I64
 */
class YinYangGraphIndex : public FlatIndex {
 protected:
  torch::Tensor similarityTensor,rowNNZTensor;
  CANDY::YinYangGraph yyg;
  // torch::Tensor dbTensor;
  int64_t maxConnection = 256;
  int64_t encodeLen = 1;
  int64_t candidateTimes = 1;
  int64_t skeletonRows = 1000;
  std::string lshMatrixType = "gaussian";
  int64_t cudaDevice = -1;
  int64_t DCOBatchSize = -1;
  /**
  * @brief to generate the sampling indices of crs
  */
  void genCrsIndices(void);
  //initialVolume = 1000, expandStep = 100;
  /**
   * @brief the distance function pointer member
   * @note will select largest distance during the following sorting, please convert if your distance is 'minimal'
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @param idx the pointer to index
   * @return The distance tensor, must sized [q*n] and remain in cpu
   */
  torch::Tensor (*distanceFunc)(torch::Tensor &db, torch::Tensor &query, int64_t cudaDev, YinYangGraphIndex *idx);
  /**
   * @brief the distance function of inner product
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @param idx the pointer to index
   * @return The distance tensor, must sized [q*n], will in GPU if cuda is valid
   */
  static torch::Tensor distanceIP(torch::Tensor &db, torch::Tensor &query, int64_t cudaDev, YinYangGraphIndex *idx);
  /**
   * @brief the distance function of L2
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @param idx the pointer to index
   * @return The distance tensor, must sized [q*n], will in GPU if cuda is valid
   */
  static torch::Tensor distanceL2(torch::Tensor &db, torch::Tensor &query, int64_t cudaDev, YinYangGraphIndex *idx);

  /**
  * @brief The iniline load the initial tensors of a data base
  * @note This will set up the skeleton of similarity matrix, no attention computation, but just the similarity
  * @param t the tensor, some index need to be single row
  * @return bool whether the loading is successful
  */
  bool loadInitialTensorInline(torch::Tensor &t);
 public:
  int64_t gpuComputingUs = 0;
  int64_t gpuCommunicationUs = 0;
  YinYangGraphIndex() {

  }

  ~YinYangGraphIndex() {

  }
  /**
   * @brief set the index-specific config related to one index
   * @param cfg the config of this class
   * @return bool whether the configuration is successful
   */
  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  /**
   * @brief insert a tensor
   * @param t the tensor, accept multiple rows
   * @return bool whether the insertion is successful
   */
  virtual bool insertTensor(torch::Tensor &t);

  /**
   * @brief search the k-NN of a query tensor, return the result tensors
   * @param t the tensor, allow multiple rows
   * @param k the returned neighbors
   * @return std::vector<torch::Tensor> the result tensor for each row of query
   */
  virtual std::vector<torch::Tensor> searchTensor(torch::Tensor &q, int64_t k);
  /**
   * @brief to get the internal statistics of this index
   * @return the statistics results in ConfigMapPtr
   */
  virtual INTELLI::ConfigMapPtr getIndexStatistics(void);
};

/**
 * @ingroup  CANDY_lib_bottom
 * @typedef YinYangGraphIndexPtr
 * @brief The class to describe a shared pointer to @ref  YinYangGraphIndex

 */
typedef std::shared_ptr<class CANDY::YinYangGraphIndex> YinYangGraphIndexPtr;
/**
 * @ingroup  CANDY_lib_bottom
 * @def newYinYangGraphIndex
 * @brief (Macro) To creat a new @ref  YinYangGraphIndex shared pointer.
 */
#define newYinYangGraphIndex std::make_shared<CANDY::YinYangGraphIndex>
}
/**
 * @}
 */

#endif //INTELLISTREAM_INCLUDE_CPPALGOS_ABSTRACTCPPALGO_H_
