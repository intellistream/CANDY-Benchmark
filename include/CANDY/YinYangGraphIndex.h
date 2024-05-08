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
namespace CANDY {

/**
 * @ingroup  CANDY_lib_bottom The main body and interfaces of library function
 * @{
 */
/**
 * @class YinYangGraphIndex CANDY/YinYangGraphIndex.h
 * @brief The class of indexing using a yinyang graph, first use LSH to roughly locate the range of a tensor, then
 * search it in the linked yinyanggraph
 * @todo implement the delete and revise later
 * @note currently single thread
 * @note config parameters
 * - vecDim, the dimension of vectors, default 768, I64
 * - maxConnection, the max number of connections in the yinyang graph (for yang vertex of data), default 256, I64
 * - candidateTimes, the times of k to determine minimum candidates, default 1 ,I64
 * - numberOfBuckets, the number of first titer buckets, default 4096, I64, suggest 2^n
 * - encodeLen, the length of LSH encoding, in bytes, default 1, I64
 * - metricType, the type of AKNN metric, default L2, String
 * - lshMatrixType, the type of lsh matrix, default gaussian, String
    * - gaussian means a N(0,1) LSH matrix
    * - random means a random matrix where each value ranges from -0.5~0.5
 * - useCRS, whether or not use column row sampling in projecting the vector, 0 (No), I64
    * - further trade off of accuracy v.s. efficiency
 * - CRSDim, the dimension which are not pruned by crs, 1/10 of vecDim, I64
 * - redoCRSIndices, whether or not re-generate the indices of CRS, 0 (No), I64
 */
class YinYangGraphIndex : public AbstractIndex {
 protected:
  INTELLI::ConfigMapPtr myCfg = nullptr;
  CANDY::YinYangGraph yyg;
  // torch::Tensor dbTensor;
  int64_t vecDim = 0;
  int64_t maxConnection = 0;
  int64_t numberOfBuckets = 4096;
  int64_t encodeLen = 1;
  int64_t candidateTimes = 1;
  int64_t useCRS = 0;
  int64_t CRSDim = 1;
  int64_t bucketsLog2 = 0;
  int64_t redoCRSIndices = 0;
  std::string lshMatrixType = "gaussian";
  std::vector<uint8_t> encodeSingleRow(torch::Tensor &tensor, uint64_t *bucket);
  torch::Tensor rotationMatrix, crsIndices;
  torch::Tensor randomProjection(torch::Tensor &a);
  /**
  * @brief to generate the sampling indices of crs
  */
  void genCrsIndices(void);
  //initialVolume = 1000, expandStep = 100;
 public:
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
    * @brief thw column row sampling to compute approximate matrix multiplication
    * @param A the left side matrix
    * @param B the right side matrix
    * @param idx the indices of sampling
    * @param _crsDim the dimension of preserved dimensions
    */
  static torch::Tensor crsAmm(torch::Tensor &A, torch::Tensor &B, torch::Tensor &indices);
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
