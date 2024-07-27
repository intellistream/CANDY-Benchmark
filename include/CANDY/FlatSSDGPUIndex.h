/*! \file FlatSSDGPUIndex.h*/
//
// Created by tony on 04/01/24.
//

#ifndef CANDY_INCLUDE_CANDY_FlatSSDGPUIndex_H_
#define CANDY_INCLUDE_CANDY_FlatSSDGPUIndex_H_
#include <include/spdk_config.h>
#include <Utils/AbstractC20Thread.hpp>
#include <Utils/ConfigMap.hpp>
#include <memory>
#include <vector>
#include <Utils/IntelliTensorOP.hpp>
#include <faiss/IndexFlat.h>
#include <CANDY/AbstractIndex.h>
#include <CANDY/FlatSSDGPUIndex/DiskMemBuffer.h>
namespace CANDY {

/**
 * @ingroup  CANDY_lib_bottom The main body and interfaces of library function
 * @{
 */
/**
 * @class FlatSSDGPUIndex CANDY/FlatSSDGPUIndex.h
 * @brief Similar to @ref FlatAMMIPObjectIndex, but runs on SSD and GPU for large scale
 * @note Only support inner product distance
 * @note currently single thread
 * @note config parameters
 * - vecDim, the dimension of vectors, default 768, I64
 * - initialVolume, the initial volume of inline database tensor, default 1000, I64
 * - SSDBufferSize, the size of memory-ssd buffer, in rows of vectors, default 1000, I64
 * - sketchSize, the sketch size of amm, default 10, I64
 * - DCOBatchSize, the batch size of internal distance comparison operation (DCO), default equal to ssdBufferSize, I64
 * - cudaDevice, the cuda device for DCO, default -1 (none), I64
 * - ammAlgo (Not used now), the amm algorithm used for compute distance, default mm, String, can be the following
    * - mm the original torch::matmul
    * - crs column row sampling
    * - smp-pca the smp-pca algorithm
 * @warning please run the benchmark/scripts/setupSPDK/drawTogether.py at generation path before using SSD
 */
class FlatSSDGPUIndex : public AbstractIndex {
 protected:
  INTELLI::ConfigMapPtr myCfg = nullptr;
  torch::Tensor dbTensor, objTensor;
  SPDKSSD ssd;
  PlainDiskMemBufferTU dmBuffer;
  int64_t ammType = 0;
  int64_t sketchSize = 10;
  int64_t DCOBatchSize = -1;
  int64_t SSDBufferSize = 1000;
  int64_t vecDim = 768;
  int64_t cudaDevice = -1;
  // Main function to process batches and find top_k closest vectors
  std::vector<int64_t> findTopKClosest(const torch::Tensor &query, int64_t top_k, int64_t batch_size);
  // torch::Tensor myMMInline(torch::Tensor &a, torch::Tensor &b, int64_t ss = 10);
  /**
  * @brief return a vector of tensors according to some index
  * @param idx the index, follow faiss's style, allow the KNN index of multiple queries
  * @param k the returned neighbors, i.e., will be the number of rows of each returned tensor
  * @return a vector of tensors, each tensor represent KNN results of one query in idx
  */
  virtual std::vector<torch::Tensor> getTensorByStdIdx(std::vector<int64_t> &idx, int64_t k);
  /**
   * @brief the distance function pointer member
   * @note will select largest distance during the following sorting, please convert if your distance is 'minimal'
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @return The distance tensor, must sized [q*n] and remain in cpu
   */
  torch::Tensor (*distanceFunc)(torch::Tensor db, torch::Tensor query, int64_t cudaDev);
  /**
   * @brief the distance function of inner product
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @return The distance tensor, must sized [q*n], will in GPU if cuda is valid
   */
  static torch::Tensor distanceIP(torch::Tensor db, torch::Tensor query, int64_t cudaDev);
  /**
   * @brief the distance function of L2
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @return The distance tensor, must sized [q*n], will in GPU if cuda is valid
   */
  static torch::Tensor distanceL2(torch::Tensor db, torch::Tensor query, int64_t cudaDev);
  // std::vector<faiss::idx_t> knnInline(torch::Tensor &query, int64_t k, int64_t distanceBatch = -1);
 public:
  FlatSSDGPUIndex() {

  }

  ~FlatSSDGPUIndex() {

  }
  /**
   * @brief some extra set-ups if the index has HPC fetures
   * @return bool whether the HPC set-up is successful
   */
  virtual bool startHPC();
  /**
    * @brief some extra termination if the index has HPC features
    * @return bool whether the HPC termination is successful
    */
  virtual bool endHPC();
  /**
    * @brief reset this index to inited status
    */
  virtual void reset();
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
   * @brief delete a tensor
   * @param t the tensor, recommend single row
   * @param k the number of nearest neighbors
   * @return bool whether the deleting is successful
   */
  virtual bool deleteTensor(torch::Tensor &t, int64_t k = 1);

  /**
   * @brief revise a tensor
   * @param t the tensor to be revised, recommend single row
   * @param w the revised value
   * @return bool whether the revising is successful
   */
  virtual bool reviseTensor(torch::Tensor &t, torch::Tensor &w);
  /**
   * @brief search the k-NN of a query tensor, return their index
   * @param t the tensor, allow multiple rows
   * @param k the returned neighbors
   * @return std::vector<faiss::idx_t> the index, follow faiss's order
   */
  //virtual std::vector<faiss::idx_t> searchIndex(torch::Tensor q, int64_t k);
  /**
   * @brief search the k-NN of a query tensor, return the result tensors
   * @param t the tensor, allow multiple rows
   * @param k the returned neighbors
   * @return std::vector<torch::Tensor> the result tensor for each row of query
   */
  virtual std::vector<torch::Tensor> searchTensor(torch::Tensor &q, int64_t k);

  /**
   * @brief return the size of ingested tensors
   * @return
   */
  virtual int64_t size() {
    return dmBuffer.size();
  }
  /**
   * @brief insert a string object
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @param strs the corresponding list of strings
   * @return bool whether the insertion is successful
   */
  // virtual bool insertStringObject(torch::Tensor &t, std::vector<std::string> &strs);

  /**
   * @brief  delete tensor along with its corresponding string object
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @param k the number of nearest neighbors
   * @return bool whether the delet is successful
   */
  //virtual bool deleteStringObject(torch::Tensor &t, int64_t k = 1);

  /**
 * @brief search the k-NN of a query tensor, return the linked string objects
 * @param t the tensor, allow multiple rows
 * @param k the returned neighbors
 * @return std::vector<std::vector<std::string>> the result object for each row of query
 */
  // virtual std::vector<std::vector<std::string>> searchStringObject(torch::Tensor &q, int64_t k);
};

/**
 * @ingroup  CANDY_lib_bottom
 * @typedef FlatSSDGPUIndexPtr
 * @brief The class to describe a shared pointer to @ref  FlatSSDGPUIndex

 */
typedef std::shared_ptr<class CANDY::FlatSSDGPUIndex> FlatSSDGPUIndexPtr;
/**
 * @ingroup  CANDY_lib_bottom
 * @def newFlatSSDGPUIndex
 * @brief (Macro) To creat a new @ref  FlatSSDGPUIndex shared pointer.
 */
#define newFlatSSDGPUIndex std::make_shared<CANDY::FlatSSDGPUIndex>
}
/**
 * @}
 */
#endif //INTELLISTREAM_INCLUDE_CPPALGOS_ABSTRACTCPPALGO_H_
