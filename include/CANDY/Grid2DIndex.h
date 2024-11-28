/*! \file Grid2DIndex.h*/
//
// Created by tony on 04/01/24.
//

#ifndef CANDY_INCLUDE_CANDY_Grid2DIndex_H_
#define CANDY_INCLUDE_CANDY_Grid2DIndex_H_
#include <Utils/ConfigMap.hpp>
#include <memory>
#include <vector>
#include <Utils/IntelliTensorOP.hpp>
#include <CANDY/AbstractIndex.h>
#include <CANDY/FlatGPUIndex/DiskMemBuffer.h>
#include <CANDY/Grid2DIndex/Grid2DOfTensor.h>
#include <CANDY/Grid2DIndex/MLPGravastarModel.h>
namespace CANDY {
class Grid2DIndex;
/**
 * @ingroup  CANDY_lib_bottom The main body and interfaces of library function
 * @{
 */
/**
 * @class Grid2DIndex CANDY/Grid2DIndex.h
 * @brief using 2D grid of gravastar inner surface for indexing, the fitting function is simple MLP
 * @note currently single thread
 * @note config parameters
 * - vecDim, the dimension of vectors, default 768, I64
 * - initialVolume, the initial volume of inline database tensor, default 1000, I64
 * - memBufferSize, the size of memory buffer, in rows of vectors, MUST larger than designed data size, default 1000, I64
 * - sketchSize, the sketch size of amm, default 10, I64
 * - DCOBatchSize, the batch size of internal distance comparison operation (DCO), default equal to memBufferSize, I64
 * - cudaDevice, the cuda device for DCO, default -1 (none), I64
 * - numberOfGrids, the number of grids used in each dimension, default 100, I64
 * - cudaDeviceInference the cuda device for inference, I64, default -1 (cpu only)
 * - learningRate the learning rate for training, Double, default 0.01
 * - hiddenLayerDim the dimension of hidden layer, I64, default the same as output layer
 * - MLTrainBatchSize the batch size of ML training, I64, default 64
 * - MLTrainMargin the margin value in regulating variance used in training, Double, default 0
 * - MLTrainEpochs the number of epochs in training, I64, default 10
 * @warning please run the benchmark/scripts/setupSPDK/drawTogether.py at generation path before using SSD
 */
class Grid2DIndex : public AbstractIndex {
 protected:
  INTELLI::ConfigMapPtr myCfg = nullptr;
  torch::Tensor dbTensor, objTensor;
  Grid2DOfTensor gridStore;
  MLPGravastarModel mlFitter;
  PlainMemBufferTU dmBuffer;
  int64_t ammType = 0;
  int64_t sketchSize = 10;
  int64_t DCOBatchSize = -1;
  int64_t memBufferSize = 1000;
  int64_t vecDim = 768;
  int64_t cudaDevice = -1;
  int64_t numberOfGrids = 100;

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
   * @param idx the pointer to index
   * @return The distance tensor, must sized [q*n] and remain in cpu
   */
  torch::Tensor (*distanceFunc)(torch::Tensor db, torch::Tensor query, int64_t cudaDev, Grid2DIndex *idx);
  /**
   * @brief the distance function of inner product
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @param idx the pointer to index
   * @return The distance tensor, must sized [q*n], will in GPU if cuda is valid
   */
  static torch::Tensor distanceIP(torch::Tensor db, torch::Tensor query, int64_t cudaDev, Grid2DIndex *idx);
  /**
   * @brief the distance function of L2
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @param idx the pointer to index
   * @return The distance tensor, must sized [q*n], will in GPU if cuda is valid
   */
  static torch::Tensor distanceL2(torch::Tensor db, torch::Tensor query, int64_t cudaDev, Grid2DIndex *idx);


  /**
   * @brief insert a tensor to grid
   * @param t the tensor, accept multiple rows
   * @return bool whether the insertion is successful
   */
  virtual bool insertTensor2Grid(torch::Tensor &t);
  // std::vector<faiss::idx_t> knnInline(torch::Tensor &query, int64_t k, int64_t distanceBatch = -1);
 public:
  Grid2DIndex() {

  }

  ~Grid2DIndex() {

  }
  int64_t gpuComputingUs = 0;
  int64_t gpuCommunicationUs = 0;
  /**
 * @brief load the initial tensors of a data base, use this BEFORE @ref insertTensor
 * @note This is majorly an offline function, and may be different from @ref insertTensor for some indexes
 * @param t the tensor, some index need to be single row
 * @return bool whether the loading is successful
 */
  virtual bool loadInitialTensor(torch::Tensor &t);
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

  /**
   * @brief to reset the internal statistics of this index
   * @return whether the reset is executed
   */
  virtual bool resetIndexStatistics(void);
  /**
   * @brief to get the internal statistics of this index
   * @return the statistics results in ConfigMapPtr
   */
  virtual INTELLI::ConfigMapPtr getIndexStatistics(void);
};

/**
 * @ingroup  CANDY_lib_bottom
 * @typedef Grid2DIndexPtr
 * @brief The class to describe a shared pointer to @ref  Grid2DIndex

 */
typedef std::shared_ptr<class CANDY::Grid2DIndex> Grid2DIndexPtr;
/**
 * @ingroup  CANDY_lib_bottom
 * @def newGrid2DIndex
 * @brief (Macro) To creat a new @ref  Grid2DIndex shared pointer.
 */
#define newGrid2DIndex std::make_shared<CANDY::Grid2DIndex>
}
/**
 * @}
 */
#endif //INTELLISTREAM_INCLUDE_CPPALGOS_ABSTRACTCPPALGO_H_
