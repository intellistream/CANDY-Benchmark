/*! \file Gravistar.h*/
//
// Created by tony on 24-11-21.
//

#ifndef CANDYBENCH_INCLUDE_CANDY_GRAVISTARINDEX_GRAVISTAR_H_
#define CANDYBENCH_INCLUDE_CANDY_GRAVISTARINDEX_GRAVISTAR_H_
#include <CANDY/FlatGPUIndex/DiskMemBuffer.h>
#include <memory>
#include <vector>
namespace CANDY {
class Gravistar;
class GravistarStatitics;
class GravistarStatitics{
 public:
  GravistarStatitics(){}
  ~GravistarStatitics(){}
  int64_t gpuComputingUs = 0, cpuComputingUs = 0, gpuCommunicationUs = 0;
};
/**
 *  @defgroup  CANDY_lib_bottom_sub The support classes for index approaches
 * @{
 */


typedef std::shared_ptr<class CANDY::Gravistar> GravistarPtr;

#define  newGravistar std::make_shared<CANDY::Gravistar>
/**
 * @class Gravistar CANDY/Gravistar/Gravistar.h
 * @brief The dark matter class, which is the basic, stackable element to hold raw data
 */
class Gravistar: public std::enable_shared_from_this<Gravistar>{
 protected:
  /**
   * @brief this is where we hold data tensors
   */
  bool lastTier = true;
  PlainMemBufferTU dmBuffer;
  int64_t bufferSize = -1;
  torch::Tensor gravityCenter;
  int64_t vecDim = -1;
  int64_t cudaDevice = -1;
  int64_t distanceMode = 0; //IP
  int64_t batchSize = 0;
  /**
   * @brief to include all probe tensors and return, namely the gravity field
   * @return the tensor
   */
  torch::Tensor constructGravityField(GravistarPtr root);
 public:
  Gravistar() {}
  ~Gravistar() {}
  GravistarStatitics * statisticsInfo = nullptr;
  bool hasDownTier = false;
  /**
   * @brief pointer to the upper tier dark matter
   */
  GravistarPtr upperTier;
  std::vector<GravistarPtr>downTiers;
  /**
   * @brief init everything
   * @param _vecDim The dimension of vectors
   * @param _bufferSize the size for both tensor cache (in rows) and  U64 cache (in sizeof(uint64_t))
   * @param _tensorBegin the begin offset of tensor storage in disk
   * @param _u64Begin the begin offset of u64 storage in disk
   * @param _dmaSize the max size of dma buffer, I64, default 1024000
   */
  void init(int64_t _vecDim,
            int64_t _bufferSize,
            int64_t _tensorBegin,
            int64_t _u64Begin,
            int64_t _dmaSize = 1024000);
  void setConstraints(int64_t cudaId,int64_t _distanceMode, int64_t _batchSize, GravistarStatitics * sta);
  /**
   * @brief insert a tensor
   * @param t the tensor,1xD
   * @param root the root gravistar
   * @return the updated root
   */
  GravistarPtr insertTensor(torch::Tensor &t,GravistarPtr root);
  /**
    * @brief to get the tensor at specified position
    * @param startPos the start position
    * @param endPos the end position
    * @return the tensor, [n*vecDim]
    */
  torch::Tensor getTensor(int64_t startPos, int64_t endPos);
  /**
   * @brief set this dark matter as last tier entity
   * @param val the value
   * @return
   */
   void setToLastTier(bool val);
   /**
    * @brief to determine is this one the last tier
    * @return
    */
   bool isLastTier(void);

  // Main function to process batches and find top_k closest vectors
  std::vector<int64_t> findTopKClosest(const torch::Tensor &query, int64_t top_k, int64_t batch_size,GravistarPtr root);

  // torch::Tensor myMMInline(torch::Tensor &a, torch::Tensor &b, int64_t ss = 10);
  /**
  * @brief return a vector of tensors according to some index
  * @param idx the index, follow faiss's style, allow the KNN index of multiple queries
  * @param k the returned neighbors, i.e., will be the number of rows of each returned tensor
  * @return a vector of tensors, each tensor represent KNN results of one query in idx
  */
 // virtual std::vector<torch::Tensor> getTensorByStdIdx(std::vector<int64_t> &idx, int64_t k);
  /**
   * @brief the distance function pointer member
   * @note will select largest distance during the following sorting, please convert if your distance is 'minimal'
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @param idx the pointer to gravistar
   * @return The distance tensor, must sized [q*n] and remain in cpu
   */
  torch::Tensor (*distanceFunc)(torch::Tensor db, torch::Tensor query, int64_t cudaDev, GravistarStatitics *idx);
  /**
   * @brief the distance function of inner product
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @param idx the pointer to index
   * @return The distance tensor, must sized [q*n], will in GPU if cuda is valid
   */
  static torch::Tensor distanceIP(torch::Tensor db, torch::Tensor query, int64_t cudaDev, GravistarStatitics *idx);
  /**
   * @brief the distance function of L2
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @param idx the pointer to index
   * @return The distance tensor, must sized [q*n], will in GPU if cuda is valid
   */
  static torch::Tensor distanceL2(torch::Tensor db, torch::Tensor query, int64_t cudaDev, GravistarStatitics *idx);
  /**
   * @brief find the the Gravistar which contains t
   * @param t the tensor, must be 1xD
   * @param batchSize, the batch size
   * @param root the root gravistar
   * @return the shared pointer to specified gravistar
   */
  GravistarPtr findGravistar(torch::Tensor &t,int64_t batchSize,GravistarPtr root);
  int64_t size(void){
    return dmBuffer.size();
  }
};

}
/**
 * @}
 */
#endif //CANDYBENCH_INCLUDE_CANDY_GRAVISTARINDEX_GRAVISTAR_H_
