/*! \file DiskMemBuffer.h*/
//
// Created by tony on 24/07/24.
//

#ifndef CANDY_INCLUDE_CANDY_FLATSSDGPUINDEX_DISKMEMBUFFER_H_
#define CANDY_INCLUDE_CANDY_FLATSSDGPUINDEX_DISKMEMBUFFER_H_
#include <stdint.h>
#include <memory>
#include <torch/torch.h>
#include <CANDY/FlatSSDGPUIndex/SPDKSSD.h>
#include <vector>
#include <atomic>
namespace CANDY{
/**
 *  @defgroup  CANDY_lib_bottom_sub The support classes for index approaches
 * @{
 */
/**
 * @class DiskHeader CANDY/FlatSSDGPUIndex/DiskMemBuffer.h
 * @brief The class to store necessary information on disk, typically at first sector
 */
class DiskHeader {
 public:
  uint64_t version = 0;
  uint64_t vecDim = 0;
  uint64_t vecCnt = 0;
  uint64_t u64Cnt = 0;
  uint64_t aknnType = 0;
  DiskHeader() {}
  ~DiskHeader() {}
};
/**
 * @class TensorCacheLine CANDY/FlatSSDGPUIndex/DiskMemBuffer.h
 * @brief The virtual cache line to buffer data, storage of tensor
 */
class TensorVCacheLine {
 public:
  int64_t beginPos = 0;
  int64_t endPos = 0;
  int64_t temperature = 0;
  torch::Tensor buffer;
  TensorVCacheLine () {}
  ~TensorVCacheLine () {}
};
/**
 * @class U64VCacheLine CANDY/FlatSSDGPUIndex/DiskMemBuffer.h
 * @brief The virtual cache line to buffer data, storage of uint64_t
 */
class U64VCacheLine {
 public:
  int64_t beginPos = 0;
  int64_t endPos = 0;
  int64_t temperature = 0;
  std::vector<uint64_t>buffer;
  U64VCacheLine () {}
  ~U64VCacheLine () {}
};
/**
 * @class PlainDiskMemBufferOfTensor CANDY/FlatSSDGPUIndex/DiskMemBuffer.h
 * @brief a straight forward plain storage of tensor and u64, will firstly use in-memory data, and switch into disk, full flush between memory and disk
 * @note will use half of namespace for tensor, another for U64
 */
class PlainDiskMemBufferTU {
 protected:
  DiskHeader diskInfo;
  TensorVCacheLine cacheT;
  U64VCacheLine cacheU;
  int64_t tensorBegin=0,u64Begin=0;
  int64_t bsize=0;
  int64_t dmaSize = 1024000;
  struct spdk_nvme_qpair* ssdQpair = NULL;
  std::atomic_bool isDirtyT = false;
  std::atomic_bool isDirtyU = false;
 public:
  struct spdk_nvme_qpair* diskQpair;
  PlainDiskMemBufferTU() {}
  ~PlainDiskMemBufferTU() {}
  SPDKSSD *ssdPtr = nullptr;
  /**
   * @brief init everything
   * @param vecDim The dimension of vectors
   * @param bufferSize the size for both tensor cache (in rows) and  U64 cache (in sizeof(uint64_t))
   * @param _tensorBegin the begin offset of tensor storage in disk
   * @param _u64Begin the begin offset of u64 storage in disk
   * @param qpair the disk io pair
   * @param _ssdPtr the pointer of linked ssd
   * @param _dmaSize the max size of dma buffer, I64, default 1024000
   */
  void init(int64_t vecDim,int64_t bufferSize,int64_t _tensorBegin,int64_t _u64Begin,SPDKSSD *_ssdPtr,int64_t _dmaSize=1024000);
  /**
   * @brief to return the size of ingested vectors
   * @return the number of rows.
   */
  int64_t size();
  /**
  * @brief clear the occupied resource
  */
  void clear();
  /**
   * @brief to get the tensor at specified position
   * @param startPos the start position
   * @param endPos the end position
   * @return the tensor, [n*vecDim]
   */
  torch::Tensor getTensor(int64_t startPos,int64_t endPos);
  /**
 * @brief to get the tensor at specified position
 * @param startPos the start position
 * @param endPos the end position
 * @return the tensor, [n*vecDim]
 */
  std::vector<uint64_t> getU64(int64_t startPos,int64_t endPos);
  /**
   * @brief to revise the tensor at specified position
   * @param startPos the start position
   * @param t the tensor, [n*vecDim]
   * @return whether it is successful
   */
  bool reviseTensor(int64_t startPos,torch::Tensor &t);
  /**
   * @brief to revise the tensor at specified position
   * @param startPos the start position
   * @param u the u64 vector, [n]
   * @return whether it is successful
   */
  bool reviseU64(int64_t startPos,std::vector<uint64_t> &u);
  /**
 * @brief to append the tensor to the end of storage region
 * @param t the tensor, [n*vecDim]
 * @return whether it is successful
 */
  bool appendTensor(torch::Tensor &t);
  /**
 * @brief to append the tensor to the end of storage region
 * @param u the u64 vector, [n]
 * @return whether it is successful
 */
  bool appendU64(std::vector<uint64_t> &u);
  /**
   * @brief to delete the tensor at specified position
   * @param startPos the start position
   * @param endPos the end position
   * @return whether it is successful
   */
  bool deleteTensor(int64_t startPos,int64_t endPos);
  /**
   * @brief to delete a U64 at specified position
   * @param startPos the start position
   * @param endPos the end position
   * @return whether it is successful
   */
  bool deleteU64(int64_t startPos,int64_t endPos);
};
}
/**
 * @}
 */
#endif //CANDY_INCLUDE_CANDY_FLATSSDGPUINDEX_DISKMEMBUFFER_H_
