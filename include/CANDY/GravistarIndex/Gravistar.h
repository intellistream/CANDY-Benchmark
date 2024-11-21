/*! \file Gravistar.h*/
//
// Created by tony on 24-11-21.
//

#ifndef CANDYBENCH_INCLUDE_CANDY_GRAVISTARINDEX_GRAVISTAR_H_
#define CANDYBENCH_INCLUDE_CANDY_GRAVISTARINDEX_GRAVISTAR_H_
#include <CANDY/FlatGPUIndex/DiskMemBuffer.h>
namespace CANDY {
/**
 *  @defgroup  CANDY_lib_bottom_sub The support classes for index approaches
 * @{
 */
/**
 * @class Gravistar_DarkMatter CANDY/Gravistar/Gravistar.h
 * @brief The dark matter class, which is the basic, stackable element to hold raw data
 */
class Gravistar_DarkMatter{
 protected:
  /**
   * @brief this is where we hold data tensors
   */
  PlainMemBufferTU dmBuffer;
 public:
  Gravistar_DarkMatter() {}
  ~Gravistar_DarkMatter() {}
  /**
   * @brief init everything
   * @param vecDim The dimension of vectors
   * @param bufferSize the size for both tensor cache (in rows) and  U64 cache (in sizeof(uint64_t))
   * @param _tensorBegin the begin offset of tensor storage in disk
   * @param _u64Begin the begin offset of u64 storage in disk
   * @param _dmaSize the max size of dma buffer, I64, default 1024000
   */
  void init(int64_t vecDim,
            int64_t bufferSize,
            int64_t _tensorBegin,
            int64_t _u64Begin,
            int64_t _dmaSize = 1024000);
};


}
/**
 * @}
 */
#endif //CANDYBENCH_INCLUDE_CANDY_GRAVISTARINDEX_GRAVISTAR_H_
