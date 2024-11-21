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
class Gravistar_DarkMatter;

/**
 *  @defgroup  CANDY_lib_bottom_sub The support classes for index approaches
 * @{
 */


typedef std::shared_ptr<class CANDY::Gravistar_DarkMatter> Gravistar_DarkMatterPtr;

#define  newGravistar_DarkMatter <std::make_shared<CANDY::Gravistar_DarkMatter>
/**
 * @class Gravistar_DarkMatter CANDY/Gravistar/Gravistar.h
 * @brief The dark matter class, which is the basic, stackable element to hold raw data
 */
class Gravistar_DarkMatter{
 protected:
  /**
   * @brief this is where we hold data tensors
   */
  bool lastTier = true;
  PlainMemBufferTU dmBuffer;
  int64_t bufferSize = -1;
  torch::Tensor gravityCenter;
  int64_t vecDim = -1;
 public:
  Gravistar_DarkMatter() {}
  ~Gravistar_DarkMatter() {}
  /**
   * @brief pointer to the upper tier dark matter
   */
  Gravistar_DarkMatterPtr upperTier;
  std::vector<Gravistar_DarkMatterPtr>downTiers;
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
  /**
   * @brief insert a tensor
   * @param t the tensor, accept multiple rows
   * @return bool whether the insertion is successful, will be false if run out of buffer
   */
  virtual bool insertTensor(torch::Tensor &t);
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

};
class Gravistar{
 public:
  Gravistar(){}
  ~Gravistar(){}
};

}
/**
 * @}
 */
#endif //CANDYBENCH_INCLUDE_CANDY_GRAVISTARINDEX_GRAVISTAR_H_
