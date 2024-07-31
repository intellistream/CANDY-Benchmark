/*! \file RBTDataLoader.h*/
//
// Created by tony on 10/05/23.
//

#ifndef CANDY_INCLUDE_DATALOADER_RBTDataLoader_H_
#define CANDY_INCLUDE_DATALOADER_RBTDataLoader_H_

#include <Utils/ConfigMap.hpp>
#include <Utils/IntelliTensorOP.hpp>
#include <assert.h>
//#include <torch/torch.h>
#include <memory>
#include <DataLoader/AbstractDataLoader.h>
namespace CANDY {
/**
 * @ingroup CANDY_DataLOADER
 * @{
 */
/**
 * @ingroup CANDY_DataLOADER_RBT The dataloader OF raw binary tensor (RBT)
 * @{
 */
/**
 * @class RBTDataLoader DataLoader/RBTDataLoader.h
 * @brief The class of RBT data loader,
 * @ingroup CANDY_DataLOADER
 * @note:
 * - Must have a global config by @ref setConfig
 * - This one support out-of-memory large data, but not work well with onlineInsert benchmark, please use onlineCUD instead
 * @note  Default behavior
* - create
* - call @ref setConfig, this function will also generate the tensor A and B correspondingly
* - call @ref getDataAt to get the raw data
* - call  @ref getQueryAt to get the query
* @note parameters of config
* - vecDim, the dimension of vectors, default 768, I64
* - vecVolume, the volume of vectors, default 1000, I64
* - driftPosition, the position of starting some 'concept drift', default 0 (no drift), I64
 * - driftOffset, the offset value of concept drift, default 0.5, Double
 * - queryNoiseFraction, the fraction of noise in query, default 0, allow 0~1, Double
* - querySize, the size of query, default 10, I64
* - seed, the random seed, default 7758258, I64
*  @note: default name tags
 * "rbt": @ref RBTDataLoader
 */
class RBTDataLoader : public AbstractDataLoader {
 protected:
  torch::Tensor A, B;
  int64_t vecDim, vecVolume, querySize, seed;
  double driftOffset, queryNoiseFraction;

 public:
  RBTDataLoader() = default;

  ~RBTDataLoader() = default;

  /**
     * @brief Set the GLOBAL config map related to this loader
     * @param cfg The config map
      * @return bool whether the config is successfully set
      * @note
     */
  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  /**
   * @brief get the data tensor at specific offset
   * @note implement and use this when the whole data tensor does not fit into main memory
   * @return the generated data tensor
   */
  virtual torch::Tensor getDataAt(int64_t startPos, int64_t endPos);

  /**
   * @brief get the data tensor
   * @return the generated data tensor
   */
  virtual torch::Tensor getData();

  /**
   * @brief get the data tensor at specific offset
   * @note implement and use this when the whole data tensor does not fit into main memory
   * @return the generated data tensor
   */
  virtual torch::Tensor getDataAt(int64_t startPos, int64_t endPos);
  /**
  * @brief get the query tensor
  * @return the generated query tensor
  */
  virtual torch::Tensor getQuery();

  /**
   * @brief get the data tensor at specific offset
   * @note implement and use this when the whole data tensor does not fit into main memory
   * @return the generated data tensor
   */
  virtual torch::Tensor getQueryAt(int64_t startPos, int64_t endPos);
  /**
   * @brief get the dimension of data
   * @return the dimension
   */
  virtual int64_t getDimension();
  /**
   * @brief get the number of rows of data
   * @return the rows
   */
  virtual int64_t size();
  /**
   * @brief
   * @param fname
   * @param t
   */
  static void createRBT(std::string fname,torch::Tensor &t);
  static void appendTensorToRBT(std::string fname,torch::Tensor &t);
};

/**
 * @ingroup CANDY_MatrixLOADER_Random
 * @typedef RBTDataLoaderPtr
 * @brief The class to describe a shared pointer to @ref RBTDataLoader

 */
typedef std::shared_ptr<class CANDY::RBTDataLoader> RBTDataLoaderPtr;
/**
 * @ingroup CANDY_MatrixLOADER_Random
 * @def newRBTDataLoader
 * @brief (Macro) To creat a new @ref RBTDataLoader under shared pointer.
 */
#define newRBTDataLoader std::make_shared<CANDY::RBTDataLoader>
/**
 * @}
 */
/**
 * @}
 */
} // CANDY

#endif //CANDY_INCLUDE_MATRIXLOADER_RBTDataLoader_H_
