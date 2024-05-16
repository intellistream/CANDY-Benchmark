/*! \file YinYangHNSWIndex.h*/
//
// Created by tony on 04/01/24.
//

#ifndef CANDY_INCLUDE_CANDY_YINYANGHNSWINDEX_H_
#define CANDY_INCLUDE_CANDY_YINYANGHNSWINDEX_H_

#include <Utils/AbstractC20Thread.hpp>
#include <Utils/ConfigMap.hpp>
#include <memory>
#include <vector>
#include <Utils/IntelliTensorOP.hpp>
#include <faiss/IndexFlat.h>
#include <tuple>
#include <CANDY/AbstractIndex.h>
namespace CANDY {

/**
 * @ingroup  CANDY_lib_bottom The main body and interfaces of library function
 * @{
 */
/**
 * @class YinYangHNSWIndex CANDY/YinYangHNSWIndex.h
 * @brief The class for yin yang HNSW index, top tier of ranging, and bottom tier of navigation
 */
class YinYangHNSWIndex: public AbstractIndex{
 protected:

 public:

  YinYangHNSWIndex() {

  }

  ~YinYangHNSWIndex() {

  }

  /**
    * @brief reset this index to inited status
    */
  virtual void reset();

  /**
   * @brief set the index-specfic config related to one index
   * @param cfg the config of this class
   * @note If there is any pre-built data structures, please load it in implementing this
   * @note If there is any initial tensors to be stored, please load it after this by @ref loadInitialTensor
   * @return bool whether the configuration is successful
   */
  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  /**
   * @brief insert a tensor
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @return bool whether the insertion is successful
   */
  virtual bool insertTensor(torch::Tensor &t);

  /**
  * @brief load the initial tensors of a data base, use this BEFORE @ref insertTensor
  * @note This is majorly an offline function, and may be different from @ref insertTensor for some indexes
  * @param t the tensor, some index need to be single row
  * @return bool whether the loading is successful
  */
  virtual bool loadInitialTensor(torch::Tensor &t);
  /**
   * @brief delete a tensor, also online function
   * @param t the tensor, some index needs to be single row
   * @param k the number of nearest neighbors
   * @return bool whether the deleting is successful
   */
  virtual bool deleteTensor(torch::Tensor &t, int64_t k = 1);

  /**
   * @brief revise a tensor
   * @param t the tensor to be revised
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
  virtual std::vector<faiss::idx_t> searchIndex(torch::Tensor q, int64_t k);

  /**
   * @brief return a vector of tensors according to some index
   * @param idx the index, follow faiss's style, allow the KNN index of multiple queries
   * @param k the returned neighbors, i.e., will be the number of rows of each returned tensor
   * @return a vector of tensors, each tensor represent KNN results of one query in idx
   */
  virtual std::vector<torch::Tensor> getTensorByIndex(std::vector<faiss::idx_t> &idx, int64_t k);
  /**
    * @brief return the rawData of tensor
    * @return The raw data stored in tensor
    */
  virtual torch::Tensor rawData();
  /**
  * @brief search the k-NN of a query tensor, return the result tensors
  * @param t the tensor, allow multiple rows
  * @param k the returned neighbors
  * @return std::vector<torch::Tensor> the result tensor for each row of query
  */
  virtual std::vector<torch::Tensor> searchTensor(torch::Tensor &q, int64_t k);

  /**
  * @brief load the initial tensors of a data base along with its string objects, use this BEFORE @ref insertTensor
  * @note This is majorly an offline function, and may be different from @ref insertTensor for some indexes
  * @param t the tensor, some index need to be single row
   *  * @param strs the corresponding list of strings
  * @return bool whether the loading is successful
  */
  //virtual bool loadInitialStringObject(torch::Tensor &t, std::vector<std::string> &strs);
  /**
   * @brief insert a string object
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @param strs the corresponding list of strings
   * @return bool whether the insertion is successful
   */
  //virtual bool insertStringObject(torch::Tensor &t, std::vector<std::string> &strs);

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
 * @brief search the k-NN of a query tensor, return the linked string objects and original tensors
 * @param t the tensor, allow multiple rows
 * @param k the returned neighbors
 * @return std::tuple<std::vector<torch::Tensor>,std::vector<std::vector<std::string>>>
 */
  /*virtual std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string>>> searchTensorAndStringObject(
      torch::Tensor &q,
      int64_t k);*/
};

/**
 * @ingroup  CANDY_lib_bottom
 * @typedef YinYangHNSWIndexPtr
 * @brief The class to describe a shared pointer to @ref  YinYangHNSWIndex

 */
typedef std::shared_ptr<class CANDY::YinYangHNSWIndex> YinYangHNSWIndexPtr;
/**
 * @ingroup  CANDY_lib_bottom
 * @def newYinYangHNSWIndex
 * @brief (Macro) To creat a new @ref  YinYangHNSWIndex shared pointer.
 */
#define newYinYangHNSWIndex std::make_shared<CANDY::YinYangHNSWIndex>
}
/**
 * @}
 */

#endif //INTELLISTREAM_INCLUDE_CPPALGOS_ABSTRACTCPPALGO_H_
