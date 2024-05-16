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
#include <CANDY/CANDYObject.h>
#include <CANDY/FlatAMMIPObjIndex.h>
#include <CANDY/YinYangGraphIndex/hnswlib/hnswlib.h>
namespace CANDY {
/**
 * @ingroup  CANDY_lib_bottom The main body and interfaces of library function
 * @{
 */
class inlineYangIndex: public FlatAMMIPObjIndex {
 public:
  inlineYangIndex(){}
  ~inlineYangIndex() {}
  /**
 * @brief set the index-specific config related to one index
 * @param cfg the config of this class
 * @return bool whether the configuration is successful
 */
  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);
};
/**
* @class YinYangHNSW_YinVertex CANDY/YinYangHNSWIndex.h
* @brief The yin vertex in a yin yang HNSW, will be linked in an HNSW graph and further index yang vertex
 * @note the current version of yang vertex is just to use @ref FlatAMMIPObjIndex
*/

class YinYangHNSW_YinVertex {
 protected:

 public:

  YinYangHNSW_YinVertex() {}
  ~YinYangHNSW_YinVertex() {}
  torch::Tensor verTensor;
  float knowMaxDistance=0;
  inlineYangIndex yangIndex;
};

/**
 * @class YinYangHNSWIndex CANDY/YinYangHNSWIndex.h
 * @brief The class for yin yang HNSW index, top tier of ranging, and bottom tier of navigation
 * @todo implement L2 later
 * @note config parameters
 * - vecDim, the dimension of vectors, default 768, I64
 * - maxHNSWVolume, the max vertex amount in HNSW graph, default 1000000, I64
 * - maxConnection, number of maximum neighbor connection at each level, default 32, I64
 * - efConstruction,  Controls index search speed/build speed tradeoff, default 200, I64
 * - ammAlgo, the amm algorithm used for compute distance, default mm, String, can be the following
    * - mm the original torch::matmul
    * - crs column row sampling
    * - smp-pca the smp-pca algorithm
 */
class YinYangHNSWIndex: public AbstractIndex{
 protected:



  INTELLI::ConfigMapPtr inlineCfg = nullptr;
  hnswlib::HierarchicalNSW<float>* alg_hnsw;
  int64_t vecDim = 0,maxHNSWVolume=1000000,maxConnection=16,efConstruction=200;
  torch::Tensor searchRow(torch::Tensor &q, int64_t k);
 public:

  YinYangHNSWIndex() {

  }

  ~YinYangHNSWIndex() {

  }

  /**
    * @brief reset this index to inited status
    */
  //virtual void reset();

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
 // virtual bool deleteTensor(torch::Tensor &t, int64_t k = 1);

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
