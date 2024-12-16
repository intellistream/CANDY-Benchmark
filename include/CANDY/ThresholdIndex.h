/*! \file ThresholdIndex.h*/
//
// Created by tony on 04/01/24.
//

#ifndef CANDY_INCLUDE_CANDY_THRESHOLDINDEX_H_
#define CANDY_INCLUDE_CANDY_THRESHOLDINDEX_H_

#include <Utils/AbstractC20Thread.hpp>
#include <Utils/ConfigMap.hpp>
#include <memory>
#include <vector>
#include <Utils/IntelliTensorOP.hpp>
#include <faiss/IndexFlat.h>
#include <CANDY/AbstractIndex.h>
#include <tuple>
namespace CANDY {

/**
 * @ingroup  CANDY_lib_bottom The main body and interfaces of library function
 * @{
 */
/**
 * @class ThresholdIndex CANDY/ThresholdIndex.h
 * @brief The Threshold class of an index approach
 */
class ThresholdIndex: public AbstractIndex{
 protected:
  faiss::MetricType faissMetric = faiss::METRIC_L2;
  int64_t containerTier = 0;
  int64_t dataThreshold;
  int64_t dataVolume;
  std::string indexAlgorithm;
  //std::vector<faiss::Index*> indices;
  std::vector<CANDY::AbstractIndexPtr> indices;


 public:
  bool isHPCStarted = false;
  ThresholdIndex() {

  }

  ~ThresholdIndex() {

  }
  /**
  * @brief set the tier of this indexing, 0 refers the entry indexing
  * @param tie the setting of tier number
  * @note The parameter of tier idx affects nothing now, but will do something later
  */
  /*virtual void setTier(int64_t tie) {
    containerTier = tie;
  }*/
  /**
    * @brief reset this index to inited status
    */
  virtual void reset();
  /**
  * @brief set the index-specific config related to one index
  * @param cfg the config of this class, using raw class
  * @note If there is any pre-built data structures, please load it in implementing this
  * @note If there is any initial tensors to be stored, please load it after this by @ref loadInitialTensor
  * @return bool whether the configuration is successful
  */
  virtual bool setConfigClass(INTELLI::ConfigMap cfg);
  /**
   * @brief set the index-specfic config related to one index
   * @param cfg the config of this class
   * @note If there is any pre-built data structures, please load it in implementing this
   * @note If there is any initial tensors to be stored, please load it after this by @ref loadInitialTensor
   * @return bool whether the configuration is successful
   */
  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);
  /**
   * @brief some extra set-ups if the index has HPC fetures
   * @return bool whether the HPC set-up is successful
   */
  //virtual bool startHPC();
  /**
   * @brief insert a tensor
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @return bool whether the insertion is successful
   */
  virtual bool insertTensor_th(torch::Tensor &t, std::string nameTag);
  

  virtual void createThresholdIndex(int64_t dimension, std::string nameTag);
  //CANDY::AbstractIndexPtr createIndex(const std::string& nameTag);

  //CANDY::AbstractIndexPtr createIndex(std::string nameTag);
  
  /**
  * @brief load the initial tensors of a data base, use this BEFORE @ref insertTensor
  * @note This is majorly an offline function, and may be different from @ref insertTensor for some indexes
  * @param t the tensor, some index need to be single row
  * @return bool whether the loading is successful
  */
 // virtual bool loadInitialTensor(torch::Tensor &t);
  /**
   * @brief delete a tensor, also online function
   * @param t the tensor, some index needs to be single row
   * @param k the number of nearest neighbors
   * @return bool whether the deleting is successful
   */
  //virtual bool deleteTensor(torch::Tensor &t, int64_t k = 1);

  /**
   * @brief revise a tensor
   * @param t the tensor to be revised
   * @param w the revised value
   * @return bool whether the revising is successful
   */
  //virtual bool reviseTensor(torch::Tensor &t, torch::Tensor &w);
  /**
   * @brief search the k-NN of a query tensor, return their index
   * @param t the tensor, allow multiple rows
   * @param k the returned neighbors
   * @return std::vector<faiss::idx_t> the index, follow faiss's order
   */
  virtual std::vector<faiss::idx_t> searchIndex(torch::Tensor q, int64_t k);

/**
  * @brief search the k-NN of a query tensor, return the result tensors
  * @param t the tensor, allow multiple rows
  * @param k the returned neighbors
  * @return std::vector<torch::Tensor> the result tensor for each row of query
  */
  virtual std::vector<torch::Tensor> searchTensor_th(torch::Tensor &q, int64_t k);
  /**
    * @brief some extra termination if the index has HPC features
    * @return bool whether the HPC termination is successful
    */
 // virtual bool endHPC();
  /**
   * @brief set the frozen level of online updating internal state
   * @param frozenLv the level of frozen, 0 means freeze any online update in internal state
   * @return whether the setting is successful
   */
 // virtual bool setFrozenLevel(int64_t frozenLv);
  /**
  * @brief offline build phase
  * @param t the tensor for offline build
  * @note This is to generate some offline data structures, NOT load offline tensors
  * @note Please use @ref loadInitialTensor for loading initial tensors
  * @return whether the building is successful
  */
  virtual bool offlineBuild(torch::Tensor &t, std::string nameTag);
  /**
   * @brief a busy waiting for all pending operations to be done
   * @return bool, whether the waiting is actually done;
   */
  //virtual bool waitPendingOperations();

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
 // virtual bool insertStringObject(torch::Tensor &t, std::vector<std::string> &strs);

  /**
   * @brief  delete tensor along with its corresponding string object
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @param k the number of nearest neighbors
   * @return bool whether the delet is successful
   */
 // virtual bool deleteStringObject(torch::Tensor &t, int64_t k = 1);

  /**
 * @brief search the k-NN of a query tensor, return the linked string objects
 * @param t the tensor, allow multiple rows
 * @param k the returned neighbors
 * @return std::vector<std::vector<std::string>> the result object for each row of query
 */
  //virtual std::vector<std::vector<std::string>> searchStringObject(torch::Tensor &q, int64_t k);



  };

/**
 * @ingroup  CANDY_lib_bottom
 * @typedef ThresholdIndexPtr
 * @brief The class to describe a shared pointer to @ref  ThresholdIndex

 */
typedef std::shared_ptr<class CANDY::ThresholdIndex> ThresholdIndexPtr;
/**
 * @ingroup  CANDY_lib_bottom
 * @def newThresholdIndex
 * @brief (Macro) To creat a new @ref  ThresholdIndex shared pointer.
 */
#define newThresholdIndex std::make_shared<CANDY::ThresholdIndex>
}
/**
 * @}
 */

#endif //INTELLISTREAM_INCLUDE_CPPALGOS_ThresholdCPPALGO_H_
