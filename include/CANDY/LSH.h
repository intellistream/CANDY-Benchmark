#ifndef CANDY_INCLUDE_CANDY_LSH_H_
#define CANDY_INCLUDE_CANDY_LSH_H_

#include <CANDY/AbstractIndex.h>
#include <Utils/IntelliTensorOP.hpp>
#include <CANDY/LSH/VectorGenerator.h>
#include <vector>
#include <queue>
#include <iostream>
#include <map>
#include <torch/torch.h>
#include <assert.h>
#include <cmath>

namespace CANDY {

using HashValue = int64_t;
using VectorIndex = size_t;
using HashTable = std::multimap<HashValue, VectorIndex>;
using HashFunction = std::function<HashValue(const torch::Tensor& t)>;

class LSH : public AbstractIndex
{
private:
    // Number of hash functions
    int64_t hashFunctionNum;
    // random vector
    std::vector<torch::Tensor> randomVectors;
    // Hash functions
    std::vector<HashFunction> hashFunctions;
    // Hash tables
    std::vector<HashTable> hashTables;
    // Vector dimension
    int64_t vecDim;
    // Tensor Database
    std::vector<INTELLI::TensorPtr> tensorDatabase;
    // Vector Generator
    NormalizationVectorGenerator vectorGenerator;
    // deleteCache
    std::queue<VectorIndex> deleteCache;

    // search indexs of the k-NN of a query tensor
    std::vector<VectorIndex> searchIndex(const torch::Tensor &q, int64_t k);
public:
    LSH();
    ~LSH();
    /**
    * @brief set the index-specific config related to one index
    * @param cfg the config of this class
    * @return bool whether the configuration is successful
    */
    bool setConfig(INTELLI::ConfigMapPtr cfg);
    /**
     * @brief insert the tensors of a data base
     * @param t the data tensor
     * @return bool whether the inserting is successful
     */
    bool insertTensor(const torch::Tensor &t);
    /**
     * @brief search the k-NN of a query tensor
     * @param t the query tensor
     * @param k the returned neighbors
     * @return std::vector<INTELLI::TensorPtr> the result tensor for each row of query
     */
    std::vector<INTELLI::TensorPtr> searchTensor(const torch::Tensor &q, int64_t k);
    /**
     * @brief delete the tensors of a data base
     * @param t the data tensor
     * @return bool whether the deleting is successful
     */
    bool deleteTensor(const torch::Tensor &t);
};

}

#endif