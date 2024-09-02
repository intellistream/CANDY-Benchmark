//
// Created by shivangi on 24/8/24.
//

#ifndef BKTINDEX_H
#define BKTINDEX_H
#include <CANDY/SPFresh/BKT/Index.h>
#include <CANDY/AbstractIndex.h>
//#include "CANDY/SPFresh/Test.h"
#include "CANDY/SPFresh/Helper/SimpleIniReader.h"
#include "CANDY/SPFresh/VectorIndex.h"
#include "CANDY/SPFresh/Common/CommonUtils.h"

#include <unordered_set>
#include <chrono>
#include "AbstractIndex.h"

namespace CANDY{
class BKTIndex : public AbstractIndex{
public:

    //SPTAG::SizeType n; //size
    SPTAG::SizeType q = 3;
    SPTAG::DimensionType m; //dimension
    int k;
    SPTAG::IndexAlgoType algo=SPTAG::IndexAlgoType::BKT;
    std::string distCalcMethod = "L2";
    std::shared_ptr<SPTAG::VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<float>());

    std::vector<char> meta;
    std::vector<std::uint64_t> metaoffset;


    /**
    * @brief set the index-specific config related to one index
    * @param cfg the config of this class
    * @return bool whether the configuration is successful
    */
    virtual bool setConfig(INTELLI::ConfigMapPtr cfg);
    /**
    * @brief load the initial tensors of a data base, use this BEFORE @ref insertTensor
    * @note This is majorly an offline function, and may be different from @ref insertTensor for some indexes
    * @param t the tensor, some index need to be single row
    * @return bool whether the loading is successful
    */
    virtual bool loadInitialTensor(torch::Tensor &t);
    /**
    * @brief insert a tensor
    * @param t the tensor, accept multiple rows
    * @return bool whether the insertion is successful
    */
    virtual bool insertTensor(torch::Tensor &t);
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
     * @brief return a vector of tensors according to some index
     * @param idx the index, follow faiss's style, allow the KNN index of multiple queries
     * @param k the returned neighbors, i.e., will be the number of rows of each returned tensor
     * @return a vector of tensors, each tensor represent KNN results of one query in idx
     */
    //virtual std::vector<torch::Tensor> getTensorByIndex(std::vector<faiss::idx_t> &idx, int64_t k);
private:
    //static float test_approx_anns(const BKTlib::search::SearchGraph& graph, const std::vector<uint32_t>& entry_vertex_indices,
                           //  const BKTlib::FeatureRepository& query_repository, const std::vector<std::unordered_set<uint32_t>>& ground_truth,
                            // const float eps, const uint32_t k, const uint32_t test_size);


};
#define newBKTIndex std::make_shared<CANDY::BKTIndex>

}
#endif //BKTINDEX_H
