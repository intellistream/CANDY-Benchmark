//
// Created by Isshin on 2024/5/31.
//

#ifndef CANDY_DEGINDEX_H
#define CANDY_DEGINDEX_H
#include <CANDY/DEGIndex/deglib.h>
#include <CANDY/AbstractIndex.h>

#include "AbstractIndex.h"

namespace CANDY{
class DEGIndex : public AbstractIndex{
public:

    int64_t vecDim=100;
    unsigned k_ext = 60;
    unsigned d = 30; //edges per vertex
    unsigned eps_ext = 0.1 ;
    unsigned k_opt = 30;
    unsigned eps_opt = 0.001f ;
    unsigned i_opt = 5;
    bool use_schemeC_ext = false; //true for SIFT ,false for GLOVE
    std::string datasetName;
    int64_t expandStep = 100;
    //float rnd = std::mt19937(7);
    std::mt19937 rnd;// default 7
    const deglib::Metric metric = deglib::Metric::L2;   // defaul metric
    const uint32_t swap_tries = 0;                      // additional swap tries between the next graph extension
    const uint32_t additional_swap_tries = 0;           // increse swap try count for each successful swap
    float dims = vecDim;
    uint32_t max_vertex_count=1000000;
    uint32_t label;
    //std::unordered_map<uint32_t, const float*> feature_map;
  //   deglib::FloatSpace feature_space = deglib::FloatSpace(dims, metric);
  // deglib::graph::SizeBoundedGraph index_ = deglib::graph::SizeBoundedGraph (max_vertex_count, vecDim, feature_space);
  //   deglib::builder::EvenRegularGraphBuilder builder = deglib::builder::EvenRegularGraphBuilder(index_, rnd, k_ext, eps_ext, use_schemeC_ext, k_opt, eps_opt, i_opt, swap_tries, additional_swap_tries);

    deglib::FloatSpace feature_space;
    deglib::graph::SizeBoundedGraph graphIndex;
    deglib::builder::EvenRegularGraphBuilder builder;

    DEGIndex() :
        feature_space(vecDim, metric),
        graphIndex(max_vertex_count, d, feature_space),
        //builder(graphIndex, rnd, k_ext, eps_ext, use_schemeC_ext, k_opt, eps_opt, i_opt, swap_tries, additional_swap_tries) {}
    builder(graphIndex, rnd, k_ext, eps_ext, use_schemeC_ext,  0, 0, 0, 0, 0) {}


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
    //static float test_approx_anns(const deglib::search::SearchGraph& graph, const std::vector<uint32_t>& entry_vertex_indices,
                           //  const deglib::FeatureRepository& query_repository, const std::vector<std::unordered_set<uint32_t>>& ground_truth,
                            // const float eps, const uint32_t k, const uint32_t test_size);


};
#define newDEGIndex std::make_shared<CANDY::DEGIndex>

}
#endif //CANDY_DEGINDEX_H

