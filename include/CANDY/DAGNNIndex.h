//
// Created by rubato on 12/6/24.
//

#ifndef DAGNNINDEX_H
#define DAGNNINDEX_H
#include <CANDY/AbstractIndex.h>
#include <CANDY/DAGNNIndex/DAGNN.h>
namespace CANDY {
    class DAGNNIndex : public CANDY::AbstractIndex {
    public:
        DynamicTuneHNSW* dagnn = nullptr;
        int64_t vecDim;



        DAGNNIndex()= default;
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
        virtual std::vector<faiss::idx_t> searchIndex(torch::Tensor q, int64_t k);
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
        virtual std::vector<torch::Tensor> getTensorByIndex(std::vector<faiss::idx_t> &idx, int64_t k);

        /**
        * @brief delete a tensor, also online function
        * @param t the tensor, some index needs to be single row
        * @param k the number of nearest neighbors
        * @return bool whether the deleting is successful
        */
        virtual bool deleteTensor(torch::Tensor &t, int64_t k = 1);

        bool deleteTensorByIndex(torch::Tensor &t);

        DynamicTuneHNSW::GraphStates getState(){
            return dagnn->graphStates;
        }

        bool performAction(const size_t action_num){
            return dagnn->performAction(action_num);
        }
    };
#define newDAGNNIndex std::make_shared<CANDY::DAGNNIndex>
}

#endif //DAGNNINDEX_H
