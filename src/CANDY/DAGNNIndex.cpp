//
// Created by rubato on 12/6/24.
//
#include <CANDY/DAGNNIndex.h>

bool CANDY::DAGNNIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
    AbstractIndex::setConfig(cfg);
    vecDim = cfg->tryI64("vecDim", 768, true);
    std::string metricType = cfg->tryString("metricType", "IP", true);
    int metric = 0;
    if(metricType=="IP") {
        metric = DAGNN_METRIC_IP;
    } else {
        metric = DAGNN_METRIC_L2;
    }
    auto M = cfg->tryI64("maxConnection", 32, true);
    DynamicTuneHNSW::DynamicTuneParams dp;
    dagnn = new CANDY::DynamicTuneHNSW(M, vecDim, metric, dp);
    return true;
}

bool CANDY::DAGNNIndex::loadInitialTensor(torch::Tensor &t) {
    auto data_size = t.size(0);
    float *new_data = t.contiguous().data_ptr<float>();

    dagnn->add(data_size, new_data);
    return true;
}

bool CANDY::DAGNNIndex::insertTensor(torch::Tensor &t) {
    auto data_size = t.size(0);
    float *new_data = t.contiguous().data_ptr<float>();

    dagnn->add(data_size, new_data);
    return true;
}

std::vector<faiss::idx_t> CANDY::DAGNNIndex::searchIndex(torch::Tensor q, int64_t k) {
    auto queryData = q.contiguous().data_ptr<float>();
    auto querySize = q.size(0);

    std::vector<faiss::idx_t> ru(k*querySize);
    std::vector<float> distance(k*querySize);
    DAGNN::DistanceQueryer disq(vecDim);

    for(int64_t i=0; i<querySize; i++) {
        disq.set_query(queryData+i*vecDim);
        DAGNN::VisitedTable vt(dagnn->storage->ntotal);
        dagnn->search(disq, k, ru.data()+i*k, distance.data()+i*k, vt);

    }
    // for(int64_t i=0; i<querySize; i++) {
    //     printf("result for %ldth query\n", i);
    //     for(int64_t j=0; j<k; j++) {
    //         printf("%ld %ld: %f\n", i*k+j,ru[i*k+j], distance[i*k+j]);
    //     }
    // }
    return ru;
}

std::vector<Tensor> CANDY::DAGNNIndex::searchTensor(torch::Tensor& q, int64_t k) {
    auto idx = searchIndex(q, k);
    return getTensorByIndex(idx, k);
}

std::vector<torch::Tensor> CANDY::DAGNNIndex::getTensorByIndex(std::vector<faiss::idx_t> &idx, int64_t k) {
    int64_t size = idx.size()/k;
    std::vector<torch::Tensor> ru(size);
    for(int64_t i=0; i<size; i++) {
        ru[i] = torch::zeros({k, vecDim});
        for(int64_t j=0; j<k; j++) {

            int64_t tempIdx = idx[i*k+j];

            float tempSlice[vecDim];

            dagnn->storage->reconstruct(tempIdx, tempSlice);
            auto tempTensor = torch::from_blob(tempSlice, {1, vecDim});

            if(tempIdx>=0) {
                ru[i].slice(0,j,j+1) = tempTensor;
            }
        }
    }
    return ru;
}





