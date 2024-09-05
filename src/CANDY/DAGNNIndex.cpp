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

    /// Setting config for DynamicTuneParams at initialization
    dp.efConstruction = cfg->tryI64("efConstruction", 40, true);
    dp.efSearch = cfg->tryI64("efSearch", 16, true);
    dp.bottom_connections_upper_bound = cfg->tryI64("bottom_connections_upper_bound", 64, true);
    dp.bottom_connections_lower_bound = cfg->tryI64("bottom_connections_lower_bound", 32, true);
    dp.distance_computation_opt = cfg->tryI64("distance_computation_opt", 0, true);
    dp.rng_alpha = cfg->tryDouble("rng_alpha", 1.0, true);
    dp.clusterExpansionStep = cfg->tryI64("clusterExpansionStep", 2, true);
    dp.clusterInnerConnectionThreshold = cfg->tryDouble("clusterInnerConnectionThreshold", 0.5, true);
    dp.optimisticN = cfg->tryI64("optimisticN", 16, true);
    dp.discardN = cfg->tryI64("discardN", 0, true);
    dp.discardClusterN = cfg->tryI64("discardClusterN", 32, true);
    dp.discardClusterProp = cfg->tryDouble("discardClusterProp", 0.3, true);
    dp.degree_std_range = cfg->tryDouble("degree_std_range", 1.5, true);
    dp.degree_allow_range = cfg->tryDouble("degree_allow_range", 0.5, true);
    dp.degree_lift_range = cfg->tryDouble("degree_lift_range", 1.75, true);
    dp.sparsePreference = cfg->tryI64("sparsePreference", 1, true);
    dp.neighborDistanceThreshold = cfg->tryDouble("neighborDistanceThreshold", 0.5, true);
    dp.expiration_timestamp = cfg->tryI64("expiration_timestamp", 450, true);
    dp.max_backtrack_steps = cfg->tryI64("max_backtrack_steps", 20, true);
    dp.steps_above_avg = cfg->tryI64("steps_above_avg", 50, true);
    dp.steps_above_max = cfg->tryI64("steps_above_max", 20, true);
    dp.nb_navigation_paths = cfg->tryI64("nb_navigation_paths", 16, true);


    dagnn = new CANDY::DynamicTuneHNSW(M, vecDim, metric, dp);
    dagnn->is_training = cfg->tryI64("is_training", 1, true);
    dagnn->is_datamining = cfg->tryI64("is_datamining", 0, true);
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

bool CANDY::DAGNNIndex::deleteTensor(torch::Tensor &t, int64_t k) {

    return true;
}

bool CANDY::DAGNNIndex::deleteTensorByIndex(torch::Tensor &t) {
    return true;
}





