//
// Created by Isshin on 2024/5/31.
//
#include<CANDY/LSHAPGIndex.h>

bool CANDY::LSHAPGIndex::setConfig(INTELLI::ConfigMapPtr cfg){
    AbstractIndex::setConfig(cfg);
    vecDim = cfg->tryI64("vecDim", 768, true);
    /// other params later;
    prep.data.dim = vecDim;

    dbTensor = torch::zeros({0, vecDim});
    lastNNZ = -1;
    expandStep = 100;
    prep.data.N = 0;
    prep.data.query_size = 0;

    return true;
}

bool CANDY::LSHAPGIndex::loadInitialTensor(torch::Tensor &t){
    bool success = INTELLI::IntelliTensorOP::appendRowsBufferMode(&dbTensor, &t, &lastNNZ, expandStep);
    assert(success);
    auto new_data = t.contiguous().data_ptr<float>();
    auto new_data_size = t.size(0);
    prep.insert_data(new_data, new_data_size);
    Parameter param1(prep, L,K,1.0f);
    param1.W = 0.3f;
    gLsh = nullptr;


    divG = new divGraph(prep, param1, "divGraph"/*unused*/,T,efC,pC,pQ);
    if(fsG!=nullptr) {
      free(fsG);
    }
    fsG = new fastGraph(divG);

    return true;
}

bool CANDY::LSHAPGIndex::insertTensor(torch::Tensor &t){
    bool success = INTELLI::IntelliTensorOP::appendRowsBufferMode(&dbTensor, &t, &lastNNZ, expandStep);
    assert(success);
    auto new_data = t.contiguous().data_ptr<float>();
    auto new_data_size = t.size(0);
    prep.insert_data(new_data, new_data_size);


    Parameter param1(prep, L,K,1.0f);
    param1.W = 0.3f;
    gLsh = nullptr;

    if(divG) free(divG);
    divG = new divGraph(prep, param1, "divGraph"/*unused*/,T,efC,pC,pQ);


    // divG->oneByOneInsert();
    // if(fsG!=nullptr) {
    //     free(fsG);
    // }
    fsG = new fastGraph(divG);

    return true;
}

std::vector<faiss::idx_t> CANDY::LSHAPGIndex::searchIndex(torch::Tensor q, int64_t k){
    k_ = k;
    auto querySize = q.size(0);
    auto query_data = q.contiguous().data_ptr<float>();
    prep.set_query(query_data, q.size(0));
    if(divG) divG->ef = k+150;

    std::string f1 = "divgraph";
    std::string f2="fold";
    auto results = search_candy(c,k,divG,prep,beta,2);
    //unsigned num0 = results[0].res.size();
    //printf(" result size %d\n", num0);
    printf("results:\n");
    std::vector<faiss::idx_t> ru(k*querySize);
    printf("result set size :%lu\n",results.size());
    for(int i=0; i<querySize; i++) {
        auto res=results[i];
        printf("query result size: %lu\n",res->res.size());
        for(int j=0; j<k; j++) {
            ru[i*k+j]=res->res[j].id;
        }
    }
    return ru;

}

std::vector<torch::Tensor> CANDY::LSHAPGIndex::searchTensor(torch::Tensor &q, int64_t k){
    auto idx = searchIndex(q,k);
    return getTensorByIndex(idx,k);

}

std::vector<torch::Tensor> CANDY::LSHAPGIndex::getTensorByIndex(std::vector<faiss::idx_t> &idx, int64_t k){
    int64_t size = idx.size() / k;
    std::vector<torch::Tensor> ru(size);
    for (int64_t i = 0; i < size; i++) {
        ru[i] = torch::zeros({k, vecDim});
        for (int64_t j = 0; j < k; j++) {
            int64_t tempIdx = idx[i * k + j];

            if (tempIdx >= 0) {
                ru[i].slice(0, j, j + 1) = dbTensor.slice(0, tempIdx, tempIdx + 1);

            };
        }
    }
    return ru;

}

