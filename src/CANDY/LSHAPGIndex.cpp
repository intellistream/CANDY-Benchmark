//
// Created by Isshin on 2024/5/31.
//
#include <CANDY/LSHAPGIndex.h>
int _lsh_UB=0;
int _G_COST=0;
int _g_dist_mes=0;
namespace CANDY{
bool LSHAPGIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  if(this->faissMetric==faiss::METRIC_INNER_PRODUCT) {
    _g_dist_mes=1;
    INTELLI_INFO("switch into inner product");
  }
  vecDim = cfg->tryI64("vecDim", 768, true);
  flatBuffer.setConfig(cfg);
  return true;
}
bool  LSHAPGIndex::loadInitialTensor(torch::Tensor &t) {
  prep=Preprocess(vecDim);
  auto tc=t.clone();
  prep.load_data(tc);
  Parameter param1(prep, L, K, 1.0f);
  divG = new divGraph(prep, param1, T, efC, pC, pQ);
  flatBuffer.loadInitialTensor(t);
  return true;
}

std::vector<torch::Tensor> LSHAPGIndex::searchTensor(torch::Tensor &q, int64_t k) {
  size_t tensors = (size_t) q.size(0);
  std::vector<torch::Tensor> ru(tensors);
  auto rawDB=flatBuffer.rawData();
  for (size_t i=0;i<tensors;i++) {
    auto rowI=q.slice(0,i,i+1);
    ru[i]=torch::zeros({k,vecDim});
    queryN qn =  queryN(c, k, divG->myData,rowI, beta);
    divG->knn(&qn);
    auto knnRes=qn.res;
    for (int j=0;j<k;j++){
      int64_t idx=knnRes[j].id;
      ru[i].slice(0,j,j+1)=rawDB.slice(0,idx,idx+1);
    }
  }
  return ru;
}
bool LSHAPGIndex::insertTensor(torch::Tensor &t) {
  auto tc=t.clone();
  divG->appendTensor(tc,&prep);
  flatBuffer.insertTensor(t);
  return true;
}

}