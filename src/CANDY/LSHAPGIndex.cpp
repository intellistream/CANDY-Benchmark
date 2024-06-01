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
  Preprocess prep(vecDim);
  prep.load_data(t);
  Parameter param1(prep, L, K, 1.0f);
  divG = new divGraph(prep, param1, T, efC, pC, pQ);
  flatBuffer.loadInitialTensor(t);
  return true;
}
/*std::vector<torch::Tensor> LSHAPGIndex::searchTensor(torch::Tensor &q, int64_t k) {

}*/

}