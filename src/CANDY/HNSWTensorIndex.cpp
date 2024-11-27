//
// Created by Isshin on 2024/1/16.
//
#include <CANDY/HNSWTensorIndex.h>

bool CANDY::HNSWTensorIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  assert(cfg);
 // is_NSW = cfg->tryI64("is_NSW", 0, true);
  vecDim = cfg->tryI64("vecDim", 768, true);
  int64_t  vecVolume =  cfg->tryI64("vecVolume", 768, true)+100;
  M_ = cfg->tryI64("maxConnection", 32, true);
  std::string metricType = cfg->tryString("metricType", "L2", true);
  faissMetric = faiss::METRIC_L2;
  if (metricType == "dot" || metricType == "IP" || metricType == "cossim") {
      faissMetric = faiss::METRIC_INNER_PRODUCT;
  }
  hnsw.init(vecVolume,M_,M_*2,0);
  return true;
}

bool CANDY::HNSWTensorIndex::deleteTensor(torch::Tensor &t, int64_t k) {
  // TODO: impl
  return false;
}

bool CANDY::HNSWTensorIndex::reviseTensor(torch::Tensor &t, torch::Tensor &w) {
  // TODO: impl
  return false;
}
bool CANDY::HNSWTensorIndex::insertTensor(torch::Tensor &t) {
  auto n = t.size(0);
  for (int64_t i=0;i<n;i++) {
    auto rowI = t.slice(0,i,i+1);
    hnsw.add(rowI.squeeze());
   // std::cout<<"Load row"+ to_string(i)<<std::endl;
  }
  ntotal += n;
  return true;
}
std::vector<torch::Tensor> CANDY::HNSWTensorIndex::searchTensor(torch::Tensor &q,
                                                               int64_t k) {
  return hnsw.multiQuerySearch(q,k);
}