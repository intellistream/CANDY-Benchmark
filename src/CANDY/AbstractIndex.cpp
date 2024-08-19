/*! \file AbstractIndex.cpp*/
//
// Created by tony on 25/05/23.
//

#include <CANDY/AbstractIndex.h>
#include <Utils/UtilityFunctions.h>
#include <time.h>
#include <chrono>
#include <assert.h>
void CANDY::AbstractIndex::reset() {

}
bool CANDY::AbstractIndex::offlineBuild(torch::Tensor &t) {
  return false;
}
bool CANDY::AbstractIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  std::string metricType = cfg->tryString("metricType", "L2", true);
  faissMetric = faiss::METRIC_L2;
  if (metricType == "dot" || metricType == "IP" || metricType == "cossim") {
    faissMetric = faiss::METRIC_INNER_PRODUCT;
  }

  return true;
}
bool CANDY::AbstractIndex::setConfigClass(INTELLI::ConfigMap cfg) {
  INTELLI::ConfigMapPtr cfgPtr=newConfigMap();
  cfgPtr->loadFrom(cfg);
  return setConfig(cfgPtr);
}

bool CANDY::AbstractIndex::setFrozenLevel(int64_t frozenLv) {
  assert(frozenLv >= 0);
  return false;
}
bool CANDY::AbstractIndex::insertTensor(torch::Tensor &t) {
  assert(t.size(1));
  return false;
}
bool CANDY::AbstractIndex::insertStringObject(torch::Tensor &t, std::vector<std::string> &strs) {
  assert(t.size(1));
  assert (strs.size());
  return false;
}

bool CANDY::AbstractIndex::loadInitialTensor(torch::Tensor &t) {
  return insertTensor(t);
}
bool CANDY::AbstractIndex::loadInitialStringObject(torch::Tensor &t, std::vector<std::string> &strs) {
  return insertStringObject(t, strs);
}
bool CANDY::AbstractIndex::deleteTensor(torch::Tensor &t, int64_t k) {
  assert(t.size(1));
  assert(k > 0);
  return false;
}
bool CANDY::AbstractIndex::deleteTensorByIndex(std::vector<faiss::idx_t> &idx){
    return false;
}
bool CANDY::AbstractIndex::deleteStringObject(torch::Tensor &t, int64_t k) {
  assert(t.size(1));
  assert(k > 0);
  return false;
}
bool CANDY::AbstractIndex::reviseTensor(torch::Tensor &t, torch::Tensor &w) {
  assert(t.size(1) == w.size(1));
  return false;
}
std::vector<faiss::idx_t> CANDY::AbstractIndex::searchIndex(torch::Tensor q, int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<faiss::idx_t> ru(1);
  return ru;
}
std::vector<std::vector<std::string>> CANDY::AbstractIndex::searchStringObject(torch::Tensor &q, int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<std::vector<std::string>> ru(1);
  ru[0] = std::vector<std::string>(1);
  ru[0][0] = "";
  return ru;
}
std::vector<torch::Tensor> CANDY::AbstractIndex::getTensorByIndex(std::vector<faiss::idx_t> &idx, int64_t k) {
  assert(k > 0);
  assert(idx.size());
  std::vector<torch::Tensor> ru(1);
  ru[0] = torch::rand({1, 1});
  return ru;
}
torch::Tensor CANDY::AbstractIndex::rawData() {
  return torch::rand({1, 1});
}

std::vector<torch::Tensor> CANDY::AbstractIndex::searchTensor(torch::Tensor &q, int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<torch::Tensor> ru(1);
  ru[0] = torch::rand({1, 1});
  return ru;
}
bool CANDY::AbstractIndex::startHPC() {
  return false;
}

bool CANDY::AbstractIndex::endHPC() {
  return false;
}
bool CANDY::AbstractIndex::waitPendingOperations() {
  return true;
}
std::tuple<std::vector<torch::Tensor>,
           std::vector<std::vector<std::string>>> CANDY::AbstractIndex::searchTensorAndStringObject(torch::Tensor &q,
                                                                                                    int64_t k) {
  auto ruT = searchTensor(q, k);
  auto ruS = searchStringObject(q, k);
  std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string>>> ru(ruT, ruS);
  return ru;
}
bool CANDY::AbstractIndex::loadInitialTensorAndQueryDistribution(torch::Tensor &t, torch::Tensor &query) {
  assert(query.size(0) > 0);
  return loadInitialTensor(t);
}