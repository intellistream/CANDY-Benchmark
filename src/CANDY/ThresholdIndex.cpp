/*! \file ThresholdIndex.cpp*/
//
// Created by tony on 25/05/23.
//

#include <CANDY/ThresholdIndex.h>
#include <Utils/UtilityFunctions.h>
#include <time.h>
#include <chrono>
#include <assert.h>


void CANDY::ThresholdIndex::reset() {

}

bool CANDY::ThresholdIndex::offlineBuild(torch::Tensor &t) {
    assert(t.size(1));
    auto index = new faiss::IndexFlatL2(t.size(1));
    //faiss::Index* newIndex = new faiss::IndexFlatL2(dimension);
    //index->train(t.size(0), t.data_ptr<float>());
    index->add(t.size(0), t.data_ptr<float>());
    indices.push_back(index);
    return true;
}


bool CANDY::ThresholdIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  std::string metricType = cfg->tryString("metricType", "L2", true);
  faissMetric = faiss::METRIC_L2;
  if (metricType == "dot" || metricType == "IP" || metricType == "cossim") {
    faissMetric = faiss::METRIC_INNER_PRODUCT;
  }

  dataThreshold = cfg->tryI64("dataThreshold", 10000);
  indexAlgorithm = cfg->tryString("indexAlgorithm", "HNSW");


  return false;
}
bool CANDY::ThresholdIndex::setConfigClass(INTELLI::ConfigMap cfg) {
  INTELLI::ConfigMapPtr cfgPtr=newConfigMap();
  cfgPtr->loadFrom(cfg);
  return setConfig(cfgPtr);
}
/*
bool CANDY::ThresholdIndex::setFrozenLevel(int64_t frozenLv) {
  assert(frozenLv >= 0);
  return false;
}*/
bool CANDY::ThresholdIndex::insertTensor(torch::Tensor &t) {
  assert(t.size(1));
  if (indices.empty() || indices.back()->ntotal >= dataThreshold) {
    createThresholdIndex(t.size(1));
  }
  indices.back()->add(t.size(0), t.data_ptr<float>());
  return true;
}

void CANDY::ThresholdIndex::createThresholdIndex(int64_t dimension) {
  //auto index = new faiss::IndexFlatL2(dimension);
  faiss::Index* index = new faiss::IndexFlatL2(dimension);
  indices.push_back(index);
}

/*
bool CANDY::ThresholdIndex::insertStringObject(torch::Tensor &t, std::vector<std::string> &strs) {
  assert(t.size(1));
  assert (strs.size());
  return false;
}
*/
bool CANDY::ThresholdIndex::loadInitialTensor(torch::Tensor &t) {
  return insertTensor(t);
}

std::vector<faiss::idx_t> CANDY::ThresholdIndex::searchIndex(torch::Tensor q, int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<faiss::idx_t> ru(1);
  return ru;
}

std::vector<torch::Tensor> CANDY::ThresholdIndex::searchTensor(torch::Tensor &q, int64_t k) {
    assert(k > 0);
    assert(q.size(1));

  //auto idx = searchIndex(q, k);
  std::vector<torch::Tensor> allKnnIndices ;

  for (int64_t i = 0; i < indices.size(); ++i) {
    std::vector<faiss::idx_t> knnIndices(k);
    std::vector<float> knnDistances(k);
    indices[i]->search(q.size(0), q.data_ptr<float>(), k, knnDistances.data(), knnIndices.data());
    for (faiss::idx_t index : knnIndices) {
        allKnnIndices.push_back(torch::tensor(index));
    }
    
    //allKnnIndices.insert(allKnnIndices.end(), knnIndices.begin(), knnIndices.end());
  }

  return allKnnIndices;
  //return getTensorByIndex(allKnnIndices, k);

}
/*
torch::Tensor CANDY::ThresholdIndex::rawData() {
  return torch::rand({1, 1});
}

bool CANDY::ThresholdIndex::startHPC() {
  return false;
}

bool CANDY::ThresholdIndex::endHPC() {
  return false;
}
bool CANDY::ThresholdIndex::waitPendingOperations() {
  return true;
}
std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string>>> CANDY::ThresholdIndex::searchTensorAndStringObject(torch::Tensor &q, int64_t k) {
  auto ruT = searchTensor(q, k);
  auto ruS = searchStringObject(q, k);
  std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string>>> ru(ruT, ruS);
  return ru;
}
bool CANDY::ThresholdIndex::loadInitialTensorAndQueryDistribution(torch::Tensor &t, torch::Tensor &query) {
  assert(query.size(0) > 0);
  return loadInitialTensor(t);
}

std::vector<std::vector<std::string>> CANDY::ThresholdIndex::searchStringObject(torch::Tensor &q, int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<std::vector<std::string>> ru(1);
  ru[0] = std::vector<std::string>(1);
  ru[0][0] = "";
  return ru;
}

std::vector<torch::Tensor> CANDY::ThresholdIndex::getTensorByIndex(std::vector<faiss::idx_t> &idx, int64_t k) {
  assert(k > 0);
  assert(idx.size());
  std::vector<torch::Tensor> ru(1);
  ru[0] = torch::rand({1, 1});
  return ru;
}

bool CANDY::ThresholdIndex::loadInitialStringObject(torch::Tensor &t, std::vector<std::string> &strs) {
  return insertStringObject(t, strs);
}
bool CANDY::ThresholdIndex::deleteTensor(torch::Tensor &t, int64_t k) {
  assert(t.size(1));
  assert(k > 0);
  return false;
}

bool CANDY::ThresholdIndex::deleteStringObject(torch::Tensor &t, int64_t k) {
  assert(t.size(1));
  assert(k > 0);
  return false;
}
bool CANDY::ThresholdIndex::reviseTensor(torch::Tensor &t, torch::Tensor &w) {
  assert(t.size(1) == w.size(1));
  return false;
}
*/