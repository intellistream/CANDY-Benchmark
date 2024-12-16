/*! \file ThresholdIndex.cpp*/
//
// Created by tony on 25/05/23.
//

#include <CANDY/ThresholdIndex.h>
#include <Utils/UtilityFunctions.h>
#include <time.h>
#include <chrono>
#include <assert.h>
#include <CANDY/IndexTable.h>



void CANDY::ThresholdIndex::reset() {

}

bool CANDY::ThresholdIndex::offlineBuild(torch::Tensor &t, std::string nameTag) {
    assert(t.size(1));
    if (indices.empty() || dataVolume >= dataThreshold) {
        createThresholdIndex(t.size(1), nameTag);
    }
    //auto index = new faiss::IndexFlatL2(t.size(1)); 
    //index->add(t.size(0), t.data_ptr<float>());
    //indices.push_back(index);

    indices.back()->insertTensor(t);
    dataVolume++;
    
    return true;
}


bool CANDY::ThresholdIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  std::string metricType = cfg->tryString("metricType", "L2", true);
  faissMetric = faiss::METRIC_L2;
  if (metricType == "dot" || metricType == "IP" || metricType == "cossim") {
    faissMetric = faiss::METRIC_INNER_PRODUCT;
  }

  dataThreshold = cfg->tryI64("dataThreshold", 100);
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
bool CANDY::ThresholdIndex::insertTensor_th(torch::Tensor &t ,std::string nameTag) {
  assert(t.size(1));
  if (indices.empty() || dataVolume >= dataThreshold) {
    createThresholdIndex(t.size(1), nameTag);
  }
  indices.back()->insertTensor(t);

  dataVolume++;
  return true;
}

void CANDY::ThresholdIndex::createThresholdIndex(int64_t dimension, std::string nameTag) {
  //auto index = new faiss::IndexFlatL2(dimension);
  //int M=32;
  //auto index = AbstractIndexPtr::createIndex(nameTag);
  
  IndexTable tab;
  auto ru= tab.getIndex(nameTag);
  
  if(ru==nullptr){
    INTELLI_ERROR("No index named "+nameTag+", return flat");
    nameTag="flat";
    INTELLI::ConfigMapPtr cfg_new = newConfigMap();
    cfg_new->edit("vecDim", (int64_t) 3);
    cfg_new->edit("M", (int64_t) 4);
    ru->setConfig(cfg_new);
    indices.push_back(tab.getIndex(nameTag));
    dataVolume=0;
    return;
  }
  INTELLI::ConfigMapPtr cfg_new = newConfigMap();
  cfg_new->edit("vecDim", (int64_t) 3);
  cfg_new->edit("M", (int64_t) 4);
  ru->setConfig(cfg_new);
  dataVolume=0;
  indices.push_back(ru);

  return;
}
/*
bool CANDY::ThresholdIndex::insertStringObject(torch::Tensor &t, std::vector<std::string> &strs) {
  assert(t.size(1));
  assert (strs.size());
  return false;
}
*/

std::vector<faiss::idx_t> CANDY::ThresholdIndex::searchIndex(torch::Tensor q, int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<faiss::idx_t> ru(1);
  return ru;
}

float getDist(const torch::Tensor& a, const torch::Tensor& b) {
    return torch::norm(a - b).item<float>();
}

std::vector<torch::Tensor> CANDY::ThresholdIndex::searchTensor_th(torch::Tensor &q, int64_t k) {
    assert(k > 0);
    assert(q.size(1));

  //auto idx = searchIndex(q, k);
  std::vector<torch::Tensor> Indicesx ;

  for (int64_t i = 0; i < indices.size(); ++i) {
    int64_t querySize = q.size(0);
    auto inx = indices[i]->searchTensor(q,k);
    //std::cout << "Index " << i << " returned " << inx.size() << " results" << endl;
    for(int64_t j=0; j< k; j++)
    {//std::cout << "Result tensor from index " << i << ": " << inx[j] << std::endl;
      Indicesx.push_back(inx[0].slice(0, j, j + 1) );
    }
    //Indicesx.insert(Indicesx.end(), inx.begin(), inx.end())

}

  if(Indicesx.size()>k)
  {
    std::vector<std::pair<float, int64_t>> dist;
    for (int64_t i = 0; i < Indicesx.size(); ++i) {
          float distance = getDist(q, Indicesx[i]);
          dist.push_back(std::make_pair(distance, i));
    }

    std::sort(dist.begin(), dist.end());
    std::vector<torch::Tensor> topK; 

    for (int64_t i = 0; i < k; ++i) {
      topK.push_back(Indicesx[dist[i].second]);
    }
    return topK;
  }

  return Indicesx;

} 

  /*std::vector<std::pair<float, faiss::idx_t>> knn;
    for (size_t i = 0; i < Indicesx.size(); ++i) {
        knn.emplace_back(Distances[i], Indicesx[i]);
    }

    std::sort(knn.begin(), knn.end(), [](const auto &a, const auto &b) {
        return a.first < b.first; 
    });

    std::vector<torch::Tensor> topK;
    for (int64_t i = 0; i < k && i < knn.size(); ++i) {
        topK.push_back(torch::tensor(knn[i].second));
    }
  */
  

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
