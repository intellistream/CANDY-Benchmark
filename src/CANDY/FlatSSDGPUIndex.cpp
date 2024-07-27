/*! \file FlatSSDGPUIndex.cpp*/
//
// Created by tony on 25/05/23.
//
#include <include/spdk_config.h>
#if CANDY_SPDK == 1
#include <CANDY/FlatSSDGPUIndex.h>
#include <Utils/UtilityFunctions.h>
#include <time.h>
#include <chrono>
#include <assert.h>
#include <Utils/IntelliLog.h>
#include <CANDY/CANDYObject.h>
#include <vector>
#include <algorithm>
#include <utility>
bool CANDY::FlatSSDGPUIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  if (faissMetric != faiss::METRIC_INNER_PRODUCT) {
    INTELLI_WARNING("I can only deal with inner product distance");
  }
  vecDim = cfg->tryI64("vecDim", 768, true);
  SSDBufferSize = cfg->tryI64("SSDBufferSize", 1000, true);
  sketchSize = cfg->tryI64("sketchSize", 10, true);
  DCOBatchSize = cfg->tryI64("DCOBatchSize", SSDBufferSize, true);
  std::string ammAlgo = cfg->tryString("ammAlgo", "mm", true);
  INTELLI_INFO("Size of DCO=" + std::to_string(DCOBatchSize));
  if (ammAlgo == "crs") {
    ammType = 1;
    INTELLI_INFO("Use crs for amm, sketch size=" + std::to_string(sketchSize));
  } else if (ammAlgo == "smp-pca") {
    ammType = 2;
    INTELLI_INFO("Use smp-pca for amm, sketch size=" + std::to_string(sketchSize));
  } else {
    ammType = 0;
  }

  return true;
}
bool CANDY::FlatSSDGPUIndex::startHPC() {
  ssd.setupEnv();
  dmBuffer.init(vecDim,SSDBufferSize,0,ssd.getNameSpaceSize()/2,&ssd);
  return true;
}
bool CANDY::FlatSSDGPUIndex::endHPC() {
  dmBuffer.clear();
  ssd.cleanEnv();
  return true;
}
void CANDY::FlatSSDGPUIndex::reset() {

}
bool CANDY::FlatSSDGPUIndex::insertTensor(torch::Tensor &t) {
  if(t.size(0)>SSDBufferSize) {
    int64_t total_vectors = t.size(0);
    for (int64_t startPos = 0; startPos < total_vectors; startPos += SSDBufferSize) {
      int64_t endPos = std::min(startPos + SSDBufferSize, total_vectors);
      auto tempTensor=t.slice(0,startPos,endPos);
      dmBuffer.appendTensor(tempTensor);
    }
    return  true;
  }
  else {
    return dmBuffer.appendTensor(t);
  }

}

bool CANDY::FlatSSDGPUIndex::deleteTensor(torch::Tensor &t, int64_t k) {

  int64_t rows = t.size(0);
  for (int64_t i = 0; i < rows; i++) {
    auto rowI = t.slice(0, i, i + 1).contiguous();
    auto idx = findTopKClosest(rowI,1,DCOBatchSize);
    if (0 <= idx[0]) {
      dmBuffer.deleteTensor(idx[0],idx[0]+1);
      //INTELLI::IntelliTensorOP::editRows(&dbTensor, &rowW, (int64_t) idx);
    }
  }
  return true;
}
static
void mergeTopK(std::vector<std::pair<float, int64_t>>& topK, const std::vector<std::pair<float, int64_t>>& newResults, int64_t top_k) {
  topK.insert(topK.end(), newResults.begin(), newResults.end());
  std::sort(topK.begin(), topK.end(), [](const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
    return a.first > b.first;
  });
  // Keep only the top_k elements
  if (topK.size() > top_k) {
    topK.resize(top_k);
  }
}
static
void mergeTopKVec(std::vector<std::vector<std::pair<float, int64_t>>>& topK, const std::vector<std::vector<std::pair<float, int64_t>>>& newResults, int64_t top_k) {
  size_t rows=topK.size();
  for(size_t i=0;i<rows;i++) {
    mergeTopK(topK[i],newResults[i],top_k);
  }
}
std::vector<int64_t> CANDY::FlatSSDGPUIndex::findTopKClosest(const torch::Tensor &query,
                                                             int64_t top_k,
                                                             int64_t batch_size
                                                             ) {
  std::vector<std::vector<std::pair<float, int64_t>>> topK;
  int64_t total_vectors = dmBuffer.size();
  int64_t queryRows = query.size(0);
  torch::Tensor transposed_query = query.t();
  for (int64_t startPos = 0; startPos < total_vectors; startPos += batch_size) {
    int64_t endPos = std::min(startPos + batch_size, total_vectors);

    // Load batch using getTensor
    torch::Tensor dbBatch = dmBuffer.getTensor(startPos, endPos);
    //std::cout<<"DB data:\n"<<dbBatch<<std::endl;
    // Compute distances
    torch::Tensor distances = torch::matmul(dbBatch, transposed_query);
    //std::cout<<"distance :\n"<<distances.t()<<std::endl;

    // Use torch::topk to get the top_k smallest distances and their indices
    auto topk_result = torch::topk(distances.t(), top_k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
   // std::cout<<"top k :\n"<<std::get<0>(topk_result)<<std::endl;
    // Extract top_k distances and indices
    torch::Tensor topk_distances = std::get<0>(topk_result);
    torch::Tensor topk_indices = std::get<1>(topk_result)+startPos;
    //
    // Create a vector of (distance, index) pairs
    std::vector<std::vector<std::pair<float, int64_t>>> batchResults(queryRows);
    for (int64_t i=0;i<queryRows;i++) {
      for (int64_t j = 0; j < top_k; j++) {
        batchResults[i].emplace_back(topk_distances[i][j].item<float>(), topk_indices[i][j].item<int64_t>());
      }
    }
    // Merge current batch results with topK
    if (topK.empty()) {
      topK = std::move(batchResults);
    } else {
      mergeTopKVec(topK, batchResults, top_k);
    }
  }

  // Extract indices from the topK vector
  std::vector<int64_t> topKIndices(top_k*queryRows);
  for (int64_t i = 0; i < queryRows; i++) {
    for (int64_t j = 0; j < top_k; j++) {
      topKIndices[i*top_k+j]=topK[i][j].second;
    }
  }
  return topKIndices;
}
bool CANDY::FlatSSDGPUIndex::reviseTensor(torch::Tensor &t, torch::Tensor &w) {
  if (t.size(0) > w.size(0) || t.size(1) != w.size(1)) {
    return false;
  }
  auto idx = findTopKClosest(t,1,DCOBatchSize);
  int64_t rows = t.size(0);
  for (int64_t i = 0; i < rows; i++) {
    //auto rowI = t.slice(0, i, i + 1).contiguous();
    if (0 <= idx[i]) {
      auto rowW = w.slice(0, i, i + 1);
      dmBuffer.reviseTensor(idx[i],rowW);
      //INTELLI::IntelliTensorOP::editRows(&dbTensor, &rowW, (int64_t) idx);
    }
  }
  return true;
}
std::vector<torch::Tensor> CANDY::FlatSSDGPUIndex::searchTensor(torch::Tensor &q, int64_t k) {
  auto idx = findTopKClosest(q,k,DCOBatchSize);
  //std::cout<<"sorting idx"<<std::endl;

  return getTensorByStdIdx(idx, k);
}

std::vector<torch::Tensor> CANDY::FlatSSDGPUIndex::getTensorByStdIdx(std::vector<int64_t> &idx, int64_t k) {
  int64_t tensors = idx.size() / k;
  std::vector<torch::Tensor> ru(tensors);
  for (int64_t i = 0; i < tensors; i++) {
    ru[i] = torch::zeros({k, vecDim});
    for (int64_t j = 0; j < k; j++) {
      int64_t tempIdx = idx[i * k + j];
      if (tempIdx >= 0) { ru[i].slice(0, j, j + 1) = dmBuffer.getTensor(tempIdx,tempIdx+1);}
    }
  }
  return ru;
}
#endif