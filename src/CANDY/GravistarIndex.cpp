//
// Created by tony on 24-10-21.
//
/*! \file GravistarIndex.cpp*/
//
// Created by tony on 25/05/23.
//

#include <Utils/UtilityFunctions.h>

#include <CANDY/GravistarIndex.h>
#include <Utils/UtilityFunctions.h>
#include <time.h>
#include <chrono>
#include <assert.h>
#include <Utils/IntelliLog.h>
#include <CANDY/CANDYObject.h>
#include <vector>
#include <algorithm>
#include <utility>
bool CANDY::GravistarIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  int64_t distanceMode = 0;
  distanceFunc = distanceIP;
  if (faissMetric != faiss::METRIC_INNER_PRODUCT) {
    INTELLI_WARNING("Switch to L2");
    distanceFunc = distanceL2;
    distanceMode = 1;
  }
  vecDim = cfg->tryI64("vecDim", 768, true);
  //int64_t  vecVolume = cfg->tryI64("vecVolume", 1000, true);
  memBufferSize = cfg->tryI64("memBufferSize", vecDim, true);
  sketchSize = cfg->tryI64("sketchSize", 10, true);
  DCOBatchSize = cfg->tryI64("DCOBatchSize", memBufferSize, true);
  if (torch::cuda::is_available()) {
    cudaDevice = cfg->tryI64("cudaDevice", -1, true);
    INTELLI_INFO("Cuda is detected. and use this cuda device for DCO:" + std::to_string(cudaDevice));
  }
  if (DCOBatchSize > memBufferSize && memBufferSize > 0) {
    INTELLI_WARNING("DCO batch size is not recommended to exceed mem buffer size.");
    //DCOBatchSize = memBufferSize;
  }
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
  dmBuffer.init(vecDim, memBufferSize, 0, 0, 0);
  root = newGravistar();
  root->init(vecDim,memBufferSize,0,0,0);
  root->setConstraints(cudaDevice,distanceMode,DCOBatchSize,&graviStatistics);
  return true;
}
bool CANDY::GravistarIndex::startHPC() {

  return true;
}
bool CANDY::GravistarIndex::endHPC() {
  //dmBuffer.clear();
  return true;
}
void CANDY::GravistarIndex::reset() {

}
bool CANDY::GravistarIndex::insertTensor(torch::Tensor &t) {

 int64_t rows = t.size(0);
 for(int64_t i=0;i<rows;i++){
   auto rowI = t.slice(0,i,i+1);
   auto newRoot = root->insertTensor(rowI,root);
   root = newRoot;
 }
  return true;
}
static
void mergeTopK(std::vector<std::pair<float, int64_t>> &topK,
               const std::vector<std::pair<float, int64_t>> &newResults,
               int64_t top_k) {
  topK.insert(topK.end(), newResults.begin(), newResults.end());
  std::sort(topK.begin(), topK.end(), [](const std::pair<float, int64_t> &a, const std::pair<float, int64_t> &b) {
    return a.first > b.first;
  });
  // Keep only the top_k elements
  if (topK.size() > (size_t) top_k) {
    topK.resize(top_k);
  }
}
static
void mergeTopKVec(std::vector<std::vector<std::pair<float, int64_t>>> &topK,
                  const std::vector<std::vector<std::pair<float, int64_t>>> &newResults,
                  int64_t top_k) {
  size_t rows = topK.size();
  for (size_t i = 0; i < rows; i++) {
    mergeTopK(topK[i], newResults[i], top_k);
  }
}
std::vector<int64_t> CANDY::GravistarIndex::findTopKClosest(const torch::Tensor &query,
                                                            torch::Tensor data,
                                                             int64_t top_k,
                                                             int64_t batch_size
) {
  std::vector<std::vector<std::pair<float, int64_t>>> topK;
  int64_t total_vectors = data.size(0);
  int64_t queryRows = query.size(0);
  //torch::Tensor transposed_query = query.t();
  for (int64_t startPos = 0; startPos < total_vectors; startPos += batch_size) {
    int64_t endPos = std::min(startPos + batch_size, total_vectors);

    // Load batch using getTensor
    torch::Tensor dbBatch = data.slice(0,startPos, endPos);
    //std::cout<<"DB data:\n"<<dbBatch<<std::endl;
    // Compute distances
    torch::Tensor distances = distanceFunc(dbBatch, query, cudaDevice, this);
    // torch::matmul(dbBatch, transposed_query);
    //std::cout<<"distance :\n"<<distances.t()<<std::endl;
    auto tStartTopK = std::chrono::high_resolution_clock::now();
    // Use torch::topk to get the top_k smallest distances and their indices
    auto topk_result = torch::topk(distances, top_k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);

    // std::cout<<"top k :\n"<<std::get<0>(topk_result)<<std::endl;
    // Extract top_k distances and indices
    torch::Tensor topk_distances = std::get<0>(topk_result);
    torch::Tensor topk_indices = std::get<1>(topk_result) + startPos;
    gpuComputingUs += chronoElapsedTime(tStartTopK);
    //
    if (cudaDevice > -1 && torch::cuda::is_available()) {
      auto tStart = std::chrono::high_resolution_clock::now();
      topk_distances = topk_distances.to(torch::kCPU);
      topk_indices = topk_indices.to(torch::kCPU);
      gpuCommunicationUs += chronoElapsedTime(tStart);
    }
    // Create a vector of (distance, index) pairs
    std::vector<std::vector<std::pair<float, int64_t>>> batchResults(queryRows);
    for (int64_t i = 0; i < queryRows; i++) {
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
  std::vector<int64_t> topKIndices(top_k * queryRows);
  for (int64_t i = 0; i < queryRows; i++) {
    for (int64_t j = 0; j < top_k; j++) {
      topKIndices[i * top_k + j] = topK[i][j].second;
    }
  }
  return topKIndices;
}
/*
bool CANDY::GravistarIndex::reviseTensor(torch::Tensor &t, torch::Tensor &w) {
  if (t.size(0) > w.size(0) || t.size(1) != w.size(1)) {
    return false;
  }
  auto idx = findTopKClosest(t, 1, DCOBatchSize);
  int64_t rows = t.size(0);
  for (int64_t i = 0; i < rows; i++) {
    //auto rowI = t.slice(0, i, i + 1).contiguous();
    if (0 <= idx[i]) {
      auto rowW = w.slice(0, i, i + 1);
      dmBuffer.reviseTensor(idx[i], rowW);
      //INTELLI::IntelliTensorOP::editRows(&dbTensor, &rowW, (int64_t) idx);
    }
  }
  return true;
}*/
bool CANDY::GravistarIndex::resetIndexStatistics() {
  dmBuffer.clearStatistics();
  gpuComputingUs = 0;
  gpuCommunicationUs = 0;
  return true;
}
INTELLI::ConfigMapPtr CANDY::GravistarIndex::getIndexStatistics() {
  auto cfg = AbstractIndex::getIndexStatistics();
  cfg->edit("hasExtraStatistics", (int64_t) 1);
  /**
   * @brief count of memory access
   */
  cfg->edit("totalMemReadCnt", (int64_t) dmBuffer.getMemoryReadCntTotal());
  cfg->edit("missMemReadCnt", (int64_t) dmBuffer.getMemoryReadCntMiss());
  double memMissHitRead = dmBuffer.getMemoryReadCntMiss();
  memMissHitRead = memMissHitRead / dmBuffer.getMemoryReadCntTotal();
  cfg->edit("memMissRead", (double) memMissHitRead);
  cfg->edit("totalMemWriteCnt", (int64_t) dmBuffer.getMemoryWriteCntTotal());
  cfg->edit("missMemWriteCnt", (int64_t) dmBuffer.getMemoryWriteCntMiss());
  double memMissHitWrite = dmBuffer.getMemoryWriteCntMiss();
  memMissHitWrite = memMissHitWrite / dmBuffer.getMemoryWriteCntTotal();
  cfg->edit("memMissWrite", (double) memMissHitWrite);
  /**
   * @brief gpu statistics
   */
  if (cudaDevice > -1 && torch::cuda::is_available()) {
    cfg->edit("gpuCommunicationUs", (int64_t) gpuCommunicationUs);
    cfg->edit("gpuComputingUs", (int64_t) gpuComputingUs);
  } else {
    cfg->edit("cpuComputingUs", (int64_t) gpuComputingUs);
  }
  return cfg;
}

std::vector<torch::Tensor> CANDY::GravistarIndex::searchTensor(torch::Tensor &q, int64_t k) {

  int64_t rows = q.size(0);
  std::vector<torch::Tensor> ru(rows);
  for(int64_t i=0;i<rows;i++){
    ru[i] = torch::zeros({k, vecDim});
    auto rowI = q.slice(0,i,i+1);
    auto gs = root->findGravistar(rowI,DCOBatchSize,root);
    auto rootData = root->getTensor(0,root->size());
    std::cout<<"tensor at root"<<rootData<<std::endl;
    std::cout<<"root property"<<root->isLastTier()<<std::endl;
    auto data = gs->getTensor(0,gs->size());
    auto idx = findTopKClosest(q,data, k, DCOBatchSize);
    ru[i]=getTensorByStdIdx(idx,k,data)[0];
  }
  return ru;
}

std::vector<torch::Tensor> CANDY::GravistarIndex::getTensorByStdIdx(std::vector<int64_t> &idx, int64_t k,torch::Tensor data){
  int64_t tensors = idx.size() / k;
  std::vector<torch::Tensor> ru(tensors);
  for (int64_t i = 0; i < tensors; i++) {
    ru[i] = torch::zeros({k, vecDim});
    for (int64_t j = 0; j < k; j++) {
      int64_t tempIdx = idx[i * k + j];
      if (tempIdx >= 0) { ru[i].slice(0, j, j + 1) = data.slice(0,tempIdx, tempIdx + 1); }
    }
  }
  return ru;
}
torch::Tensor CANDY::GravistarIndex::distanceIP(torch::Tensor db,
                                                 torch::Tensor query,
                                                 int64_t cudaDev,
                                                 GravistarIndex *idx) {
  torch::Tensor q0 = query;
  torch::Tensor dbTensor = db;
  int64_t compTime = 0, commTime = 0;
  auto tStart = std::chrono::high_resolution_clock::now();
  if (cudaDev > -1 && torch::cuda::is_available()) {
    // Move tensors to GPU 1
    auto device = torch::Device(torch::kCUDA, cudaDev);
    q0 = q0.to(device);
    dbTensor = dbTensor.to(device);
    commTime = chronoElapsedTime(tStart);
    idx->gpuCommunicationUs += commTime;
  }
  torch::Tensor distances = torch::matmul(dbTensor, q0.t());
  auto ru = distances.t();
  compTime = chronoElapsedTime(tStart) - commTime;
  idx->gpuComputingUs += compTime;
  /*if(cudaDev>-1&&torch::cuda::is_available()){
   ru = ru.to(torch::kCPU);
  }*/
  return ru;
}

torch::Tensor CANDY::GravistarIndex::distanceL2(torch::Tensor db0,
                                                 torch::Tensor _q,
                                                 int64_t cudaDev,
                                                 GravistarIndex *idx) {
  torch::Tensor dbTensor = db0;
  torch::Tensor query = _q;
  int64_t compTime = 0, commTime = 0;
  auto tStart = std::chrono::high_resolution_clock::now();
  if (cudaDev > -1 && torch::cuda::is_available()) {
    // Move tensors to GPU 1
    auto device = torch::Device(torch::kCUDA, cudaDev);
    dbTensor = dbTensor.to(device);
    query = query.to(device);
    commTime = chronoElapsedTime(tStart);
    idx->gpuCommunicationUs += commTime;
  }
  auto n = dbTensor.size(0);
  auto q = query.size(0);
  auto vecDim = dbTensor.size(1);

  // Ensure result tensor is on the same device as input tensors
  torch::Tensor result = torch::empty({q, n}, dbTensor.options());

  // Compute L2 distance using a for loop
  for (int64_t i = 0; i < q; ++i) {
    auto query_row = query[i].view({1, vecDim}); // [1, vecDim]
    auto diff = dbTensor - query_row; // [n, vecDim]
    auto dist_squared = diff.pow(2).sum(1); // [n]
    result[i] = dist_squared;
  }
  result = -result;
  compTime = chronoElapsedTime(tStart) - commTime;
  idx->gpuComputingUs += compTime;
  /*if(cudaDev>-1&&torch::cuda::is_available()){
    // Move tensors to GPU 1
    result = result.to(torch::kCPU);
  }*/
  return result;
}
