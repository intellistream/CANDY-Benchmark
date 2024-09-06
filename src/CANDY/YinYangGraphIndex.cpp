/*! \file YinYangGraphIndex.cpp*/
//
// Created by tony on 25/05/23.
//

#include <CANDY/YinYangGraphIndex.h>
#include <Utils/UtilityFunctions.h>
#include <time.h>
#include <chrono>
#include <assert.h>

static inline uint32_t nnzBits(uint32_t a) {
  if (a == 0) {
    return 1;
  }
  return 32 - __builtin_clz(a);
}
#ifndef ONLINEIVF_NEXT_POW_2
/**
 *  compute the next number, greater than or equal to 32-bit unsigned v.
 *  taken from "bit twiddling hacks":
 *  http://graphics.stanford.edu/~seander/bithacks.html
 */
#define ONLINEIVF_NEXT_POW_2(V)                           \
    do {                                        \
        V--;                                    \
        V |= V >> 1;                            \
        V |= V >> 2;                            \
        V |= V >> 4;                            \
        V |= V >> 8;                            \
        V |= V >> 16;                           \
        V++;                                    \
    } while(0)
#endif

#ifndef HASH
#define HASH(X, MASK, SKIP) (((X) & MASK) >> SKIP)
#endif
void CANDY::YinYangGraphIndex::genCrsIndices() {
  int64_t n = vecDim;
  // Probability distribution
  torch::Tensor probs = torch::ones(n) / n;  // default: uniform
}

bool CANDY::YinYangGraphIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  FlatIndex::setConfig(cfg);
  distanceFunc = distanceIP;
  if (faissMetric != faiss::METRIC_INNER_PRODUCT) {
    INTELLI_WARNING("Switch to L2");
    distanceFunc = distanceL2;
  }
  maxConnection = cfg->tryI64("maxConnection", 256, true);

  candidateTimes = cfg->tryI64("candidateTimes", 1, true);
  skeletonRows = cfg->tryI64("skeletonRows",-1, true);
  rowNNZTensor = torch::zeros({initialVolume, 1}, torch::kInt64);
  similarityTensor = torch::zeros({initialVolume, maxConnection}, torch::kInt64);
  DCOBatchSize = cfg->tryI64("DCOBatchSize", -1, true);
  if (torch::cuda::is_available()) {
    cudaDevice = cfg->tryI64("cudaDevice", -1, true);
    INTELLI_INFO("Cuda is detected. and use this cuda device for DCO:" + std::to_string(cudaDevice));
  }
  /**
   ** @brief generate the rotation matrix for random projection
   */
  torch::manual_seed(114514);
  return true;
}
bool CANDY::YinYangGraphIndex::insertTensor(torch::Tensor &t) {

  return true;
}

std::vector<torch::Tensor> CANDY::YinYangGraphIndex::searchTensor(torch::Tensor &q, int64_t k) {
  int64_t rows = q.size(0);
  std::vector<torch::Tensor> ru((size_t) rows);

  return ru;
}

torch::Tensor CANDY::YinYangGraphIndex::distanceIP(torch::Tensor &db,
                                                 torch::Tensor &query,
                                                 int64_t cudaDev,
                                                   YinYangGraphIndex *idx) {
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

torch::Tensor CANDY::YinYangGraphIndex::distanceL2(torch::Tensor &db0,
                                                 torch::Tensor &_q,
                                                 int64_t cudaDev,
                                                   YinYangGraphIndex *idx) {
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
    auto dist = dist_squared.sqrt(); // [n]
    result[i] = dist;
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
INTELLI::ConfigMapPtr CANDY::YinYangGraphIndex::getIndexStatistics() {
  auto cfg = AbstractIndex::getIndexStatistics();
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