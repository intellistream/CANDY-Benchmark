/*! \file YinYangGraphIndex.cpp*/
//
// Created by tony on 25/05/23.
//

#include <CANDY/YinYangGraphIndex.h>
#include <Utils/UtilityFunctions.h>
#include <time.h>
#include <chrono>
#include <assert.h>
#include <map>


bool CANDY::YinYangGraphIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  FlatIndex::setConfig(cfg);
  distanceFunc = distanceIP;
  if (faissMetric != faiss::METRIC_INNER_PRODUCT) {
    INTELLI_WARNING("Switch to L2");
    distanceFunc = distanceL2;
  }
  maxConnection = cfg->tryI64("maxConnection", 256, true);
  if (maxConnection<2) {
    maxConnection = 4;
  }
  candidateTimes = cfg->tryI64("candidateTimes", 1, true);
  skeletonRows = cfg->tryI64("skeletonRows",-1, true);
  rowNNZTensor = torch::zeros({initialVolume, 1}, torch::kInt64);
  similarityTensor = torch::zeros({initialVolume, maxConnection}, torch::kInt64);
  DCOBatchSize = cfg->tryI64("DCOBatchSize", -1, true);
  if (torch::cuda::is_available()) {
    cudaDevice = cfg->tryI64("cudaDevice", -1, true);
    INTELLI_INFO("Cuda is detected. and use this cuda device for DCO:" + std::to_string(cudaDevice));
  }
  lastNNZ = -1;
  lastNNZSim = -1;
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
  for(int64_t i=0;i<rows;i++) {
    ru[i] = torch::zeros({k, vecDim});
    auto qi = q.slice(0, i, i + 1);
    auto maxSimRow = searchSingleRowIdx(qi, 1000);
    auto candidateIdx = similarityTensor.slice(0,maxSimRow,maxSimRow+1);
    auto collectedRows = collectDataRows(candidateIdx);
    auto candidateDistance = distanceFunc(collectedRows, q, -1, this);
    auto topk_result = torch::topk(candidateDistance, k, 1, true, true);
    torch::Tensor topk_indices = std::get<1>(topk_result);
    std::cout<<"top k:\n"<<topk_indices<<std::endl;
    std::cout<<"candidate idx:\n"<<candidateIdx<<std::endl;
    auto array = topk_indices.contiguous().data_ptr<int64_t>();
    for (int64_t j = 0; j < k; j++) {
      int64_t tempIdx = array[j];
      if (tempIdx >= 0) {
        ru[i].slice(0, j, j + 1) = collectedRows.slice(0, tempIdx, tempIdx + 1);
      }
    }
  }
  return ru;
}
torch::Tensor CANDY::YinYangGraphIndex::genCompressedSimilarityMask (torch::Tensor &t, int64_t cols,bool circle) {
  /**
   * @brief 1. compute the similarity in the beginning tensors
   */
  int64_t num_rows = t.size(0);
  auto similarityMat = distanceFunc(t,t,cudaDevice,this);
  if(circle) {
    // Get the diagonal elements B[i][i]
    torch::Tensor diag_elements = similarityMat.diag();

    // Assign B[i+1][i] = B[i][i]
    similarityMat.index_put_({torch::arange(1, similarityMat.size(0), torch::kInt64).to(similarityMat.device()),
                              torch::arange(0, similarityMat.size(0) - 1, torch::kInt64).to(similarityMat.device())},
                             diag_elements.slice(0, 0, similarityMat.size(0) - 1));

    // Assign B[i][i+1] = B[i][i]
    similarityMat.index_put_({torch::arange(0, similarityMat.size(0) - 1, torch::kInt64).to(similarityMat.device()),
                              torch::arange(1, similarityMat.size(0), torch::kInt64).to(similarityMat.device())},
                             diag_elements.slice(0, 0, similarityMat.size(0) - 1));
    similarityMat[num_rows - 1][0] = similarityMat[num_rows - 1][num_rows - 1];
  }
  std::cout<< "similarity is\n"<<similarityMat<<std::endl;
  /**
   * @brief 2. only preserve the top
   */
  torch::Tensor values, indices;
  std::tie(values, indices) = torch::topk(similarityMat, cols, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  torch::Tensor B = torch::zeros_like(t);
  std::cout<< "indices is\n"<<indices<<std::endl;
  return indices;
}
bool CANDY::YinYangGraphIndex::loadInitialTensorInline(torch::Tensor &t) {
  auto indices = genCompressedSimilarityMask(t,maxConnection);
  std::cout<< "indices is\n"<<indices<<std::endl;
  if(indices.is_cuda()){
    // Move tensors to GPU 1
    indices = indices.to(torch::kCPU);
  }
  INTELLI::IntelliTensorOP::appendRowsBufferMode(&similarityTensor, &indices, &lastNNZSim, expandStep);
  INTELLI::IntelliTensorOP::appendRowsBufferMode(&dbTensor, &t, &lastNNZ, expandStep);
  return true;
}
bool CANDY::YinYangGraphIndex::loadInitialTensor(torch::Tensor &t) {
  return loadInitialTensorInline(t);
}
torch::Tensor CANDY::YinYangGraphIndex::collectDataRows(torch::Tensor &idx) {

  torch::Tensor indices = idx.flatten().to(dbTensor.device());
  // Collect the values from B using the indices in A
  torch::Tensor collectedRows = dbTensor.index_select(0, indices);
  return collectedRows;
}
int64_t CANDY::YinYangGraphIndex::searchSingleRowIdx(torch::Tensor &q,int64_t maxIter) {

  // Initialize an std::map to track visited rows
  std::map<int64_t, bool> visited;


  int64_t nextRow = 0;
  float thisSim = -99999.0;
  // Iterate up to max_iter times
  for (int64_t iter = 0; iter < maxIter; ++iter) {
    // Check if the next row has been visited
    if (visited.find(nextRow) == visited.end() || !visited[nextRow]) {
      // Get the row P[nextRow]
      torch::Tensor idxRow = similarityTensor.slice(0,nextRow,nextRow+1);
      // Convert A to a 1D tensor (flatten it) and ensure it's on the same device as B
      std::cout<<"collecting rows:\n"<<idxRow<<std::endl;
      // Collect the values from B using the indices in A
      torch::Tensor collectedRows = collectDataRows(idxRow);
      auto tempSimilarity = distanceFunc(collectedRows,q,-1,this);
      // Mark this row as visited
      // Mark this row as visited
      visited[nextRow] = true;
      // Find the index of the maximum absolute value in the row
      auto j = torch::argmax(tempSimilarity).item<int64_t>();
      std::cout<<"temp similarity:\n"<<tempSimilarity<<std::endl;

      // Compare similarity and update if necessary
      if (tempSimilarity[0][j].item<float>() > thisSim) {
        thisSim = tempSimilarity[0][j].item<float>();
        nextRow = idxRow[0][j].item<float>();
        std::cout<<"Now select"<<nextRow<<std::endl;
      } else {
        break; // Exit the loop if similarity doesn't improve
      }
    } else {
      // Increment nextRow and handle bounds
      nextRow = nextRow + 1;
      if (nextRow >= dbTensor.size(0)) {
        break; // Exit if we've gone out of bounds
      }
    }
  }
  std::cout<<"Finally select"<<nextRow<<std::endl;
  return nextRow;
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