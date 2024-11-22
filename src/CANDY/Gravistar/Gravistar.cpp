//
// Created by tony on 24-11-21.
//
#include <CANDY/GravistarIndex/Gravistar.h>
#include <Utils/UtilityFunctions.h>
namespace CANDY {
void Gravistar::init(int64_t _vecDim,
                                int64_t _bufferSize,
                                int64_t _tensorBegin,
                                int64_t _u64Begin,
                                int64_t _dmaSize) {
  dmBuffer.init(_vecDim,_bufferSize+1,_tensorBegin,_u64Begin,_dmaSize);
  bufferSize = _bufferSize;
  vecDim = _vecDim;
  downTiers = std::vector<GravistarPtr>(bufferSize);
}
void  CANDY::Gravistar::setConstraints(int64_t cudaId,int64_t _distanceMode, int64_t _batchSize,GravistarStatitics * sta) {
  cudaDevice = cudaId;
  statisticsInfo = sta;
  distanceMode = _distanceMode;
  distanceFunc = distanceIP;
  batchSize = _batchSize;
  if(distanceMode!=0) {
    distanceFunc = distanceL2;
  }
  return;
}
static
torch::Tensor compute_gravi_approximation(const torch::Tensor& a) {
  /*
  使用 LibTorch API 计算近似解 b
  输入:
      a: [n, m] 的张量，每行是一个向量 a_i
  输出:
      b: [1, m] 的张量，近似解
  */

  // 检查输入张量的维度
  TORCH_CHECK(a.dim() == 2, "Input tensor 'a' must be 2D");
  /*int64_t n = a.size(0); // 行数
  int64_t m = a.size(1); // 列数*/

  // Step 1: 计算 x0 作为 a 的行均值
  torch::Tensor x0 = a.mean(0, /*keepdim=*/true); // [1, m]

  // Step 2: 计算 a_i 和 x0 的点积
  torch::Tensor dot_products = torch::matmul(a, x0.t()); // [n, 1]

  // Step 3: 构造权重矩阵 W
  torch::Tensor weights = 1.0 / torch::pow(dot_products, 2); // [n, 1]

  // 对角化权重矩阵 W
  torch::Tensor W = torch::diag(weights.view(-1)); // [n, n]

  // Step 4: 计算归一化因子
  torch::Tensor normalization_factor = 1.0 / torch::pow(torch::sum(1.0 / dot_products), 2); // 标量

  // Step 5: 计算 b
  torch::Tensor b = normalization_factor * torch::matmul(weights.t(), a); // [1, m]

  return b;
}

GravistarPtr CANDY::Gravistar::insertTensor(torch::Tensor &t,GravistarPtr root) {
 /* if (t.size(0)+dmBuffer.size() > bufferSize) {
    return false;
  } else {
    return dmBuffer.appendTensor(t);
  }*/
  auto rootRu =root;
  GravistarPtr starProb = findGravistar(t,batchSize,root);
  /**
   * @brief insert to this one
   */
  if(starProb->size()<starProb->capacity()){
    starProb->dmBuffer.appendTensor(t);
    return rootRu;
  }
  /**
   * @brief insert to upper
   */
  auto newUpptier = starProb->upperTier;
  if(newUpptier== nullptr) {
    newUpptier = root;
  }
  auto newStar = newGravistar();
  newStar->init(vecDim,bufferSize,0,0,0);
  newStar->setConstraints(cudaDevice,distanceMode,batchSize,statisticsInfo);
  while(newUpptier->size()>bufferSize-1&&newUpptier!=root){
      if(newUpptier->upperTier!= nullptr ) {
        newUpptier = newUpptier->upperTier;
      }
    }
    /**
     * @brief even the root is filled now, should summarize root and create new root
     */
    if(newUpptier==root && newUpptier->size()>bufferSize-1) {
      INTELLI_INFO("Should create new root now");
      /**
       * @brief calculate the approximate center of collapsed gravistar
       */
      auto fullTensor = root->getTensor(0,root->size());
      if (cudaDevice > -1 && torch::cuda::is_available()) {
        auto tStart0 = std::chrono::high_resolution_clock::now();
        auto device = torch::Device(torch::kCUDA, cudaDevice);
        fullTensor = fullTensor.to(device);
        statisticsInfo->gpuCommunicationUs += chronoElapsedTime(tStart0);
      }
      auto tStart1 = std::chrono::high_resolution_clock::now();
      auto summary = compute_gravi_approximation(fullTensor);
      int64_t compTime = chronoElapsedTime(tStart1);
      if (cudaDevice > -1 && torch::cuda::is_available()) {
        auto tStart2 = std::chrono::high_resolution_clock::now();
        auto device = torch::Device(torch::kCPU);
        summary = summary.to(device);
        statisticsInfo->gpuCommunicationUs += chronoElapsedTime(tStart2);
        statisticsInfo->gpuComputingUs+=compTime;
      }
      else {
        statisticsInfo->cpuComputingUs+=compTime;
      }
      std::cout<<"Summarize "<<fullTensor<<"as "<<summary<<std::endl;
      /**
       * @brief form a new star and return new root
       */
       auto summaryStar = newGravistar();
       summaryStar->init(vecDim,1,0,0,0);
       summaryStar->setConstraints(cudaDevice,distanceMode,batchSize,statisticsInfo);
      summaryStar->downTiers[0]=root;
      summaryStar->dmBuffer.appendTensor(summary);

      newUpptier = newGravistar();
      newUpptier->init(vecDim,bufferSize,0,0,0);
      newUpptier->setConstraints(cudaDevice,distanceMode,batchSize,statisticsInfo);
      newUpptier->downTiers[0]=summaryStar;
      newUpptier->dmBuffer.appendTensor(summary);
      newUpptier->setToLastTier(false);
      rootRu = newUpptier;
      root->setToLastTier(false);
     /* newUpptier = newGravistar();
      newUpptier->init(vecDim,bufferSize,0,0,0);
      newUpptier->setConstraints(cudaDevice,distanceMode,batchSize,statisticsInfo);
      newUpptier->downTiers[newUpptier->size()]=root;
      newUpptier->dmBuffer.appendTensor(summary);
      newUpptier->setToLastTier(false);
      rootRu = newUpptier;
      root->setToLastTier(false);*/

    }
    else{
      INTELLI_INFO("Find position at upper");
    }
    newStar->dmBuffer.appendTensor(t);
    newStar ->upperTier = newUpptier;
    newUpptier->downTiers[newUpptier->size()] = newStar;
    newUpptier->dmBuffer.appendTensor(t);


  return rootRu;
}
torch::Tensor CANDY::Gravistar::getTensor(int64_t startPos, int64_t endPos) {
  return dmBuffer.getTensor(startPos,endPos);
}
void CANDY::Gravistar::setToLastTier(bool val) {
  lastTier = val;
}
bool  CANDY::Gravistar::isLastTier(void) {
  return  lastTier;
}

GravistarPtr  CANDY::Gravistar::findGravistar(torch::Tensor &t,int64_t _batchSize,GravistarPtr root) {
  bool reachesTheEnd = false;
  int64_t goBackToRoot = 0;
  int64_t  maxGoBackToRoot = vecDim/2;
  auto sharedThis = shared_from_this();
  auto currentGravistar = sharedThis;
  while (!currentGravistar->isLastTier()) {
    auto idx = currentGravistar->findTopKClosest(t,1,_batchSize,root);
    int64_t positionNumber = idx[0];
    if(positionNumber< this->size()) {
      currentGravistar = downTiers[positionNumber];
    //  INTELLI_WARNING("Go down");
    }
    else if(this->size()<=positionNumber&&positionNumber<this->size()+root->size()) {
      goBackToRoot++;
      if(goBackToRoot<maxGoBackToRoot){
        currentGravistar = root->downTiers[positionNumber- this->size()];
      //  INTELLI_WARNING("Go back to root");
      }
      else {
        currentGravistar = downTiers[0];
      }
    }
    else {
      currentGravistar = upperTier->downTiers[positionNumber- this->size()-root->size()];
    //  INTELLI_WARNING("Go left");
      if(currentGravistar==sharedThis) {
        currentGravistar = downTiers[0];
      }
    }
  }
  return  currentGravistar;
}


static inline
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
static inline
void mergeTopKVec(std::vector<std::vector<std::pair<float, int64_t>>> &topK,
                  const std::vector<std::vector<std::pair<float, int64_t>>> &newResults,
                  int64_t top_k) {
  size_t rows = topK.size();
  for (size_t i = 0; i < rows; i++) {
    mergeTopK(topK[i], newResults[i], top_k);
  }
}

torch::Tensor CANDY::Gravistar::constructGravityField(GravistarPtr root) {
  int64_t total_vectors=0;
  if(this->size()>0){
    total_vectors+=this->size();
  }
  if(root!= nullptr) {
    total_vectors+=root->size();
  }
  if(upperTier!= nullptr && upperTier!=root) {
    total_vectors+=upperTier->size();
  }
  torch::Tensor fieldTensor = torch::zeros({total_vectors,vecDim});
  if(this->size()>0){
   fieldTensor.slice(0,0, this->size()) = this->dmBuffer.getTensor(0, this->size());
  }
  if(root!= nullptr) {
    fieldTensor.slice(0,this->size(), this->size()+root->size()) = root->dmBuffer.getTensor(0, root->size());
  }
  if(upperTier!= nullptr && upperTier!=root) {
    fieldTensor.slice(0, this->size()+root->size(),total_vectors) = upperTier->dmBuffer.getTensor(0, root->size());
  }
  return fieldTensor;
}
std::vector<int64_t> CANDY::Gravistar::findTopKClosest(const torch::Tensor &query,
                                                                  int64_t top_k,
                                                                  int64_t batch_size,GravistarPtr root)
{
  auto fieldTensor = constructGravityField(root);
  int64_t total_vectors=fieldTensor.size(0);
  std::vector<std::vector<std::pair<float, int64_t>>> topK;
  int64_t queryRows = query.size(0);
  //torch::Tensor transposed_query = query.t();
  for (int64_t startPos = 0; startPos < total_vectors; startPos += batch_size) {
    int64_t endPos = std::min(startPos + batch_size, total_vectors);

    // Load batch using getTensor
    torch::Tensor dbBatch = fieldTensor.slice(0,startPos, endPos);
    //std::cout<<"DB data:\n"<<dbBatch<<std::endl;
    // Compute distances
    torch::Tensor distances = distanceFunc(dbBatch, query, cudaDevice, statisticsInfo);
    // torch::matmul(dbBatch, transposed_query);
    //std::cout<<"distance :\n"<<distances.t()<<std::endl;
    auto tStartTopK = std::chrono::high_resolution_clock::now();
    // Use torch::topk to get the top_k smallest distances and their indices
    auto topk_result = torch::topk(distances, top_k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);

    // std::cout<<"top k :\n"<<std::get<0>(topk_result)<<std::endl;
    // Extract top_k distances and indices
    torch::Tensor topk_distances = std::get<0>(topk_result);
    torch::Tensor topk_indices = std::get<1>(topk_result) + startPos;
    if (cudaDevice > -1 && torch::cuda::is_available()) {
      statisticsInfo->gpuComputingUs += chronoElapsedTime(tStartTopK);
    }
    else {
      statisticsInfo->cpuComputingUs += chronoElapsedTime(tStartTopK);
    }
    //
    if (cudaDevice > -1 && torch::cuda::is_available()) {
      auto tStart = std::chrono::high_resolution_clock::now();
      topk_distances = topk_distances.to(torch::kCPU);
      topk_indices = topk_indices.to(torch::kCPU);
      statisticsInfo->gpuCommunicationUs += chronoElapsedTime(tStart);
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

torch::Tensor CANDY::Gravistar::distanceIP(torch::Tensor db,
                                              torch::Tensor query,
                                              int64_t cudaDev,
                                              GravistarStatitics *idx) {
  assert(idx);
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
  if (cudaDev > -1 && torch::cuda::is_available()) {
    idx->gpuComputingUs += compTime;
  } else{
    idx->cpuComputingUs += compTime;
  }
  /*if(cudaDev>-1&&torch::cuda::is_available()){
   ru = ru.to(torch::kCPU);
  }*/
  return ru;
}

torch::Tensor CANDY::Gravistar::distanceL2(torch::Tensor db0,
                                              torch::Tensor _q,
                                              int64_t cudaDev,
                                           GravistarStatitics *idx) {
  assert(idx);
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
  if (cudaDev > -1 && torch::cuda::is_available()) {
    idx->gpuComputingUs += compTime;
  } else{
    idx->cpuComputingUs += compTime;
  }
  //idx->gpuComputingUs += compTime;
  /*if(cudaDev>-1&&torch::cuda::is_available()){
    // Move tensors to GPU 1
    result = result.to(torch::kCPU);
  }*/
  return result;
}

}