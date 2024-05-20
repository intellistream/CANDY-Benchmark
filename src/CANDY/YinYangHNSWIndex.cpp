//
// Created by tony on 16/05/24.
//
#include <CANDY/YingYangHNSWIndex.h>
#if defined(__GNUC__) && (__GNUC__ >= 4)
#define ADB_memcpy(dst, src, size) __builtin_memcpy(dst, src, size)
#else
#define ADB_memcpy(dst, src, size) memcpy(dst, src, size)
#endif
bool CANDY::inlineYangIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  if (faissMetric != faiss::METRIC_INNER_PRODUCT) {
    INTELLI_WARNING("I can only deal with inner product distance");
  }
  vecDim = cfg->tryI64("vecDim", 768, true);
  initialVolume = cfg->tryI64("initialVolume", 100, true);
  expandStep = cfg->tryI64("expandStep", 100, true);
  sketchSize = cfg->tryI64("sketchSize", 10, true);
  DCOBatchSize = cfg->tryI64("DCOBatchSize", 128, true);

  std::string ammAlgo = cfg->tryString("ammAlgo", "mm", true);
  //INTELLI_INFO("Size of DCO=" + std::to_string(DCOBatchSize));
  if (ammAlgo == "crs") {
    ammType = 1;
    //INTELLI_INFO("Use crs for amm, sketch size=" + std::to_string(sketchSize));
  } else if (ammAlgo == "smp-pca") {
    ammType = 2;
    //INTELLI_INFO("Use smp-pca for amm, sketch size=" + std::to_string(sketchSize));
  } else {
    ammType = 0;
  }
  dbTensor = torch::zeros({initialVolume, vecDim});
  objTensor = torch::zeros({initialVolume, 1}, torch::kLong);

  lastNNZ = -1;
  lastNNZObj = -1;
  return true;
}
bool CANDY::YinYangHNSWIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  if (faissMetric != faiss::METRIC_INNER_PRODUCT) {
    INTELLI_WARNING("I can only deal with inner product distance");
  }
  vecDim = cfg->tryI64("vecDim", 768, true);
  maxHNSWVolume = cfg->tryI64("maxHNSWVolume", 1000000, true);
  maxConnection = cfg->tryI64("maxConnection", 32, true);
  efConstruction = cfg->tryI64("efConstruction", 200, true);
  initialVertex = cfg->tryI64("initialVertex", -1, true);
  inlineCfg = newConfigMap();
  inlineCfg->loadFrom(cfg.get()[0]);
  inlineCfg->edit("initialVolume",(int64_t)1);
  inlineCfg->edit("DCOBatchSize",(int64_t )128);
  hnswlib::InnerProductSpace space(vecDim);
  alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, maxHNSWVolume, maxConnection, efConstruction);
  return true;
}
bool CANDY::YinYangHNSWIndex::loadInitialTensorVertex(torch::Tensor &t) {
  /**
  * @brief just to make each row as vertex in HNSW
  */
  auto n = t.size(0);
  /*auto tdata=new float[vecDim*n];
  ADB_memcpy(tdata,t.contiguous().data_ptr<float>(),t.numel());*/
  for (int64_t i = 0; i < n; i++) {
    auto tCopy = t.slice(0, i, i + 1).contiguous();
    YinYangHNSW_YinVertex *ver=new YinYangHNSW_YinVertex();
    ver->verTensor=tCopy;
    ver->yangIndex.setConfig(inlineCfg);
    ver->yangIndex.insertTensor(tCopy);
    auto idx = reinterpret_cast<long>(ver);
    if(i%(n/100)==0){
      std::cout<<"Done "+ to_string(i)+"/"+ to_string(n)<<std::endl;
    }

    alg_hnsw->addPoint(tCopy.contiguous().data_ptr<float>(), idx);
    // std::cout<<idx<<std::endl;
  }
  return true;
}
bool CANDY::YinYangHNSWIndex::loadInitialTensor(torch::Tensor &t) {
  /**
   * @brief just to make each row as vertex in HNSW
   */
  auto n = t.size(0);
  if(initialVertex<0||(initialVertex>n)){
    return loadInitialTensorVertex(t);
  }
  int64_t num_rows = n;
  // Generate a random permutation of row indices
  torch::Tensor indices = torch::randperm(num_rows);
  // Index the tensor with the shuffled indices
  torch::Tensor shuffled_tensor = t.index_select(0, indices);
  auto verTensor = shuffled_tensor.slice(0,0,initialVertex);
  auto rangeTensor= shuffled_tensor.slice(0,initialVertex,n);
  INTELLI_INFO(to_string(initialVertex)+" rows in initial graph vertex");
  loadInitialTensorVertex(verTensor);
  insertTensor(rangeTensor);
  return true;
}
bool CANDY::YinYangHNSWIndex::insertTensor(torch::Tensor &t) {
  auto n = t.size(0);
  for (int64_t i = 0; i < n; i++) {
    auto tCopy = t.slice(0, i, i + 1);
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(tCopy.contiguous().data_ptr<float>(), 1);
    auto ver =  reinterpret_cast<YinYangHNSW_YinVertex *>((result.top().second));
    //std::cout <<result.top().second<<","<<ver->yangIndex.size()<<std::endl;
    ver->yangIndex.insertTensor(tCopy);
  }
  return true;
}
torch::Tensor CANDY::YinYangHNSWIndex::searchRow(torch::Tensor &q, int64_t k) {
  torch::Tensor ru=torch::zeros({k, vecDim});
  std::priority_queue<std::pair<float, hnswlib::labeltype>> hnswRu = alg_hnsw->searchKnn(q.contiguous().data_ptr<float>(), k);
  int64_t copiedResults=0;
  while ((!hnswRu.empty())&&(copiedResults<k)) {
    auto ver =  reinterpret_cast<YinYangHNSW_YinVertex *>((hnswRu.top().second));
    int64_t copyThisTime = ver->yangIndex.size();
    if(copyThisTime+copiedResults>k) {
      copyThisTime=k- copiedResults;
    }
    ru.slice(0,copiedResults,copiedResults+copyThisTime)=ver->yangIndex.searchTensor(q,copyThisTime)[0];
    copiedResults+=copyThisTime;
    hnswRu.pop();
  }
 return ru;
}
std::vector<torch::Tensor> CANDY::YinYangHNSWIndex::searchTensor(torch::Tensor &q, int64_t k) {
  size_t tensors = (size_t) q.size(0);
  std::vector<torch::Tensor> ru(tensors);
  for (int64_t i = 0; i < tensors; i++) {
    auto tCopy = q.slice(0, i, i + 1);
    ru[i] =searchRow(tCopy,k);
  }
  return ru;

}