//
// Created by Isshin on 2024/3/25.
//
#include<Algorithms/FlannIndex.h>
#include<faiss/utils/distances.h>
#include <Utils/UtilityFunctions.h>

bool CANDY::FlannIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  vecDim = cfg->tryI64("vecDim", 768, true);
  flann_index = cfg->tryI64("flannIndexTag", 2, true);
  allAuto = cfg->tryI64("allAuto", 0, true);
  if (flann_index == FLANN_AUTO) {
    INTELLI_INFO("Auto-tuning!");
    CANDY::DataLoaderTable dataLoaderTable;

    // pre-load the data...perhaps could be better
    std::string dataLoaderTag = cfg->tryString("dataLoaderTag", "random", true);
    auto dataLoader = dataLoaderTable.findDataLoader(dataLoaderTag);
    if (dataLoader == nullptr) {
      return -1;
    }
    dataLoader->setConfig(cfg);
    int64_t trainRows = cfg->tryI64("initialRows", 0, true) * 0.1;
    int64_t testRows = trainRows * 0.1 + 1;
    auto dataTensorAll = dataLoader->getData().nan_to_num(0);
    auto trainSet = dataTensorAll.slice(0, 0, trainRows);
    auto testSet = dataTensorAll.slice(0, 0, testRows);

    std::vector<CANDY::FlannParam> params;

    // first kd-tree
    int64_t nums_tree[] = {4, 8, 16, 32, 64};
    for (size_t i = 0; i < 5; i++) {
      printf("Now for KdTree with %ld roots\n", nums_tree[i]);
      CANDY::FlannParam param;
      param.flann_index = FLANN_KDTREE;
      param.num_trees = nums_tree[i];

      CANDY::KdTree *sample = new KdTree();
      sample->setConfig(cfg);
      sample->num_trees = nums_tree[i];

      auto start = std::chrono::high_resolution_clock::now();
      sample->addPoints(trainSet);
      auto tBuild = chronoElapsedTime(start);
      param.buildTime = tBuild;

      start = std::chrono::high_resolution_clock::now();
      std::vector<faiss::idx_t> ru(5 * testRows);
      std::vector<float> distance(5 * testRows);
      sample->knnSearch(testSet, ru.data(), distance.data(), 5);
      auto tSearch = chronoElapsedTime(start);
      param.searchTime = tSearch;

      params.push_back(param);
    }

    // then kmeans
    int64_t maxIterations[] = {1, 2, 4, 8};
    int64_t branches[] = {16, 32, 64, 128};
    double cb_indexes[] = {0.2, 0.4, 0.6, 0.8};
    for (size_t i = 0; i < 4; i++) {
      CANDY::FlannParam param;
      param.flann_index = FLANN_KMEANS;
      param.maxIterations = maxIterations[i];

      for (size_t j = 0; j < 4; j++) {
        param.branching = branches[j];

        CANDY::KmeansTree *sample = new CANDY::KmeansTree();
        sample->setConfig(cfg);
        sample->branching = branches[j];
        sample->iterations = maxIterations[i];

        auto start = std::chrono::high_resolution_clock::now();
        sample->addPoints(trainSet);
        auto tBuild = chronoElapsedTime(start);
        param.buildTime = tBuild;

        start = std::chrono::high_resolution_clock::now();
        std::vector<faiss::idx_t> ru(5 * testRows);
        std::vector<float> distance(5 * testRows);

        for (size_t k = 0; k < 4; k++) {
          printf("Now for KMeansTree with %ld iterations, %ld branches, %.2lf cb\n",
                 nums_tree[i],
                 branches[j],
                 cb_indexes[k]);
          param.cb_index = cb_indexes[k];
          sample->knnSearch(testSet, ru.data(), distance.data(), 5);
          auto tSearch = chronoElapsedTime(start);
          param.searchTime = tSearch;

          params.push_back(param);
        }
      }
    }

    size_t best = 0;
    CANDY::FlannParam best_param = params[0];
    auto best_score = 0.01 * best_param.buildTime + best_param.searchTime;
    printf("%lf\n ", best_score);
    for (size_t i = 1; i < params.size(); i++) {
      auto score = 0.01 * params[i].buildTime + params[i].searchTime;
      printf("%lf\n", score);
      if (score < best_score) {
        best = i;
        best_param = params[i];
        best_score = score;
      }
    }

    if (best_param.flann_index == FLANN_KDTREE) {
      INTELLI_INFO("AUTOTUNED TO BE INIT AS FLANN KDTREE!");
      index = new CANDY::KdTree();
      index->setConfig(cfg);
      index->setParams(best_param);
    } else if (best_param.flann_index == FLANN_KMEANS) {
      INTELLI_INFO("AUTOTUNED TO BE INIT AS FLANN KMEANS TREE!");
      index = new CANDY::KmeansTree();
      index->setConfig(cfg);
      index->setParams(best_param);
    }

  } else if (flann_index == FLANN_KDTREE) {
    INTELLI_INFO("INIT AS FLANN KDTREE!");
    index = new CANDY::KdTree();
    index->setConfig(cfg);
  } else if (flann_index == FLANN_KMEANS) {
    CANDY::FlannParam best_param;
    if (allAuto) {
      CANDY::DataLoaderTable dataLoaderTable;
      // pre-load the data...perhaps could be better
      std::string dataLoaderTag = cfg->tryString("dataLoaderTag", "random", true);
      auto dataLoader = dataLoaderTable.findDataLoader(dataLoaderTag);
      if (dataLoader == nullptr) {
        return -1;
      }
      dataLoader->setConfig(cfg);
      int64_t trainRows = cfg->tryI64("initialRows", 0, true) * 0.1;
      int64_t testRows = trainRows * 0.1 + 1;
      auto dataTensorAll = dataLoader->getData().nan_to_num(0);
      auto trainSet = dataTensorAll.slice(0, 0, trainRows);
      auto testSet = dataTensorAll.slice(0, 0, testRows);

      std::vector<CANDY::FlannParam> params;

      int64_t maxIterations[] = {1, 2, 4, 8};
      int64_t branches[] = {16, 32, 64, 128};
      double cb_indexes[] = {0.2, 0.4, 0.6, 0.8};
      for (size_t i = 0; i < 4; i++) {
        CANDY::FlannParam param;
        param.flann_index = FLANN_KMEANS;
        param.maxIterations = maxIterations[i];

        for (size_t j = 0; j < 4; j++) {
          param.branching = branches[j];

          CANDY::KmeansTree *sample = new CANDY::KmeansTree();
          sample->setConfig(cfg);
          sample->branching = branches[j];
          sample->iterations = maxIterations[i];

          auto start = std::chrono::high_resolution_clock::now();
          sample->addPoints(trainSet);
          auto tBuild = chronoElapsedTime(start);
          param.buildTime = tBuild;

          start = std::chrono::high_resolution_clock::now();
          std::vector<faiss::idx_t> ru(5 * testRows);
          std::vector<float> distance(5 * testRows);

          for (size_t k = 0; k < 4; k++) {
            printf("Now for KMeansTree with %ld iterations, %ld branches, %.2lf cb\n", maxIterations[i],
                   branches[j], cb_indexes[k]);
            param.cb_index = cb_indexes[k];
            sample->knnSearch(testSet, ru.data(), distance.data(), 5);
            auto tSearch = chronoElapsedTime(start);
            param.searchTime = tSearch;

            params.push_back(param);
          }
        }
      }

      size_t best = 0;
      best_param = params[0];
      auto best_score = 0.01 * best_param.buildTime + best_param.searchTime;
      for (size_t i = 1; i < params.size(); i++) {
        auto score = 0.01 * params[i].buildTime + params[i].searchTime;
        if (score < best_score) {
          best = i;
          best_param = params[i];
          best_score = score;
        }
      }
    }
    INTELLI_INFO("INIT AS FLANN KMEANS TREE!");
    index = new CANDY::KmeansTree();
    index->setConfig(cfg);
    if (allAuto) {
      index->setParams(best_param);
    }
  }
  return true;
}

bool CANDY::FlannIndex::loadInitialTensor(torch::Tensor &t) {
  index->addPoints(t);
  return true;
}

bool CANDY::FlannIndex::insertTensor(torch::Tensor &t) {
  index->addPoints(t);
  return true;
}

std::vector<faiss::idx_t> CANDY::FlannIndex::searchIndex(torch::Tensor q, int64_t k) {
  auto querySize = q.size(0);
  std::vector<faiss::idx_t> ru(k * querySize);
  std::vector<float> distance(k * querySize);
  index->knnSearch(q, ru.data(), distance.data(), k);
  return ru;
}

std::vector<torch::Tensor> CANDY::FlannIndex::searchTensor(torch::Tensor &q, int64_t k) {
  auto idx = searchIndex(q, k);
  int64_t size = idx.size() / k;
//    for(int64_t i=0; i<size; i++){
//        auto query_data = q.slice(0,i,i+1).contiguous().data_ptr<float>();
//        printf("obtained result:\n");
//        for (int64_t j = 0; j < k; j++) {
//            int64_t tempIdx = idx[i * k + j];
//            auto data = index->dbTensor.slice(0, tempIdx, tempIdx + 1).contiguous().data_ptr<float>();
//            auto dist = faiss::fvec_L2sqr(query_data, data, vecDim);
//            printf("%ld %f\n", tempIdx, dist);
//        }
//    }
  return getTensorByIndex(idx, k);
}

std::vector<torch::Tensor> CANDY::FlannIndex::getTensorByIndex(std::vector<faiss::idx_t> &idx, int64_t k) {
  int64_t size = idx.size() / k;
  std::vector<torch::Tensor> ru(size);
  for (int64_t i = 0; i < size; i++) {
    ru[i] = torch::zeros({k, vecDim});
    for (int64_t j = 0; j < k; j++) {
      int64_t tempIdx = idx[i * k + j];

      if (tempIdx >= 0) {
        ru[i].slice(0, j, j + 1) = index->dbTensor.slice(0, tempIdx, tempIdx + 1);

      };
    }
  }
  return ru;
}

