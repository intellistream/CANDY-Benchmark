//
// Created by tony on 24-11-25.
//
#define CATCH_CONFIG_MAIN
#include <CANDY/HNSWNaive/HNSWTensorSim.h>

#include "catch.hpp"
#include <CANDY.h>

using namespace std;
TEST_CASE("Test HNSWTensor Data structure", "[short]") {
  CANDY::HNSWTensorSim hnsw;
  hnsw.init(10, 5, 10, 0.5);

  torch::Tensor vec1 = torch::tensor({1.0, 2.0}, torch::kFloat32);
  torch::Tensor vec2 = torch::tensor({2.0, 3.0}, torch::kFloat32);
  torch::Tensor vec3 = torch::tensor({4.0, 5.0}, torch::kFloat32);

  hnsw.add(vec1);
  hnsw.add(vec2);
  hnsw.add(vec3);

  torch::Tensor queries = torch::tensor({{1.5, 2.5}, {4.0, 5.0}}, torch::kFloat32);
  auto results = hnsw.multiQuerySearch(queries, 2);

  for (size_t i = 0; i < results.size(); ++i) {
    std::cout << "Query " << i << " neighbors:\n" << results[i] << std::endl;
  }

}
TEST_CASE("Test HNSWTensorIndex", "[short]") {
  //int a = 0;
  torch::manual_seed(114514);
  // place your test here
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::IndexTable it;
  auto annsIdx = it.getIndex("HNSWTensor");
  auto flatIdx = it.getIndex("flat");
  cfg->edit("vecDim", (int64_t) 4);
  cfg->edit("metricType", "IP");
  cfg->edit("memBufferSize", (int64_t)4);
  annsIdx->setConfig(cfg);
  flatIdx->setConfig(cfg);
  auto ta = torch::rand({10, 4});
  annsIdx->insertTensor(ta);
  flatIdx->insertTensor(ta);
  auto tc =ta.slice(0,5,6);
  std::cout << "data\n" <<ta << std::endl;
  std::cout << "query\n" <<tc << std::endl;
  auto ruTensorsFlat = flatIdx->searchTensor(tc, 2);
  auto ruTensorsAnns = annsIdx->searchTensor(tc, 2);
  std::cout << "result tensor from flat\n" << ruTensorsFlat[0] << std::endl;
  std::cout << "result tensor from anns\n" << ruTensorsAnns[0] << std::endl;

}