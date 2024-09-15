//
// Created by Isshin on 2024/1/18.
//
#define CATCH_CONFIG_MAIN
#include <CANDY/HNSWNaive/HNSW.h>
#include <CANDY/HNSWNaiveIndex.h>
#include "catch.hpp"
#include <CANDY.h>

using namespace std;
TEST_CASE("Test HNSWiNDEX", "[short]") {
  CANDY::HNSWNaiveIndex hnswIdx;
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  cfg->edit("vecDim", (int64_t) 3);
  cfg->edit("maxConnection", (int64_t) 4);
  CANDY::VisitedTable vt;
  hnswIdx.setConfig(cfg);
  auto x_in = torch::rand({150, 3});
  CANDY::DistanceQueryer qdis(3);
  hnswIdx.insertTensor(x_in);
  cout << "insertion finish" << endl;
  size_t k = 1;
  auto ru = hnswIdx.searchTensor(x_in, k);
  for (int64_t i = 0; i < x_in.size(0); i++) {

    auto new_in = newTensor(x_in.slice(0, i, i + 1));
    cout << "looking for" << *new_in << endl;
    cout << endl << ru[i] << endl << endl;
  }

}

TEST_CASE("Test HNSWIndex", "") {
  CANDY::HNSWNaiveIndex hnswIdx;
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  cfg->edit("vecDim", (int64_t) 10);
  cfg->edit("maxConnection", (int64_t) 32);

  hnswIdx.setConfig(cfg);

  auto x_in_1 = torch::rand({15000, 10});
  auto x_in_2 = torch::rand({15000, 10});
  auto x_in_3 = torch::rand({15000, 10});
  auto x_in_4 = torch::rand({15000, 10});

  hnswIdx.insertTensor(x_in_1);
  hnswIdx.insertTensor(x_in_2);
  hnswIdx.insertTensor(x_in_3);
  hnswIdx.insertTensor(x_in_4);
  cout << "insertion finish" << endl;

  auto query_1 = x_in_1.slice(0, 100, 110);

  int k_search = 1;
  auto ru_1 = hnswIdx.searchTensor(query_1, k_search);

  for (int64_t i = 0; i < query_1.size(0); i++) {
    auto new_in = newTensor(query_1.slice(0, i, i + 1));
    cout << "looking for" << *new_in << endl;
    cout << endl << ru_1[i] << endl << endl;
  }

  cout << "search finish" << endl;

  if (hnswIdx.deleteTensor(query_1)){
    cout << "delete finish" << endl;
  } else {
    cout << "delete failed" << endl;
  };

  cout << "\033[1;31m" << "==============================================================" << "\033[0m" << endl << endl;

  auto ru_2 = hnswIdx.searchTensor(query_1, k_search);
  for (int64_t i = 0; i < query_1.size(0); i++) {
    auto new_in = newTensor(query_1.slice(0, i, i + 1));
    cout << "looking for" << *new_in << endl;
    cout << endl << ru_2[i] << endl << endl;
  }
}
