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
    cfg->edit("M", (int64_t) 4);
    CANDY::VisitedTable vt;
    hnswIdx.setConfig(cfg);
    /*
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
    }*/

    auto ta = torch::rand({1, 3});
    hnswIdx.insertTensor(ta);
    auto tb = torch::rand({1, 3});
    hnswIdx.insertTensor(tb);
    auto tc = torch::rand({1, 3});
    hnswIdx.insertTensor(tc);

    for(int i = 0; i < 50; i++) {
      auto x_in = torch::rand({1, 3});
      hnswIdx.insertTensor(x_in);
    }

   // hnswIdx.insertTensor(ta);
    //hnswIdx.insertTensor(tb);
    //hnswIdx.insertTensor(tc);

    std::cout << "Insertion finished" << std::endl;

    // Search for 'ta'
    std::cout << "1. Now, do the query\n" << ta << std::endl;
    auto ruTensors = hnswIdx.searchTensor(ta, 5);
    std::cout << "Get tensor\n";

    for (const auto& tensor : ruTensors) {
      std::cout << tensor << std::endl;
    }
    std::cout << "Total results: " << ruTensors.size() << std::endl;


}
