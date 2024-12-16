//
// Created by tony on 05/01/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <CANDY.h>
#include <iostream>
#include <CANDY/ThresholdIndex.h>
#include <CANDY/HNSWNaive/HNSW.h>
#include <CANDY/HNSWNaiveIndex.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;
TEST_CASE("Test threshold index", "[short]") {
    torch::manual_seed(114514);
    INTELLI::ConfigMapPtr cfg = newConfigMap();
    CANDY::ThresholdIndex thresholdIndex;

    cfg->edit("metricType", "L2");
    cfg->edit("dataThreshold", (int64_t) 60);

    thresholdIndex.setConfig(cfg);

    auto ta = torch::rand({1, 3});
    thresholdIndex.insertTensor_th(ta, "HNSWNaive");
    auto tb = torch::rand({1, 3});
    thresholdIndex.insertTensor_th(tb, "HNSWNaive");
    auto tc = torch::rand({1, 3});
    thresholdIndex.insertTensor_th(tc, "HNSWNaive");

    for(int i = 0; i < 100; i++) {
        auto x_in = torch::rand({1, 3});
        thresholdIndex.insertTensor_th(x_in, "HNSWNaive");
    }

    thresholdIndex.insertTensor_th(ta, "HNSWNaive");
    thresholdIndex.insertTensor_th(tb, "HNSWNaive");
    thresholdIndex.insertTensor_th(tb, "HNSWNaive");

    std::cout << "Insertion finished" << std::endl;

    // Search for 'ta'
    std::cout << "1. Now, do the query\n" << ta << std::endl;
    auto ruTensors = thresholdIndex.searchTensor_th(ta, 2);
    std::cout << "Get tensor\n";

    for (const auto& tensor : ruTensors) {
        std::cout << tensor << std::endl;
    }
    std::cout << "Total results: " << ruTensors.size() << std::endl;
}

    //size_t k = 1;
    /*
    for(int i=0; i<20; i++)
    {
      auto x_in = torch::rand({1, 3});
      hnswIdx.insertTensor(x_in);
    }
    cout << "insertion finish" << endl;
    auto q = torch::rand({1, 3});
    std::cout << "1. now, do the query\n" << q << std::endl;
    auto ruTensors = hnswIdx.searchTensor(q, 3);
    std::cout << "get tensor\n" << ruTensors[0] << std::endl;
    auto y_in = torch::rand({110, 3});
    thresholdIndex.insertTensor_th(y_in, "HNSWNaive");
    auto ru = thresholdIndex.searchTensor_th(x_in, k);
    for (int64_t i = 0; i < x_in.size(0); i++) {

    auto new_in = newTensor(x_in.slice(0, i, i + 1));
    cout << "looking for" << *new_in << endl;
    cout << endl << ru[i] << endl << endl;
  }*/
    /*
    auto ta = torch::rand({1, 3});
    thresholdIndex.insertTensor_th(ta, "HNSWNaive");
    auto tb = torch::rand({1, 3});
    thresholdIndex.insertTensor_th(tb, "HNSWNaive");
    auto tc = torch::rand({1, 3});
    thresholdIndex.insertTensor_th(ta, "HNSWNaive");
    thresholdIndex.insertTensor_th(tb, "HNSWNaive");
    thresholdIndex.insertTensor_th(tc, "HNSWNaive");
    //std::cout << "0.now, the data base is\n" << thresholdIndex.rawData() << std::endl;
    */
    
  
  /*
    //offline
    auto x_offline = torch::rand({150, 3});
    REQUIRE(thresholdIndex.offlineBuild(x_offline) == true);
    cout << "Offline build complete" << endl;

    //online --> need change not correct
    auto x_online = torch::rand({10, 3});
    REQUIRE(thresholdIndex.insertTensor(x_online) == true);
    cout << "Online insertion complete" << endl;
    
    //search
    size_t k = 3;
    auto tb = torch::rand({1, 4});
    auto ruTensors = thresholdIndex.searchTensor(tb, 2);
    std::cout << "get tensor\n" << ruTensors[0] << std::endl;
    */
   
    //auto db = torch::rand({6, 4});
    //thresholdIndex.insertTensor_th(db, "HNSWNaive");
    //std::cout << "data base is\n" << db << std::endl;
    //auto query = db.slice(0, 2, 3);
    //std::cout << "query is\n" << query << std::endl;
    //auto thRu = thresholdIndex.searchTensor(query, 2);
    //std::cout << "get tensor\n" << thRu[0] << std::endl;

  /*CANDY::IndexTable it;
  auto flatIdx = it.getIndex("flat");
  cfg->edit("vecDim", (int64_t) 4);
  flatIdx->setConfig(cfg);
  auto ta = torch::rand({1, 4});
  flatIdx->insertTensor(ta);
  auto tb = torch::rand({1, 4});
  flatIdx->insertTensor(tb);
  auto tc = torch::rand({1, 4});
  flatIdx->insertTensor(tc);
  flatIdx->insertTensor(tb);
  std::cout << "0.now, the data base is\n" << flatIdx->rawData() << std::endl;
  auto ruIdx = flatIdx->searchIndex(tb, 2);
  auto ruTensors = flatIdx->searchTensor(tb, 2);
  std::cout << "1. now, do the query\n" << flatIdx->rawData() << std::endl;
  for (uint64_t i = 0; i < ruIdx.size(); i++) {
    std::cout << "result [" + to_string(i) + "]:idx=" << ruIdx[i] << std::endl;
  }
  std::cout << "get tensor\n" << ruTensors[0] << std::endl;
  std::cout << "3. now, delete 2 similar\n" << std::endl;
  flatIdx->deleteTensor(tb, 2);
  std::cout << "the data base is\n" << flatIdx->rawData() << std::endl;
  std::cout << "4. now, do the edit\n" << std::endl;
  flatIdx->reviseTensor(tc, tb);
  std::cout << "the data base is\n" << flatIdx->rawData() << std::endl;
  REQUIRE(a == 0);
  auto ru = thresholdIndex.searchTensor(x_online, k);
    for (int64_t i = 0; i < x_online.size(0); i++) {
        auto new_in = x_online.slice(0, i, i + 1);
        cout << "Query tensor:" << new_in << endl;
        cout << "Search result:" << ru[i] << endl << endl;*/

