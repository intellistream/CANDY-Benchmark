//
// Created by tony on 05/01/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <CANDY.h>
#include <iostream>
#include <CANDY/ThresholdIndex.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;
TEST_CASE("Test threshold index", "[short]")
{
  //int a = 0;
  torch::manual_seed(114514);
  // place your test here
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::ThresholdIndex thresholdIndex;
  //CANDY::IndexTable it;
  //auto thresholdIndex = it.getIndex("threshold");
    cfg->edit("metricType", "L2");
    cfg->edit("dataThreshold", (int64_t) 8);
    cfg->edit("indexAlgorithm", "HNSW");

    thresholdIndex.setConfig(cfg);
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
   
    auto db = torch::rand({6, 4});
    thresholdIndex.insertTensor(db);
    std::cout << "data base is\n" << db << std::endl;
    auto query = db.slice(0, 2, 3);
    std::cout << "query is\n" << query << std::endl;
    auto thRu = thresholdIndex.searchTensor(query, 2);
    std::cout << "get tensor\n" << thRu[0] << std::endl;

    auto ta = torch::rand({1, 4});
    thresholdIndex.insertTensor(ta);
    auto tb = torch::rand({1, 4});
    thresholdIndex.insertTensor(tb);
    auto tc = torch::rand({1, 4});
    thresholdIndex.insertTensor(tc);
    thresholdIndex.insertTensor(tb);
    thresholdIndex.insertTensor(ta);
    std::cout << "0.now, the data base is\n" << thresholdIndex.rawData() << std::endl;
    auto ruTensors = thresholdIndex.searchTensor(tb, 2);
    std::cout << "1. now, do the query\n" << thresholdIndex.rawData() << std::endl;
  
    std::cout << "get tensor\n" << ruTensors[0] << std::endl;
    
    
    
  } 

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

