//
// Created by tony on 05/01/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <CANDY.h>
#include <iostream>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;
TEST_CASE("Test flat index", "[short]")
{
  int a = 0;
  torch::manual_seed(114514);
  // place your test here
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::IndexTable it;
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
}
