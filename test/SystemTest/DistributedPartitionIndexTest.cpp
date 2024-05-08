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
TEST_CASE("Test distributed partition index", "[short]")
{
  int a = 0;
  torch::manual_seed(114514);
  // place your test here
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::IndexTable it;
  auto flatIdx = it.getIndex("flat");
  cfg->edit("vecDim", (int64_t) 4);
  flatIdx->setConfig(cfg);
  auto ta = torch::rand({6, 4});
  flatIdx->insertTensor(ta);
  std::cout << "0.now, the data base is\n" << flatIdx->rawData() << std::endl;
  auto as0 = ta.slice(0, 1, 2);
  auto ruTensors = flatIdx->searchTensor(as0, 2);
  std::cout << "query tensor from flat\n" << ruTensors[0] << std::endl;

  auto ppIndex = it.getIndex("distributedPartition");
  cfg->edit("distributedWorkers", (int64_t) 3);
  ppIndex->setConfig(cfg);
  ppIndex->startHPC();
  for (int64_t i = 0; i < 6; i++) {
    auto asi = ta.slice(0, i, i + 1);
    ppIndex->insertTensor(asi);
  }
  auto as2 = ta.slice(0, 1, 2);
  auto ruTensors2 = ppIndex->searchTensor(as2, 2);
  std::cout << "query tensor from pp\n" << ruTensors2[0] << std::endl;
  ppIndex->deleteTensor(as2, 2);
  ppIndex->endHPC();
  REQUIRE(a == 0);
}
