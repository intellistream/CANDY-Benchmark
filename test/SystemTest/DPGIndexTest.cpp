//
// Created by honeta on 04/04/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include <CANDY.h>

#include <iostream>

#include "catch.hpp"
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;
TEST_CASE("Test dpg index", "[short]") {
  int a = 0;
  torch::manual_seed(114514);
  // place your test here
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::IndexTable it;
  auto flatIdx = it.getIndex("flat");
  cfg->edit("vecDim", (int64_t)2);
  flatIdx->setConfig(cfg);
  auto ta = torch::rand({8, 2});
  flatIdx->insertTensor(ta);
  std::cout << "0.now, the data base is\n" << flatIdx->rawData() << std::endl;
  auto as0 = ta.slice(0, 1, 2);
  auto ruTensors = flatIdx->searchTensor(as0, 2);
  std::cout << "query tensor from flat\n" << ruTensors[0] << std::endl;

  auto dpgIndex = it.getIndex("DPG");
  cfg->edit("graphK", int64_t(2));
  cfg->edit("parallelWorkers", int64_t(12));
  cfg->edit("delta", 0.01);
  cfg->edit("rho", 1.0);
  dpgIndex->setConfig(cfg);
  dpgIndex->setFrozenLevel(1);

  dpgIndex->offlineBuild(ta);
  auto ruTensors2 = dpgIndex->searchTensor(as0, 2);
  std::cout << "query tensor from DPG (offline build)" << std::endl;
  for (auto tensor : ruTensors2) {
    std::cout << tensor << std::endl;
  }

  dpgIndex->reset();
  dpgIndex->insertTensor(ta);
  ruTensors2 = dpgIndex->searchTensor(as0, 2);
  std::cout << "query tensor from DPG (online update)" << std::endl;
  for (auto tensor : ruTensors2) {
    std::cout << tensor << std::endl;
  }
  REQUIRE(a == 0);
}
