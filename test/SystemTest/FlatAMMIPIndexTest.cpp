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
TEST_CASE("Test flatAMMIP index", "[short]")
{
  //int a = 0;
  torch::manual_seed(114514);
  // place your test here
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::IndexTable it;
  auto flatIdx = it.getIndex("flatAMMIP");
  cfg->edit("vecDim", (int64_t) 4);
  cfg->edit("metricType", "IP");
  flatIdx->setConfig(cfg);
  auto ta = torch::rand({1, 4});
  flatIdx->insertTensor(ta);
  auto tb = torch::rand({1, 4});
  flatIdx->insertTensor(tb);
  auto tc = torch::rand({2, 4});
  flatIdx->insertTensor(tc);
  //flatIdx->insertTensor(tb);
  std::cout << "0.now, the data base is\n" << flatIdx->rawData() << std::endl;

}
