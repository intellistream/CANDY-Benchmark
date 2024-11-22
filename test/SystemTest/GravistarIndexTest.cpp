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
TEST_CASE("Test gravistar index", "[short]")
{
  //int a = 0;
  torch::manual_seed(114514);
  // place your test here
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::IndexTable it;
  auto annsIdx = it.getIndex("graviStar");
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
  /*auto tb = torch::rand({1, 4});
  flatIdx->insertTensor(tb);
  auto tc = torch::rand({2, 4});
  flatIdx->insertTensor(tc);*/
  //flatIdx->insertTensor(tb);
 // std::cout << "0.now, the data base is\n" << flatIdx->rawData() << std::endl;

}
