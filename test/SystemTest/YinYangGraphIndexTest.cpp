//
// Created by tony on 02/02/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <iostream>
#include <CANDY.h>
#include <faiss/IndexFlat.h>
#include <CANDY/YinYangGraphIndex/YinYangGraph.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;
TEST_CASE("Test  yinyang graph index insert", "[short]")
{
  int a = 0;
  torch::manual_seed(114514);
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::IndexTable it;
  auto yinYangIdx = it.getIndex("yinYang");
  cfg->edit("vecDim", (int64_t) 4);
  cfg->edit("encodeLen", (int64_t) 4);
  cfg->edit("candidateTimes", (int64_t) 1);
// cfg->edit("numberOfBuckets", (int64_t) 2);
  yinYangIdx->setConfig(cfg);
  auto db = torch::rand({6, 4});
  yinYangIdx->insertTensor(db);
  std::cout << "data base is\n" << db << std::endl;
  auto query = db.slice(0, 2, 3);
  auto flatIndex = it.getIndex("flat");
  flatIndex->setConfig(cfg);
  flatIndex->insertTensor(db);
  auto flatRu = flatIndex->searchTensor(query, 2);
  std::cout << "flat result is\n" << flatRu[0] << std::endl;
  auto pqRu = yinYangIdx->searchTensor(query, 2);
  std::cout << "YinYang result is\n" << pqRu[0] << std::endl;
  REQUIRE(a == 0);
}