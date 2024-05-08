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
TEST_CASE("Test flatAMMIPObj index", "[short]")
{
  //int a = 0;
  torch::manual_seed(114514);
  // place your test here
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::IndexTable it;
  auto flatIdx = it.getIndex("flatAMMIPObj");
  cfg->edit("vecDim", (int64_t) 4);
  cfg->edit("metricType", "IP");
  flatIdx->setConfig(cfg);
  std::vector<std::string> strs(4);
  strs[0] = "tensor 0";
  strs[1] = "tensor 1";
  strs[2] = "tensor 2";
  strs[3] = "tensor 3";
  auto tc = torch::rand({4, 4});
  flatIdx->insertStringObject(tc, strs);
  //flatIdx->insertTensor(tb);
  std::cout << "0.now, the data base is\n" << flatIdx->rawData() << std::endl;
  auto tb = tc.slice(0, 3, 4);
  auto ruIdx = flatIdx->searchIndex(tb, 2);
  std::cout << "the idx is:";
  for (size_t i = 0; i < ruIdx.size(); i++) {
    std::cout << ruIdx[i] << ",";
  }
  std::cout << endl;
  std::cout << "the object is:";
  auto ruObj = flatIdx->searchStringObject(tb, 2);
  for (size_t i = 0; i < ruObj.size(); i++) {
    for (size_t j = 0; j < ruObj[i].size(); j++) {
      std::cout << ruObj[i][j] << endl;
    }
  }

}
