//
// Created by tony on 05/01/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <CANDY.h>
#include <iostream>
#include <CANDY/YingYangHNSWIndex.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;
TEST_CASE("Test YinYangHNSW index", "[short]")
{
  //int a = 0;
  torch::manual_seed(114514);
  // place your test here
  INTELLI::ConfigMapPtr cfg = newConfigMap();

  auto yyHNSWIdx = newYinYangHNSWIndex();
  cfg->edit("vecDim", (int64_t) 4);
  cfg->edit("metricType", "IP");
  yyHNSWIdx->setConfig(cfg);
  auto ta = torch::rand({2, 4});
  std::cout << "0. Load initial"<<std::endl;
  yyHNSWIdx->loadInitialTensor(ta);
  auto tb = torch::rand({1, 4});
  std::cout << "1. insert"<<std::endl;
  yyHNSWIdx->insertTensor(tb);
  auto tc = torch::rand({2, 4});
  cout <<tc<< endl;
  yyHNSWIdx->insertTensor(tc);
  std::cout << "2. search"<<std::endl;
  cout << yyHNSWIdx->searchTensor(tc, 1)[0] << endl;
  //yyHNSWIdx->insertTensor(tb);
  //std::cout << "0.now, the data base is\n" << yyHNSWIdx->rawData() << std::endl;

}
