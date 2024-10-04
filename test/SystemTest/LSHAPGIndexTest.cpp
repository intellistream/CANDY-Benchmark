//
// Created by tony on 05/01/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <CANDY.h>
#include <iostream>
#include <CANDY/LSHAPGIndex.h>
#include <CANDY/FlatAMMIPIndex.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;
TEST_CASE("Test LSHAPG index", "[short]")
{
  //int a = 0;
  torch::manual_seed(114514);
  // place your test here
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  LSHAPGIndex aknnIdx;
  FlatAMMIPIndex bfIdx;
  cfg->edit("vecDim", (int64_t) 4);
  cfg->edit("metricType", "IP");
  aknnIdx.setConfig(cfg);
  bfIdx.setConfig(cfg);
  auto ta = torch::rand({4, 4});
  aknnIdx.loadInitialTensor(ta);
  bfIdx.loadInitialTensor(ta);
  auto tb = torch::rand({1, 4});
  std::cout <<"To search:" <<tb<< std::endl;
  auto ru2=bfIdx.searchTensor(tb,1);
  auto ru=aknnIdx.searchTensor(tb,1);
  std::cout <<"result form aknn:" <<ru[0]<< std::endl;
  std::cout <<"result form brutal force:" <<ru2[0]<< std::endl;


  std::cout <<"To insert:" <<tb<< std::endl;
  bfIdx.insertTensor(tb);
  aknnIdx.insertTensor(tb);
  ru2=bfIdx.searchTensor(tb,1);
  ru=aknnIdx.searchTensor(tb,1);
  std::cout <<"result form aknn:" <<ru[0]<< std::endl;
  std::cout <<"result form brutal force:" <<ru2[0]<< std::endl;
  //lshAPGIdx->insertTensor(tb);
  //std::cout << "0.now, the data base is\n" << lshAPGIdx->rawData() << std::endl;
}

TEST_CASE("Test LSHAPG Index Delete", "[short]")
{
  torch::manual_seed(114514);

  CANDY::LSHAPGIndex lshAPGIdx;
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  cfg->edit("vecDim", (int64_t) 4);
  cfg->edit("metricType", "IP");
  lshAPGIdx.setConfig(cfg);

  auto init_tensor = torch::rand({4, 4});
  lshAPGIdx.loadInitialTensor(init_tensor);

  auto x_in = torch::rand({2, 4});
  lshAPGIdx.insertTensor(x_in);

  std::cout << "Finish loading initial tensor" << std::endl;

  auto query_x_1 = x_in.slice(0, 0, 1);
  auto query_x_2 = x_in.slice(0, 1, 2);

  auto result_1 = lshAPGIdx.searchTensor(query_x_1, 1);
  auto result_2 = lshAPGIdx.searchTensor(query_x_2, 1);

  for (size_t i = 0; i < result_1.size(); i++) {
    std::cout << "Query 1: " << query_x_1.slice(0, i, i + 1) << std::endl;
    std::cout << "Result 1: " << result_1[i] << std::endl;

    REQUIRE(result_1[i].equal(query_x_1.slice(0, i, i + 1)));
    REQUIRE(result_2[i].equal(query_x_2.slice(0, i, i + 1)));
  }

  std::cout << "Finish searching" << std::endl;

  lshAPGIdx.deleteTensor(query_x_1);

  std::cout << "Finish deleting" << std::endl;

  auto result_3 = lshAPGIdx.searchTensor(query_x_1, 1);
  auto result_4 = lshAPGIdx.searchTensor(query_x_2, 1);

  for (size_t i = 0; i < result_3.size(); i++) {
    REQUIRE(!result_3[i].equal(query_x_1.slice(0, i, i + 1)));
    REQUIRE(result_4[i].equal(query_x_2.slice(0, i, i + 1)));
  }
}