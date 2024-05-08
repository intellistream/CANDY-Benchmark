//
// Created by Isshin on 2024/3/24.
//
#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <CANDY/FlannIndex/KdTree.h>
#include <CANDY/FlannIndex/Kmeans.h>
using namespace std;
TEST_CASE("Test KDTreebuild", "[short]") {
  auto kd = new CANDY::KmeansTree();
  kd->vecDim = 32;
  kd->ntotal = 0;
  kd->lastNNZ = -1;
  kd->expandStep = 100;
  kd->branching = 5;
  kd->iterations = 2;
  kd->dbTensor = torch::zeros({0, kd->vecDim});
  kd->centerChooser = new CANDY::FLANN::RandomCenterChooser(&kd->dbTensor, kd->vecDim);

  auto data_1 = torch::rand({10000, kd->vecDim});
  printf("starting  add points\n");
  kd->addPoints(data_1);
  printf("starting  add points\n");
  auto data_2 = torch::rand({1000, kd->vecDim});
  kd->addPoints(data_2);

}