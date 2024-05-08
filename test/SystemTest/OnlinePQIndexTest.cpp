//
// Created by tony on 05/01/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <CANDY.h>
#include <iostream>
#include <CANDY/OnlinePQIndex/SimpleStreamClustering.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;
/*TEST_CASE("Test  clustering idx", "[short]")
{
int a = 0;
torch::manual_seed(114514);
auto db = torch::rand({6, 4});
CANDY::SimpleStreamClusteringPtr sp=newSimpleStreamClustering();
sp->buildCentroids(db,
    2,
    1000,
    SimpleStreamClustering::euclideanDistance);
auto labels=sp->classifyMultiRow(db);
auto centroids=sp->exportCentroids();
for(int64_t i=0;i<6;i++)
{ std::cout<<i<<"at cluster"<<labels[i]<<std::endl;
  std::cout<<db.slice(0,i,i+1)-centroids.slice(0,labels[i],labels[i]+1)<<std::endl;
}
REQUIRE(a == 0);
}*/

TEST_CASE("Test  online pq index build", "[short]")
{
  int a = 0;
  torch::manual_seed(114514);
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::IndexTable it;
  auto onlinePQIdx = it.getIndex("onlinePQ");
  cfg->edit("vecDim", (int64_t) 4);
  cfg->edit("coarseGrainedClusters", (int64_t) 2);
  cfg->edit("fineGrainedClusters", (int64_t) 2);
  cfg->edit("maxBuildIteration", (int64_t) 100);
  cfg->edit("subQuantizers", (int64_t) 2);
  onlinePQIdx->setConfig(cfg);
  auto db = torch::rand({6, 4});
  onlinePQIdx->offlineBuild(db);
  REQUIRE(a == 0);
}

TEST_CASE("Test  online pq index insert", "[short]")
{
  int a = 0;
  torch::manual_seed(114514);
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::IndexTable it;
  auto onlinePQIdx = it.getIndex("onlinePQ");
  cfg->edit("vecDim", (int64_t) 4);
  cfg->edit("coarseGrainedClusters", (int64_t) 2);
  cfg->edit("fineGrainedClusters", (int64_t) 2);
  cfg->edit("maxBuildIteration", (int64_t) 100);
  cfg->edit("subQuantizers", (int64_t) 2);
  onlinePQIdx->setConfig(cfg);
  auto db = torch::rand({6, 4});
  onlinePQIdx->insertTensor(db);
  std::cout << "data base is\n" << db << std::endl;
  auto query = db.slice(0, 2, 3);
  auto flatIndex = it.getIndex("flat");
  flatIndex->setConfig(cfg);
  flatIndex->insertTensor(db);
  auto flatRu = flatIndex->searchTensor(query, 2);
  std::cout << "flat result is\n" << flatRu[0] << std::endl;
  auto pqRu = onlinePQIdx->searchTensor(query, 2);
  std::cout << "pq result is\n" << pqRu[0] << std::endl;
  REQUIRE(a == 0);
}