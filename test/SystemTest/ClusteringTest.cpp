//
// Created by Isshin on 2024/1/8.
//

#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <iostream>
#include <CANDY.h>
#include <Algorithms/Clustering.h>
#include <faiss/IndexFlat.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;

TEST_CASE("Test CLUSTERING train", "[short]")
{
  // place your test here
  torch::manual_seed(1919810);

  INTELLI::ConfigMapPtr cfg = newConfigMap();
  auto cst = new Clustering();

  size_t n = 1000;
  size_t k = 15;

  size_t d = 6;
  faiss::IndexFlatL2 *index = new faiss::IndexFlatL2(d);
  cfg->edit("vecDim", (int64_t) d);
  cfg->edit("k", (int64_t) k);
  cst->setConfig(cfg);

  auto x_in = 100 * torch::rand({n, d});
  auto weights = torch::rand({n});
  cst->train(n, x_in, index, weights);
  cout << "centroids: " << endl << cst->getCentroids() << endl;
}

//TEST_CASE("Test Clustering of PQ index", "[short]")
//{
//    int a = 0;
//    torch::manual_seed(1919810);
//    // place your test here
//    INTELLI::ConfigMapPtr cfg = newConfigMap();
//    auto cst = new Clustering();
//    cfg->edit("vecDim", (int64_t) 6);
//    cfg->edit("k", (int64_t)5);
//    size_t n = 1000;
//    size_t k = 5;
//    size_t k_frozen = 0;
//    size_t d=6;
//    cst->setConfig(cfg);
//    auto centroids_non_ptr = cst->getCentroids();
//    auto centroids = &centroids_non_ptr;
//    auto x_in = 100*torch::rand({10000,6});
//    cout<<"centroids at the beginning:"<<*centroids<<endl;
//    int64_t assign[1000];
//    for(int64_t i=0;i<n;i++){
//        assign[i] = i % k;
//    }
//    auto weights = torch::rand({n});
//    auto hassign = torch::zeros({k});
//    cst->computeCentroids(d,k,n,k_frozen,x_in, assign, weights, &hassign, centroids);
//    cout<<"centroids at the end:" << *centroids<<endl;
//    cout<<hassign<<endl;
//    hassign[3] = 0;
//    cst->splitClusters(d,k,n,k_frozen, &hassign, centroids);
//    cout<<"centroids after split:" << *centroids<<endl;
//    REQUIRE(a == 0);
//}
