//
// Created by Isshin on 2024/1/11.
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
TEST_CASE("Test PQ index", "[short]")
{
  torch::manual_seed(114514);
  // place your test here
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::IndexTable it;
  auto PQIdx = it.getIndex("pq");
  //PQIndex* PQIdx = new PQIndex();
  cfg->edit("vecDim", (int64_t) 32);
  cfg->edit("subQuantizers", (int64_t) 4);
  cfg->edit("nBits", (int64_t) 8);
  PQIdx->setConfig(cfg);
//    cout<<"d: "<<PQIdx->pq_.d_<<endl;
//    cout<<"nbits: "<<PQIdx->pq_.nbits_<<endl;
//    cout<<"M: "<<PQIdx->pq_.M_<<endl;
//    cout<<"subK: "<<PQIdx->pq_.subK_<<endl;
//    cout<<"subVecDims: "<<PQIdx->pq_.subvecDims_<<endl;
//    cout<<"code_size: "<<PQIdx->pq_.code_size_<<endl;
  //int64_t nx = 5000;
  auto x_in_base = 100 * torch::rand({5000, 32});
  PQIdx->offlineBuild(x_in_base);
  cout << "train" << endl;

  //cout<<"centroids:"<<endl;
  //cout<< PQIdx->pq_.centroids_<<endl;
  //cout<<"codes"<<endl;
  //cout<<PQIdx->codes_.size()<<endl;
//    for(int i = 0; i<PQIdx->npoints_; i++){
//        cout<<"code "<<i<<": ";
//        for(int code_step =0; code_step <PQIdx->pq_.code_size_; code_step++){
//            cout<<PQIdx->codes_[i*PQIdx->pq_.code_size_ + code_step] <<" ";
//        }
//        cout<<endl;
//    }

//    PQIdx->pq_.compute_distance_tables(new_x, &dist, 4);
  //cout<<"dist table"<<endl;
  //cout<<dist<<endl;
  auto new_nx = 5;
  auto x_new = 100 * torch::rand({(int64_t) new_nx, 32});

  PQIdx->insertTensor(x_new);
//    for(int i=0; i<new_nx; i++){
//        auto x_new_single = x_new.slice(0, i, i+1);
//        PQIdx->insertTensor(x_new_single);
//    }
  cout << "codes size: " << PQIdx->rawData().size(0) << endl;
  cout << "search result" << endl;
  int querysize = 5;
  for (int i = 0; i < querysize; i++) {
    auto query_old = x_in_base.slice(0, i, i + 1);
    auto query_new = x_new.slice(0, i, i + 1);
    cout << "old query" << endl;
    cout << PQIdx->searchTensor(query_old, 25)[0] << endl;
    cout << "new query" << endl;
    cout << PQIdx->searchTensor(query_new, 25)[0] << endl;
  }
  cout << "raw data codes " << endl;
  //cout<<PQIdx->rawData()<<endl;


}
