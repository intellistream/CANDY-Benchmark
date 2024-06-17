//
// Created by rubato on 16/6/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <iostream>
#include <CANDY.h>
#include <CANDY/DAGNNIndex.h>
#include <faiss/IndexFlat.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;

TEST_CASE("Test DAGNN INIT", "[short]") {
    struct DynamicTuneHNSW::DynamicTuneParams dp;

    int64_t vector_dim = 128;
    int64_t database_size = 200;
    auto dhnsw =  new DynamicTuneHNSW(32,vector_dim,0,dp);


    std::mt19937_64 gen;
    gen = std::mt19937_64(114514);
    std::uniform_real_distribution<> distrib(0,1);
    // Create database with 1000 vectors, each with 768 dimensions
    float *database_data=new float[database_size * vector_dim];
    for (int i = 0; i < database_size; i++) {
        for (int j = 0; j < vector_dim; j++)
        {database_data[ vector_dim* i + j] = distrib(gen);}
        database_data[vector_dim * i] += i / database_size;
    }

    dhnsw->add(database_size/2, database_data);
    dhnsw->add(database_size/2, database_data+database_size/2*vector_dim);

}

TEST_CASE("Test DAGNN GREEDY PHASE IN INSERT", "[insert]") {
    struct DynamicTuneHNSW::DynamicTuneParams dp;

    int64_t vector_dim = 128;
    int64_t database_size = 200;
    auto dhnsw =  new DynamicTuneHNSW(32,vector_dim,0,dp);


    std::mt19937_64 gen;
    gen = std::mt19937_64(114514);
    std::uniform_real_distribution<> distrib(0,1);
    // Create database with 1000 vectors, each with 768 dimensions
    float *database_data=new float[database_size * vector_dim];
    for (int i = 0; i < database_size; i++) {
        for (int j = 0; j < vector_dim; j++)
        {database_data[ vector_dim* i + j] = distrib(gen);}
        database_data[vector_dim * i] += i / database_size;
    }

    dhnsw->add(database_size/2, database_data);
    dhnsw->add(database_size/2, database_data+database_size/2*vector_dim);
    auto entry = dhnsw->entry_points[0];
    dhnsw->direct_link(entry, 1,0);
    dhnsw->direct_link(entry, 2,0);
    dhnsw->direct_link(1, 2,0);
    dhnsw->direct_link(2, 3,0);
    dhnsw->direct_link(2, 4,0);
    dhnsw->direct_link(entry, 3,0);
    dhnsw->direct_link(3, 4,0);
    dhnsw->direct_link(entry, 4,0);
    dhnsw->direct_link(entry, 5,0);
    dhnsw->direct_link(entry, 6,0);

    dhnsw->direct_link(entry, 1,1);
    dhnsw->direct_link(entry, 4,1);
    dhnsw->direct_link(entry, 5,1);
    dhnsw->direct_link(entry, 6,1);

    dhnsw->direct_link(4, 1,0);
    dhnsw->direct_link(4, 2,0);
    dhnsw->direct_link(4, 4,0);
    dhnsw->direct_link(4, 3,0);
    dhnsw->direct_link(4, 5,0);
    dhnsw->direct_link(4, 6,0);
    dhnsw->direct_link(4, 7,0);

    DAGNN::DistanceQueryer disq(vector_dim);
    disq.set_query(database_data+7*vector_dim);
    std::priority_queue<DynamicTuneHNSW::Candidate> candidates;
    auto node = dhnsw->linkLists[7];
    dhnsw->greedy_insert(disq, *node, candidates);
}