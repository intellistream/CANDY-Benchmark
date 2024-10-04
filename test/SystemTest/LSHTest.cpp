#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <CANDY.h>
#include <iostream>
#include <CANDY/LSH.h>
#include <torch/torch.h>
#include <Utils/IntelliLog.h>

using namespace std;
using namespace CANDY;

TEST_CASE("Test LSH", "[short]")
{
    INTELLI::ConfigMapPtr cfg = newConfigMap();
    cfg->edit("vecDim", (int64_t) 10);
    cfg->edit("hashFunctionNum", (int64_t) 10);
    LSH lsh;
    lsh.setConfig(cfg);

    auto x_in = torch::rand({200, 10});
    lsh.insertTensor(x_in);

    auto x_query = x_in.slice(0, 20, 22);
    auto result = lsh.searchTensor(x_query, 1);
    for (size_t i = 0; i < result.size(); i++) {
        auto new_tensorPtr = newTensor(x_query.slice(0, i, i + 1));
        REQUIRE(new_tensorPtr->equal(result[i].slice(0, 0, 1)));
    }

    lsh.deleteTensor(x_query);
    result = lsh.searchTensor(x_query, 1);
    for (size_t i = 0; i < result.size(); i++) {
        auto new_tensorPtr = newTensor(x_query.slice(0, i, i + 1));
        REQUIRE(!new_tensorPtr->equal(result[i].slice(0, 0, 1)));
    }

    auto x_in2 = torch::rand({200, 10});
    lsh.insertTensor(x_in2);
    auto x_query2 = x_in2.slice(0, 20, 25);

    result = lsh.searchTensor(x_query2, 1);
    for (size_t i = 0; i < result.size(); i++) {
        auto new_tensorPtr = newTensor(x_query2.slice(0, i, i + 1));
        REQUIRE(new_tensorPtr->equal(result[i].slice(0, 0, 1)));
    }
}