#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <CANDY.h>
#include <iostream>
#include <CANDY/LSH.h>

using namespace std;
using namespace CANDY;

TEST_CASE("Test LSH", "[short]")
{
    INTELLI::ConfigMapPtr cfg = newConfigMap();
    cfg->edit("vecDim", (int64_t) 768);
    cfg->edit("hashFunctionNum", (int64_t) 10);
    LSH lsh;
    lsh.setConfig(cfg);

    auto x_in = torch::rand({2000000, 768});
    lsh.insertTensor(x_in);

    auto x_query = x_in.slice(0, 20, 22);
    auto result = lsh.searchTensor(x_query, 1);
    for (size_t i = 0; i < result.size(); i++) {
        REQUIRE(result[i] != nullptr);
        REQUIRE(result[i]->equal(x_query.slice(0, i, i + 1)));
    }

    lsh.deleteTensor(x_query);
    result = lsh.searchTensor(x_query, 1);
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] != nullptr) REQUIRE(!result[i]->equal(x_query.slice(0, i, i + 1)));
    }

    auto x_in2 = torch::rand({2000000, 768});
    lsh.insertTensor(x_in2);
    auto x_query2 = x_in2.slice(0, 20, 25);

    result = lsh.searchTensor(x_query2, 1);
    for (size_t i = 0; i < result.size(); i++) {
        REQUIRE(result[i]->equal(x_query2.slice(0, i, i + 1)));
    }

    int k = 3;

    result = lsh.searchTensor(x_query2, k);
    for (size_t i = 0; i < x_query2.size(0); i++) {
        std::cout << "\033[32m" << "================ Query " << i << " ===============" << std::endl << "\033[0m";
        for (size_t j = 0; j < k; j++) {
            if (result[i * k + j] == nullptr) {
                std::cout << "No neighbor found" << std::endl;
                continue;
            }
            std::cout << "Neighbor in dist " << torch::dist(result[i * k + j]->reshape(-1), x_query2.slice(0, i, i + 1).reshape(-1)).item<double>() << std::endl;
        }
    }
}