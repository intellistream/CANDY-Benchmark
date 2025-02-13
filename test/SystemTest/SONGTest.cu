/*
* Copyright (C) 2024 by the INTELLI team
 * Created on: 25-2-12 下午7:26
 * Description: ${DESCRIPTION}
 */
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <CANDY/SONG/SONG.hpp>
#include <torch/torch.h>
#include <CANDY.h>

using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;

TEST_CASE("SONG: Insert and Search", "[SONG]") {
    CANDY::SONG index;
    INTELLI::ConfigMapPtr config = newConfigMap();
    config->edit("vecDim", (int64_t)4);
    config->edit("vecVolume", (int64_t)100);
    config->edit("metricType", "L2");
    REQUIRE(index.setConfig(config));

    // 生成 5 个随机 4 维张量
    torch::Tensor data = torch::rand({50, 4});
    REQUIRE(index.insertTensor(data));

    // 检查索引大小是否正确
    REQUIRE(index.size() == 50);

    // 进行搜索
    torch::Tensor query = torch::rand({1, 4});
    auto results = index.searchTensor(query, 10);

    REQUIRE(results.size() == 1);
    REQUIRE(results[0].size(0) == 10);

    // 输出搜索结果
    std::cout << "Search Results: " << results[0] << std::endl;
}
