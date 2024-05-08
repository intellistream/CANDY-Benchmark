#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <Utils/ConfigMap.hpp>

using namespace std;

TEST_CASE("Test basic", "[short]")
{
  int a = 0;
  // place your test here
  REQUIRE(a == 0);
}

TEST_CASE("Generate spase matrix", "[short]")
{
  int a = 0;
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  cfg->edit("aRow", (uint64_t) 6);
  cfg->edit("aCol", (uint64_t) 6);
  cfg->edit("bCol", (uint64_t) 6);
  cfg->edit("aDensity", (double) 1.0);
  cfg->edit("bDensity", (double) 0.5);
  cfg->edit("aReduce", (uint64_t) 1);
  cfg->edit("bReduce", (uint64_t) 2);
  REQUIRE(a == 0);
}