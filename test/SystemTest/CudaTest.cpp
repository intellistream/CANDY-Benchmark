//
// Created by tony on 12/06/24.
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
extern torch::Tensor CudaMM(torch::Tensor a, torch::Tensor b);
using namespace std;

TEST_CASE("Test basic", "[short]")
{
  int a = 0;
  // place your test here
  REQUIRE(a == 0);
}

TEST_CASE("Test cuda mm", "[short]")
{
  auto a=torch::rand({128,128});
  auto c=CudaMM(a,a);
  std::cout<<c;
  REQUIRE(c.size(0)==128);
}