//
// Created by tony on 10/05/23.
//

#include <DataLoader/AbstractDataLoader.h>
#include <Utils/IntelliLog.h>
//do nothing in abstract class
using namespace std;

bool CANDY::AbstractDataLoader::hijackConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  return true;
}
bool CANDY::AbstractDataLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);

  return true;
}

torch::Tensor CANDY::AbstractDataLoader::getData() {
  return torch::rand({1, 1});
}

torch::Tensor CANDY::AbstractDataLoader::getQuery() {
  return torch::rand({1, 1});
}
