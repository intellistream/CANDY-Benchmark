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

torch::Tensor CANDY::AbstractDataLoader::getDataAt(int64_t startPos, int64_t endPos) {
  auto ru = getData();
  return ru.slice(0, startPos, endPos).nan_to_num(0);
}
torch::Tensor CANDY::AbstractDataLoader::getQueryAt(int64_t startPos, int64_t endPos) {
  auto ru = getQuery();
  return ru.slice(0, startPos, endPos).nan_to_num(0);
}
int64_t CANDY::AbstractDataLoader::size() {
  auto ru = getData();
  return ru.size(0);
}
int64_t CANDY::AbstractDataLoader::getDimension() {
  auto ru = getData();
  return ru.size(1);
}