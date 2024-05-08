/*! \file OnlineInsert.cpp*/
//
// Created by tony on 06/01/24.
//
#include <iostream>
#include <torch/torch.h>
#include <CANDY.h>
#include <Utils/UtilityFunctions.h>
using namespace INTELLI;

/**
 *
 */
int main(int argc, char **argv) {

  /**
   * @brief 1. load the configs
   */
  INTELLI::ConfigMapPtr inMap = newConfigMap();
  //size_t incrementalBuildTime = 0, incrementalSearchTime = 0;
  if (inMap->fromCArg(argc, argv) == false) {
    if (argc >= 2) {
      std::string fileName = "";
      fileName += argv[1];
      if (inMap->fromFile(fileName)) { std::cout << "load config from file " + fileName << endl; }
    }
  }
  /**
   * @brief 2. create the data and query
   */
  CANDY::DataLoaderTable dataLoaderTable;
  std::string dataLoaderTag = inMap->tryString("dataLoaderTag", "random", true);

  auto dataLoader = dataLoaderTable.findDataLoader(dataLoaderTag);
  INTELLI_INFO("2.Data loader =" + dataLoaderTag);
  if (dataLoader == nullptr) {
    return -1;
  }
  dataLoader->setConfig(inMap);
  auto dataTensor = dataLoader->getData();
  auto queryTensor = dataLoader->getQuery();
  torch::Tensor buildTensor = torch::zeros({dataTensor.size(0) + queryTensor.size(0), dataTensor.size(1)});
  int64_t lastNNZ = -1;
  INTELLI::IntelliTensorOP::appendRowsBufferMode(&buildTensor, &dataTensor, &lastNNZ);
  INTELLI::IntelliTensorOP::appendRowsBufferMode(&buildTensor, &queryTensor, &lastNNZ);
  INTELLI_INFO(
      "Demension =" + std::to_string(dataTensor.size(1)) + ",#data=" + std::to_string(dataTensor.size(0)));

  /**
   * @brief 4. creat index
   */
  CANDY::IndexTable indexTable;
  std::string indexTag = inMap->tryString("indexTag", "flat", true);
  auto indexPtr = indexTable.getIndex(indexTag);
  if (indexPtr == nullptr) {
    return -1;
  }
  indexPtr->setConfig(inMap);
  indexPtr->startHPC();
  torch::manual_seed(2);
  int64_t sampleRows = inMap->tryI64("sampleRows", 8192, true);
  int64_t knownRows = inMap->tryI64("knownRows", buildTensor.size(0), true);

  if (knownRows != buildTensor.size(0)) {
    INTELLI_INFO("I will pretend rows from" + std::to_string(knownRows) + "upon are UNKNOWN");
  }
  auto knownTensor = buildTensor.slice(0, 0, knownRows);
  // Probability distribution
  int64_t n = knownTensor.size(0);
  torch::Tensor probs = torch::ones(n) / n;  // default: uniform
  // Sample k indices from range 0 to n for given probability distribution
  torch::Tensor indices = torch::multinomial(probs, sampleRows, true);
  auto trainTensor = knownTensor.index_select(0, indices);

  INTELLI_INFO("3.Building NOW!!!");
  auto start = std::chrono::high_resolution_clock::now();
  indexPtr->offlineBuild(trainTensor);
  auto tDone = chronoElapsedTime(start);
  indexPtr->endHPC();
  auto briefOutCfg = newConfigMap();
  briefOutCfg->edit("buildTime", tDone / 1000);
  briefOutCfg->toFile("offlineBuild.csv");

  return 0;
}