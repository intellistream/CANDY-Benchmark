// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 20/07/2021.
//

#ifndef SESAME_INCLUDE_ALGORITHM_OFFLINECLUSTERING_KMEANS_HPP_
#define SESAME_INCLUDE_ALGORITHM_OFFLINECLUSTERING_KMEANS_HPP_

#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/OfflineRefinement/OfflineRefinement.hpp"
#include "Algorithm/Param.hpp"
#include "Sinks/DataSink.hpp"
#include "Utils/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace SESAME {
class KMeans : public OfflineRefinement {
public:
  KMeans() {}
  KMeans(const SesameParam &param) {}
  // TODO: use template here
  void Run(SesameParam &param, std::vector<PointPtr> &online_centers,
           DataSinkPtr sinkPtr);
  void Run(SesameParam &param, std::vector<PointPtr> &online_centers,
           std::vector<PointPtr> &results);
  void produceResult(std::vector<std::vector<PointPtr>> &groups,
                     DataSinkPtr sinkPtr);
  void runKMeans(int numberOfCenters, int numberOfInput,
                 std::vector<PointPtr> &centers, std::vector<PointPtr> &input,
                 std::vector<std::vector<PointPtr>> &oldGroups,
                 std::vector<std::vector<PointPtr>> &newGroups, int seed,
                 bool KMeansPP);
  void storeResult(std::vector<std::vector<PointPtr>> &groups,
                   std::vector<PointPtr> &output);

private:
  // Randomly chooses k centres with kMeans++ distribution
  double calculateEluDistance(PointPtr &point, PointPtr &center);
  void calculateClusterCenter(PointPtr &center, std::vector<PointPtr> &group);
  void randomSelectCenters(int numberOfCenters, int numberOfInput,
                           std::vector<PointPtr> &input,
                           std::vector<PointPtr> &centers);
  void selectCentersFromWeight(int numberOfCenters, int numberOfInput,
                               std::vector<PointPtr> &input,
                               std::vector<PointPtr> &centers);
  void groupPointsByCenters(int numberOfCenters, int numberOfInput,
                            std::vector<PointPtr> &input,
                            std::vector<PointPtr> &centers,
                            std::vector<std::vector<PointPtr>> &groups);
  void adjustClusteringCenters(std::vector<PointPtr> &centers,
                               std::vector<std::vector<PointPtr>> &groups);
  void refreshGroup(std::vector<std::vector<PointPtr>> &oldGroups,
                    std::vector<std::vector<PointPtr>> &newGroups);
  void checkStopStatus(bool &flag,
                       std::vector<std::vector<PointPtr>> &oldGroups,
                       std::vector<std::vector<PointPtr>> &newGroups);
};
} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_OFFLINECLUSTERING_KMEANS_HPP_
