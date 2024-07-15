//
// Created by Zhenyu on 2021/8/20.
//
#ifndef SESAME_INCLUDE_ALGORITHM_OFFLINECLUSTERING_DBSCAN_HPP_
#define SESAME_INCLUDE_ALGORITHM_OFFLINECLUSTERING_DBSCAN_HPP_
#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/OfflineRefinement/OfflineRefinement.hpp"
#include "Algorithm/Param.hpp"
#include "Sinks/DataSink.hpp"
#include "Utils/UtilityFunctions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace SESAME {

// #define BORDER_POINT 2
class DBSCAN : public OfflineRefinement {
public:
  DBSCAN(const SesameParam &param) {}
  DBSCAN(unsigned int minPts, float eps);
  DBSCAN();
  ~DBSCAN();
  void Run(SesameParam &param, std::vector<PointPtr> &input,
           DataSinkPtr sinkPtr);

  void run(std::vector<PointPtr> &input);
  void produceResult(std::vector<PointPtr> &input, DataSinkPtr sinkPtr);

private:
  std::vector<int> calculateCluster(std::vector<PointPtr> &input,
                                    PointPtr &point) const;
  int expandCluster(std::vector<PointPtr> &input, PointPtr &point,
                    int clusterID) const;
  static bool judgeCorePoint(PointPtr &point, PointPtr &other);
  static double calculateEluDistance(PointPtr &point, PointPtr &other);
  // Obtain private members
  unsigned int getTotalPointSize() const { return pointSize; }
  unsigned int getMinimumClusterSize() const { return min_points; }
  double getEpsilonSize() const { return epsilon; }
  int getClusterID() const { return clusterID; }
  unsigned int pointSize;
  unsigned int min_points;
  int clusterID;
  double epsilon;
};
} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_OFFLINECLUSTERING_DBSCAN_HPP_
