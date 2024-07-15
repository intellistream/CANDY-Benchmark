//
// Created by 1124a on 2021/8/23.
//
#ifndef SESAME_INCLUDE_ALGORITHM_DENSTREAM_HPP_
#define SESAME_INCLUDE_ALGORITHM_DENSTREAM_HPP_

#include "Algorithm/Algorithm.hpp"
#include "Algorithm/DataStructure/MicroCluster.hpp"
#include "Algorithm/OfflineRefinement/DBSCAN.hpp"
#include "Algorithm/WindowModel/DampedWindow.hpp"
#include "Utils/BenchmarkUtils.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>

namespace SESAME {
#define noVisited (-1)
class DenStreamParams : public SesameParam {
public:
  int buf_sizeSize;        // number of data point for Initialization
  unsigned int min_points; // minimum point of core point in DBSCAN
  double epsilon; // maximum distance if point belongs to the density area of
                  // core point
  double base;    // base of decay function
  double lambda;
  double mu;   // used to calculate minimum weight minWeight=mu*beta;
  double beta; // used to calculate minimum weight
};

class DenStream : public Algorithm {
public:
  DenStreamParams denStreamParams;
  std::shared_ptr<DBSCAN>
      dbscan; // used for initialization and offline re-clustering
  DampedWindowPtr dampedWindow;
  std::vector<MicroClusterPtr> pMicroClusters;
  std::vector<MicroClusterPtr> oMicroClusters;
  int startTime;
  int pointArrivingTime; // clock_t
  int lastPointTime;
  int lastUpdateTime; // for calculating time interval
  double Tp;
  int pMicroClusterIndex;
  int oMicroClusterIndex;

  DenStream(param_t &cmd_params);
  ~DenStream();
  void Init() override;
  void RunOnline(PointPtr input) override;
  void RunOffline(DataSinkPtr sinkPtr) override;
  double getMinWeight() { return minWeight; };

private:
  bool isInitial = false;
  vector<PointPtr> initialBuffer;
  double minWeight;
  void Init(vector<PointPtr> &initData);
  void merge(PointPtr dataPoint);
  void pointsNearCorePoint(vector<PointPtr> &initData,
                           std::vector<int> pointIndex,
                           MicroClusterPtr microCluster);
  MicroClusterPtr nearestNeighbor(PointPtr dataPoint,
                                  std::vector<MicroClusterPtr> microClusters);
  bool mergeToMicroCluster(PointPtr dataPoint,
                           std::vector<MicroClusterPtr> microClusters);
  bool mergeToOMicroCluster(PointPtr dataPoint,
                            std::vector<MicroClusterPtr> microClusters);
  static void microClusterToPoint(std::vector<MicroClusterPtr> &microClusters,
                                  vector<PointPtr> &points);
  int findIndex(std::vector<MicroClusterPtr> &microClusters,
                MicroClusterPtr MC);
  // TODO overlap functions with Clustream, may need to remove to utils folder
};
} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DENSTREAM_HPP_
