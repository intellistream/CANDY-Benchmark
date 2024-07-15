//
// Created by 1124a on 2021/8/30.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DBSTREAM_HPP_
#define SESAME_INCLUDE_ALGORITHM_DBSTREAM_HPP_

#include "Algorithm/Algorithm.hpp"
#include "Algorithm/DataStructure/WeightedAdjacencyList.hpp"
#include "Algorithm/OfflineRefinement/ConnectedRegions.hpp"
#include "Utils/BenchmarkUtils.hpp"

namespace SESAME {

typedef std::vector<std::vector<MicroClusterPtr>> Clusters;
class DBStreamParams : public SesameParam {
public:
  double radius;
  double lambda;
  int clean_interval; // Time gap
  double min_weight;  // minimum weight
  double alpha;       // α, intersection factor
  double base;        // base of decay function
};

class DBStream : public Algorithm {
public:
  DBStreamParams dbStreamParams;
  DampedWindowPtr dampedWindow;
  std::vector<MicroClusterPtr> microClusters;
  SESAME::WeightedAdjacencyList weightedAdjacencyList;
  std::vector<MicroClusterPtr>
      microClusterNN; // micro clusters found in function findFixedRadiusNN
  double weakEntry;   // W_weak, weak entries
  double aWeakEntry;
  timespec startTime;
  timespec lastArrivingTime0;
  timespec pointArrivingTime0;
  timespec lastCleanTime0;

  int lastArrivingTime;
  int pointArrivingTime;
  int lastCleanTime;
  int microClusterIndex;
  // Final output of clusters
  Clusters finalClusters;
  ConnectedRegions connectedRegions;
  // Connectivity graph
  // unordered_map<int,std::vector<int>> connecvtivityGraphId;

  DBStream(param_t &cmd_params);
  ~DBStream();
  void Init() override;
  void RunOnline(PointPtr input) override;
  void RunOffline(DataSinkPtr sinkPtr) override;

private:
  bool isInitial = false;

  void update(PointPtr dataPoint);
  bool checkMove(std::vector<MicroClusterPtr> microClusters) const;
  std::vector<MicroClusterPtr> findFixedRadiusNN(PointPtr dataPoint,
                                                 double decayFactor);
  void cleanUp(int nowTime);
};

} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DBSTREAM_HPP_
