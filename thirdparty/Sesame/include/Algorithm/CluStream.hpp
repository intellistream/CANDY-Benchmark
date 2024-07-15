// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by 1124a on 2021/8/16.
//

#ifndef SESAME_INCLUDE_ALGORITHM_CLUSTREAM_HPP_
#define SESAME_INCLUDE_ALGORITHM_CLUSTREAM_HPP_

#include "Algorithm/Algorithm.hpp"
#include "Algorithm/DataStructure/MicroCluster.hpp"
#include "Algorithm/DataStructure/Snapshot.hpp"
#include "Algorithm/OfflineRefinement/KMeans.hpp"
#include "Algorithm/WindowModel/LandmarkWindow.hpp"
#include "Timer/Timer.hpp"
#include "Utils/BenchmarkUtils.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>

namespace SESAME {

class CluStreamParameter : public SesameParam {
public:
  int num_last_arr;
  int time_window;
  unsigned int time_interval;
  int num_clusters;         // total number of micro clusters online
  int num_offline_clusters; // total number of micro clusters online
  double radius;            // radius factor
  int buf_size;
  int offline_time_window;
  int seed;
};

const double doubleMax = std::numeric_limits<double>::max();
class CluStream : public Algorithm {
public:
  CluStreamParameter CluStreamParam;
  std::shared_ptr<KMeans> kmeans; // used for offline initialization
  LandmarkWindowPtr window;
  MicroClusters
      microClusters; // Defined in Snapshot, std::vector <MicroclusterPtr>
  MicroClusters delMicroClusters;
  int pointsFitted;
  int pointsForgot;
  int pointsMerged;
  int startTime;
  int lastUpdateTime;
  CluStream(param_t &cmd_params);
  ~CluStream();

  void Init() override;
  void RunOnline(PointPtr input) override;
  void RunOffline(DataSinkPtr sinkPtr) override;

private:
  void initOffline(vector<PointPtr> &initData, vector<PointPtr> &initialData);
  void incrementalCluster(PointPtr data);
  double calRadius(MicroClusterPtr closestCluster);
  void insertIntoCluster(PointPtr data, MicroClusterPtr closestCluster);
  bool deleteCreateCluster(PointPtr data);
  void MergeCreateCluster(PointPtr data);
  void microClusterToPoint(MicroClusters &microClusters,
                           vector<PointPtr> &points) const;
  static double distance(dataPoint a, dataPoint b, int dim);

  bool initilized = false;
  vector<PointPtr> initialInputs;
};

} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_CLUSTREAM_HPP_
