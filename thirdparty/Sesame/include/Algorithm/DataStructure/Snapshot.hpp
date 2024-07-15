// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by 1124a on 2021/8/16.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_SNAPSHOT_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_SNAPSHOT_HPP_
#include <Algorithm/DataStructure/MicroCluster.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <limits>

namespace SESAME {
class Snapshot;
typedef std::shared_ptr<Snapshot> SnapshotPtr;

typedef std::vector<MicroClusterPtr> MicroClusters;
typedef std::vector<SnapshotPtr> QueueSnapshotPtr;
typedef std::vector<QueueSnapshotPtr> QueueOrderSnapshot;

class Snapshot {
public:
  int elapsedTime;
  MicroClusters microClusters;

  /**
  QueueSnapshotPtr: Data Structure representing order ith snapshots list
  QueueOrderSnapshotPtr: Data Structure representing orders
  **/
  Snapshot(MicroClusters &otherMicroClusters, int elapsedTime);
  ~Snapshot();
  static SnapshotPtr findSnapshot(QueueOrderSnapshot orderSnapShots,
                                  int landmarkTime, int currentElapsedTime,
                                  unsigned int currentOrder);

  static SnapshotPtr substractSnapshot(SnapshotPtr snapshotCurrent,
                                       const SnapshotPtr &snapshotLandmark,
                                       unsigned int num_clusters);

  SnapshotPtr copy();
};
} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_SNAPSHOT_HPP_
