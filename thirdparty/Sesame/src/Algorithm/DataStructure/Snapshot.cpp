// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by 1124a on 2021/8/16.
//

#include <Algorithm/DataStructure/DataStructureFactory.hpp>
#include <Algorithm/DataStructure/Snapshot.hpp>
#include <Utils/Logger.hpp>

SESAME::Snapshot::Snapshot(MicroClusters &otherMicroClusters, int elapsedTime) {
  this->elapsedTime = elapsedTime;
  for (int a = 0; a < otherMicroClusters.size(); a++) {
    this->microClusters.push_back(otherMicroClusters[a]->copy());
  }
}
SESAME::Snapshot::~Snapshot() {
  std::vector<MicroClusterPtr>().swap(this->microClusters);
}
SESAME::SnapshotPtr
SESAME::Snapshot::findSnapshot(QueueOrderSnapshot orderSnapShots,
                               int landmarkTime, int currentElapsedTime,
                               unsigned int currentOrder) {
  int i = 0;
  int minDistance = currentElapsedTime;
  int dist, tempMinDistance = -1, elapsedTimeSnapshot;
  int finalOrder, indexSnapshot; // the index of found snapshotPtr
  // Found snapshotPtr
  while (i <= currentOrder) {
    for (int a = 0; a < orderSnapShots[i].size(); a++) {
      elapsedTimeSnapshot = orderSnapShots[i][a]->elapsedTime;
      dist = abs((int)(elapsedTimeSnapshot - landmarkTime));
      if (minDistance > dist ||
          (minDistance == dist && tempMinDistance < elapsedTimeSnapshot)) //
      {
        minDistance = dist;
        tempMinDistance = elapsedTimeSnapshot;
        finalOrder = i;
        indexSnapshot = a;
        //        SESAME_INFO(" this one Elapsed Time is "<<
        //        elapsedTimeSnapshot<<",elapsed time is "<<tempMinDistance);
        // SESAME_INFO(" in order  "<< finalOrder<<",  "<<tempMinDistance<<"th
        // snapshots shared ptr");
      }
    }
    i++;
  }

  SnapshotPtr nearestSnapshot = DataStructureFactory::createSnapshot(
      orderSnapShots[finalOrder][indexSnapshot]->microClusters,
      orderSnapShots[finalOrder][indexSnapshot]->elapsedTime);
  // SESAME_INFO("close landmark "<< tempMinDistance);
  // SESAME_INFO("just for contrast : "<<
  // orderSnapShots[finalOrder][indexSnapshot]->elapsedTime);
  return nearestSnapshot;
}
SESAME::SnapshotPtr
SESAME::Snapshot::substractSnapshot(SnapshotPtr snapshotCurrent,
                                    const SnapshotPtr &snapshotLandmark,
                                    unsigned int num_clusters) {
  //  SESAME_INFO("Start substract "<<snapshotCurrent->elapsedTime<<" LANDMARK
  //  ET
  //  "<<snapshotLandmark->elapsedTime);
  for (unsigned int i = 0; i < num_clusters;
       i++) { // If the micro cluster raised from merging
    if (snapshotCurrent->microClusters[i]->id.size() > 1) {
      for (unsigned int j = 0; j < num_clusters; j++) {
        if (snapshotLandmark->microClusters[j]->id.size() > 1) {
          if (snapshotCurrent->microClusters[i]->judgeMerge(
                  snapshotLandmark->microClusters[j]))
            snapshotCurrent->microClusters[i]->subtractClusterVector(
                snapshotLandmark->microClusters[j]);
        } else {
          int clusterIdLandmark;
          clusterIdLandmark = snapshotLandmark->microClusters[j]->id[0];
          if (std::find(snapshotCurrent->microClusters[i]->id.begin(),
                        snapshotCurrent->microClusters[i]->id.end(),
                        clusterIdLandmark) !=
              snapshotCurrent->microClusters[i]->id.end())
            snapshotCurrent->microClusters[i]->subtractClusterVector(
                snapshotLandmark->microClusters[j]);
        }
      }
    }
    // The micro cluster raised from creating new ones or the original ones
    else {
      for (unsigned int j = 0; j < num_clusters; j++) {
        if (snapshotLandmark->microClusters[j]->id.size() == 1) {
          int clusterIdLandmark = snapshotLandmark->microClusters[j]->id[0];
          if (snapshotCurrent->microClusters[i]->id[0] == clusterIdLandmark)
            snapshotCurrent->microClusters[i]->subtractClusterVector(
                snapshotLandmark->microClusters[j]);
        }
      }
    }
  }

  return snapshotCurrent;
}
SESAME::SnapshotPtr SESAME::Snapshot::copy() {
  return std::make_shared<Snapshot>(*this);
}
