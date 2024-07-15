// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 19/07/2021.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_DATASTRUCTUREFACTORY_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_DATASTRUCTUREFACTORY_HPP_

#include "Algorithm/DataStructure/CFTree.hpp"
#include "Algorithm/DataStructure/Cache.hpp"
#include "Algorithm/DataStructure/CoresetTree.hpp"
#include "Algorithm/DataStructure/DPNode.hpp"
#include "Algorithm/DataStructure/DPTree.hpp"
#include "Algorithm/DataStructure/FeatureVector.hpp"
#include "Algorithm/DataStructure/MicroCluster.hpp"
#include "Algorithm/DataStructure/OutlierReservoir.hpp"
#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/DataStructure/Snapshot.hpp"
#include "Algorithm/DataStructure/TreeNode.hpp"
#include "Algorithm/DataStructure/WeightedAdjacencyList.hpp"

#include <memory>

namespace SESAME {

class DataStructureFactory {
public:
  static TreeNodePtr createTreeNode();
  static void clearTreeNode(TreeNodePtr treeNode);
  static MicroClusterPtr createMicroCluster(int dim, int id);
  static MicroClusterPtr createMicroCluster(int dim, int id, PointPtr dataPoint,
                                            double radius);
  static void clearMicroCluster(MicroClusterPtr microCluster);
  static SnapshotPtr createSnapshot(MicroClusters &otherMicroClusters,
                                    int elapsedTime);
  static void clearSnapshot(SnapshotPtr snapshot);
  static CFTreePtr createCFTree();
  static NodePtr createNode();
  // EDMStream
  static DPTreePtr createDPTree(int num, double r);
  static DPNodePtr createDPNode();
  static DPNodePtr createDPNode(SESAME::PointPtr p, double time);
  static CachePtr creatCache();
  static CachePtr creatCache(int num, double a, double lamd, double r);
  static OutPtr createOutlierReservoir();
  static OutPtr createOutlierReservoir(double r, double a, double lamd);

  static MicroClusterPairPtr
  createMicroClusterPair(MicroClusterPtr microCluster1,
                         MicroClusterPtr microCluster2);
  static void clearMicroClusterPair(MicroClusterPairPtr microClusterPair);
  static AdjustedWeightPtr createAdjustedWeight(double weight, int pointTime,
                                                timespec pointTime0);
  static void clearAdjustedWeight(AdjustedWeightPtr adjustedWeight);
};
} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_DATASTRUCTUREFACTORY_HPP_
