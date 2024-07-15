// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 19/07/2021.
//

#include <Algorithm/DataStructure/DataStructureFactory.hpp>

SESAME::TreeNodePtr SESAME::DataStructureFactory::createTreeNode() {
  return std::make_shared<TreeNode>();
}
void SESAME::DataStructureFactory::clearTreeNode(SESAME::TreeNodePtr treeNode) {
  treeNode.reset();
}

SESAME::MicroClusterPtr
SESAME::DataStructureFactory::createMicroCluster(int dim, int id) {
  return std::make_shared<SESAME::MicroCluster>(dim, id);
}
SESAME::MicroClusterPtr SESAME::DataStructureFactory::createMicroCluster(
    int dim, int id, PointPtr dataPoint, double radius) {
  return std::make_shared<SESAME::MicroCluster>(dim, id, dataPoint, radius);
}
void SESAME::DataStructureFactory::clearMicroCluster(
    SESAME::MicroClusterPtr microCluster) {
  microCluster.reset();
}

SESAME::SnapshotPtr SESAME::DataStructureFactory::createSnapshot(
    SESAME::MicroClusters &otherMicroClusters, int elapsedTime) {
  return std::make_shared<SESAME::Snapshot>(otherMicroClusters, elapsedTime);
}

void SESAME::DataStructureFactory::clearSnapshot(SESAME::SnapshotPtr Snapshot) {
  Snapshot.reset();
}
SESAME::CFTreePtr SESAME::DataStructureFactory::createCFTree() {
  return std::make_shared<SESAME::CFTree>(0, 0, 0);
}
SESAME::NodePtr SESAME::DataStructureFactory::createNode() {
  return std::make_shared<SESAME::CFNode>();
}
SESAME::DPNodePtr SESAME::DataStructureFactory::createDPNode() {
  return std::make_shared<SESAME::DPNode>();
}
SESAME::DPNodePtr SESAME::DataStructureFactory::createDPNode(SESAME::PointPtr p,
                                                             double time) {
  return std::make_shared<SESAME::DPNode>(p, time);
}
SESAME::OutPtr SESAME::DataStructureFactory::createOutlierReservoir() {
  return std::make_shared<SESAME::OutlierReservoir>();
}
SESAME::CachePtr SESAME::DataStructureFactory::creatCache() {
  return std::make_shared<SESAME::Cache>();
}
SESAME::OutPtr
SESAME::DataStructureFactory::createOutlierReservoir(double r, double a,
                                                     double lamd) {
  return std::make_shared<SESAME::OutlierReservoir>(r, a, lamd);
}
SESAME::CachePtr SESAME::DataStructureFactory::creatCache(int num, double a,
                                                          double lamd,
                                                          double r) {
  return std::make_shared<SESAME::Cache>(num, a, lamd, r);
}
SESAME::DPTreePtr SESAME::DataStructureFactory::createDPTree(int num,
                                                             double r) {
  return std::make_shared<SESAME::DPTree>(num, r);
}

SESAME::MicroClusterPairPtr
SESAME::DataStructureFactory::createMicroClusterPair(
    MicroClusterPtr microCluster1, MicroClusterPtr microCluster2) {
  return std::make_shared<SESAME::MicroClusterPair>(microCluster1,
                                                    microCluster2);
}

void SESAME::DataStructureFactory::clearMicroClusterPair(
    MicroClusterPairPtr microClusterPair) {
  microClusterPair.reset();
}

SESAME::AdjustedWeightPtr
SESAME::DataStructureFactory::createAdjustedWeight(double weight, int pointTime,
                                                   timespec pointTime0) {
  return std::make_shared<SESAME::AdjustedWeight>(weight, pointTime,
                                                  pointTime0);
}
void SESAME::DataStructureFactory::clearAdjustedWeight(
    SESAME::AdjustedWeightPtr adjustedWeight) {
  adjustedWeight.reset();
}
