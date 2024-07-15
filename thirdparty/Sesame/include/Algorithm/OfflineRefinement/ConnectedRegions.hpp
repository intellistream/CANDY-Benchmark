//
// Created by 1124a on 2021/11/23.
//

#ifndef SESAME_INCLUDE_ALGORITHM_OFFLINECLUSTERING_CONNECTEDREGIONS_HPP_
#define SESAME_INCLUDE_ALGORITHM_OFFLINECLUSTERING_CONNECTEDREGIONS_HPP_

#include "Algorithm/DataStructure/DataStructureFactory.hpp"
#include "Algorithm/DataStructure/MicroCluster.hpp"
#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/DataStructure/WeightedAdjacencyList.hpp"
#include "Algorithm/OfflineRefinement/OfflineRefinement.hpp"
#include "Utils/Logger.hpp"

namespace SESAME {

class ConnectedRegions : public OfflineRefinement {
public:
  double alpha;      // intersection factor, alpha
  double min_weight; // minimum weight
  std::vector<std::vector<MicroClusterPtr>> finalClusters;
  unordered_map<int, std::vector<int>> connecvtivityGraphId;
  ConnectedRegions();
  ConnectedRegions(double alpha, double min_weight);
  void connection(std::vector<MicroClusterPtr> &microClusters,
                  SESAME::WeightedAdjacencyList &weightedAdjacencyList);
  std::vector<PointPtr> ResultsToDataSink();

  /**
   * @Description:  insert vertices and entries into connectivity graph when
   * micro cluster pair connectivity value greater than the intersection
   * threshold if the graph has testing micro cluster, add connected strong MC
   * in the corresponding entries else, create new V,E into the graph
   * @Param: connectivity graph, micro cluster 1 and 2
   * @Return: void
   */
  void insertIntoGraph(const std::vector<MicroClusterPtr> &microClusters,
                       int microClusterId, int OtherId);
  void insertIntoGraph(const std::vector<MicroClusterPtr> &microClusters,
                       int microClusterId);
  /**
   * @Description:  findConnectedComponents function visit the existing
   * connectivity graph and find all connected strong MCs that will finally form
   * arbitrary-shaped macro clusters each macro cluster will be stored as a
   * vector of micro clusters, which will be transformed into point that stores
   * in sink later
   * @Param: connectivity graph
   * @Return: void
   */
  void
  findConnectedComponents(const std::vector<MicroClusterPtr> &microClusters);
};

} // namespace SESAME

#endif // SESAME_INCLUDE_ALGORITHM_OFFLINECLUSTERING_CONNECTEDREGIONS_HPP_
