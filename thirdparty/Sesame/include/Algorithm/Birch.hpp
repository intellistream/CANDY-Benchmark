//
// Created by tuidan on 2021/8/24.
//

#ifndef SESAME_INCLUDE_ALGORITHM_BIRCH_HPP_
#define SESAME_INCLUDE_ALGORITHM_BIRCH_HPP_
#include <Algorithm/Algorithm.hpp>
#include <Algorithm/DataStructure/CFTree.hpp>
#include <Algorithm/OfflineRefinement/KMeans.hpp>
#include <Algorithm/WindowModel/LandmarkWindow.hpp>
#include <Sinks/DataSink.hpp>
#include <Utils/BenchmarkUtils.hpp>
namespace SESAME {

class BirchParameter : public SesameParam {
public:
  int max_in_nodes;          // B
  int max_leaf_nodes;        // L
  double distance_threshold; // T
};

class Birch : public Algorithm {
public:
  BirchParameter BirchParam;
  std::shared_ptr<KMeans> kmeans; // used for offline initialization
  int leafMask = 0;
  NodePtr root;
  vector<NodePtr> leafNodes;
  CFTreePtr cfTree;
  Birch(param_t &cmd_params);

  ~Birch();

  void Init() override;

  void RunOnline(PointPtr input) override;

  void RunOffline(DataSinkPtr sinkPtr) override;

private:
  void forwardInsert(PointPtr point);
  void backwardEvolution(NodePtr &curNode, PointPtr &point);
  void calculateCorDistance(vector<vector<double>> &distance,
                            vector<NodePtr> &nodes);
  double calculateRadius(PointPtr &point, PointPtr &centroid);
  void selectChild(vector<NodePtr> &children, PointPtr &insertPoint,
                   NodePtr &node);
  double clusterToClusterDist(NodePtr &nodeA, NodePtr &nodeB);
  void pointToClusterDist(PointPtr &insertPoint, NodePtr &node, double &dist);
  void calculateCentroid(CFPtr &cf, PointPtr &centroid);
  void updateNLS(NodePtr &node, PointPtr &point, bool updateAll);
  void initializeCF(CFPtr &cf, int dim);
  void setCFToBlankNode(SESAME::NodePtr &curNode, SESAME::PointPtr &point);
  void addNodeNLSToNode(SESAME::NodePtr &child, SESAME::NodePtr &parent);
  void clearChildParents(vector<SESAME::NodePtr> &children);
};
} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_BIRCH_HPP_
