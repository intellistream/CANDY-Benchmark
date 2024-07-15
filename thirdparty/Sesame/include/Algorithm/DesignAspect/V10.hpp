//
// Created by tuidan on 2021/9/1.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DESIGNASPECT_V10_HPP_
#define SESAME_INCLUDE_ALGORITHM_DESIGNASPECT_V10_HPP_
#include <Algorithm/Algorithm.hpp>
#include <Algorithm/DataStructure/Cache.hpp>
#include <Algorithm/DataStructure/DPNode.hpp>
#include <Algorithm/DataStructure/DPTree.hpp>
#include <Algorithm/DataStructure/DataStructureFactory.hpp>
#include <Algorithm/DataStructure/OutlierReservoir.hpp>
#include <Sinks/DataSink.hpp>
#include <Utils/BenchmarkUtils.hpp>
#include <unordered_set>

namespace SESAME {
class V10Parameter : public SesameParam {
public:
  bool isInit = false;

  double alpha;
  double lamda;
  double beta;
  int num_cache;
  double radius;
  int landmark;
  double minDelta;
  int opt;
};

class V10 : public Algorithm {
public:
  double deltaT;
  int actCluMaxNum = 10000;
  double minRho;
  double alpha;

  V10Parameter V10Param;
  DPTreePtr dpTree;
  OutPtr outres;
  CachePtr cache;
  std::vector<PointPtr> onlineCenters;
  std::unordered_set<ClusterPtr> clusters;

  V10(param_t &cmd_params);
  void Init() override;
  void setMinDelta(double minDelta);
  void CountNode(const SESAME::DPNodePtr &node, int &num);
  void InitDP(double time);
  SESAME::DPNodePtr streamProcess(SESAME::PointPtr p, int opt, double time);
  double computeAlpha();
  double adjustMinDelta();
  void delCluster();
  SESAME::DPNodePtr retrive(SESAME::PointPtr p, int opt, double time);

  void RunOnline(SESAME::PointPtr input) override;

  void RunOffline(DataSinkPtr sinkPtr) override;
  void OutputOnline(std::vector<PointPtr> &onlineCenters) override;
};
} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DESIGNASPECT_V10_HPP_
