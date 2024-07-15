//
// Created by tuidan on 2021/9/1.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_DPTREE_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_DPTREE_HPP_
#include <Algorithm/DataStructure/DPNode.hpp>
#include <Algorithm/DataStructure/FeatureVector.hpp>
#include <Algorithm/DataStructure/OutlierReservoir.hpp>
#include <Algorithm/DataStructure/Point.hpp>
#include <iostream>
#include <memory>
#include <vector>
namespace SESAME {

// define the share point of the class object
class DPTree;
typedef std::shared_ptr<DPTree> DPTreePtr;

class DPTree {
private:
  double lastTime;
  int size;
  int num;
  std::vector<SESAME::DPNodePtr> Clus;

  double a;
  double lamd;
  double CluR;

  int cluLabel;

public:
  double GetLastTime();
  void SetLastTime(double last_time);
  int GetSize();
  void SetSize(int size);
  int GetNum();
  void SetNum(int num);
  std::vector<SESAME::DPNodePtr> &GetClus();
  void SetClus(std::vector<SESAME::DPNodePtr> &clus);
  double GetA();
  void SetA(double a);
  double GetLamd();
  void SetLamd(double lamd);
  double GetCluR();
  void SetCluR(double clu_r);
  int GetCluLabel();
  void SetCluLabel(int clu_label);
  double GetMinDelta();
  void SetMinDelta(double min_delta);

private:
  double minDelta;

public:
  DPTree();
  ~DPTree();
  DPTree(int num, double CluR);
  void insert(SESAME::DPNodePtr &cc, int opt);
  void Init(std::vector<SESAME::DPNodePtr> &clus, int size, double minRho,
            double minDelta, SESAME::OutPtr &outs,
            std::unordered_set<ClusterPtr> &clusters);
  SESAME::DPNodePtr findNN(PointPtr p, double coef, int opt, double time);
  void adjustNoDelta(int index);
  void adjustNoOpt(int index);
  void computeDeltaNoOpt(int index);
  void adjustOpt1(int index);
  void computeDeltaF1(int index);
  void adjust(int index);
  void computeHeadDelta();
  void computeDelta(int index);
  void deleteInact(SESAME::OutPtr &outres, double minRho, double time);
  double computeAlpha(double minDelta);
  double adjustMinDelta(double alpha);
  // double djustMinDelta(double alpha, double minDelta);
  double fun(double alpha, double upavg, double downavg, double avg);
  // bool check(SESAME::OutPtr outres);
  // void check(std::vector<SESAME::ClusterPtr> clusters);
  void adjustCluster(std::unordered_set<SESAME::ClusterPtr> &clusters);
};

} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_DPTREE_HPP_
