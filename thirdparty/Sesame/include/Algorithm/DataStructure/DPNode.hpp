//
// Created by tuidan on 2021/9/9.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_DPNODE_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_DPNODE_HPP_
#include <Algorithm/DataStructure/Point.hpp>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>
namespace SESAME {
class DPNode;
class Cluster;
typedef std::shared_ptr<DPNode> DPNodePtr;
typedef std::shared_ptr<Cluster> ClusterPtr;

class Cluster {
private:
  int label;
  std::unordered_set<DPNodePtr> cells;

public:
  explicit Cluster();
  explicit Cluster(int label);
  void add(DPNodePtr &node);
  void remove(DPNodePtr &node);
  int GetLabel();
  void SetLabel(int label);
  [[nodiscard]] std::unordered_set<DPNodePtr> &GetCells();
  void SetCells(std::unordered_set<DPNodePtr> &cells);
};
class DPNode {
private:
  int cid;
  int Cid;
  int num;
  double rho;
  double delta;
  SESAME::DPNodePtr dep; // TODO: father
  SESAME::PointPtr center;
  double lastTime;
  bool active;
  std::unordered_set<DPNodePtr> sucs; // TODO: children
  SESAME::ClusterPtr cluster;
  // public double sumDelta;
  // public int sucNum;
  double inactiveTime;

  /**
   * we will use dis to quickly update the delta of CluCell
   */
  double dis;

public:
  DPNode();
  ~DPNode();
  DPNode(SESAME::PointPtr &p, double time);
  [[nodiscard]] int GetId();
  void SetId(int id);
  [[nodiscard]] int GetCId();
  void SetCId(int Cid);
  [[nodiscard]] int GetNum();
  void SetNum(int num);
  [[nodiscard]] double GetRho();
  void SetRho(double rho);
  [[nodiscard]] double GetDelta();
  void SetDelta(double delta);
  [[nodiscard]] DPNodePtr &GetDep();
  void SetDep(DPNodePtr &dep);
  [[nodiscard]] PointPtr &GetCenter();
  void SetCenter(PointPtr &center);
  [[nodiscard]] double GetLastTime();
  void SetLastTime(double last_time);
  [[nodiscard]] bool IsActive();
  void SetActive(bool active);
  [[nodiscard]] std::unordered_set<SESAME::DPNodePtr> &GetSucs();
  void SetSucs(std::unordered_set<SESAME::DPNodePtr> &sucs);
  [[nodiscard]] ClusterPtr &GetCluster();
  void SetCluster(SESAME::ClusterPtr &cluster);
  [[nodiscard]] double GetInactiveTime();
  void SetInactiveTime(double inactive_time);
  [[nodiscard]] double GetDis();
  void SetDis(double dis);
  SESAME::DPNodePtr copy();

  void insert(double startTime);
  void add(double coef, double startTime);
  double getDisTo(SESAME::DPNodePtr &node);
  void removeSuccessor(SESAME::DPNodePtr &node);
  bool hasSuccessor();

  void addSuccessor(DPNodePtr &node);
};

} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_DPNODE_HPP_
