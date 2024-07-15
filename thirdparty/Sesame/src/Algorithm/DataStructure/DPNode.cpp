//
// Created by tuidan on 2021/9/9.
//

#include <Algorithm/DataStructure/DPNode.hpp>
#include <cassert>
#include <cfloat>
#include <cmath>
/**
 * DPNode
 */
static int id = 0;
SESAME::DPNode::DPNode() {
  this->cid = -1;
  this->Cid = -1;
  this->lastTime = 0;
  this->inactiveTime = 0;
  this->num = 0;
  this->active = false;
}

SESAME::DPNode::~DPNode() {}

int SESAME::DPNode::GetId() { return cid; }
void SESAME::DPNode::SetId(int id) { DPNode::cid = id; }
int SESAME::DPNode::GetCId() { return Cid; }
void SESAME::DPNode::SetCId(int Cid) { DPNode::Cid = Cid; }
int SESAME::DPNode::GetNum() { return num; }
void SESAME::DPNode::SetNum(int num) { DPNode::num = num; }
double SESAME::DPNode::GetRho() { return rho; }
void SESAME::DPNode::SetRho(double rho) { DPNode::rho = rho; }
double SESAME::DPNode::GetDelta() { return delta; }
void SESAME::DPNode::SetDelta(double delta) { DPNode::delta = delta; }
SESAME::DPNodePtr &SESAME::DPNode::GetDep() { return dep; }
void SESAME::DPNode::SetDep(SESAME::DPNodePtr &dep) { DPNode::dep = dep; }
SESAME::PointPtr &SESAME::DPNode::GetCenter() { return center; }
void SESAME::DPNode::SetCenter(SESAME::PointPtr &center) {
  DPNode::center = center;
}
double SESAME::DPNode::GetLastTime() { return lastTime; }
void SESAME::DPNode::SetLastTime(double last_time) { lastTime = last_time; }
bool SESAME::DPNode::IsActive() { return active; }
void SESAME::DPNode::SetActive(bool active) { DPNode::active = active; }
std::unordered_set<SESAME::DPNodePtr> &SESAME::DPNode::GetSucs() {
  return sucs;
}
void SESAME::DPNode::SetSucs(std::unordered_set<SESAME::DPNodePtr> &sucs) {
  DPNode::sucs = sucs;
}
SESAME::ClusterPtr &SESAME::DPNode::GetCluster() { return cluster; }
void SESAME::DPNode::SetCluster(SESAME::ClusterPtr &cluster) {
  DPNode::cluster = cluster;
}
double SESAME::DPNode::GetInactiveTime() { return inactiveTime; }
void SESAME::DPNode::SetInactiveTime(double inactive_time) {
  inactiveTime = inactive_time;
}
double SESAME::DPNode::GetDis() { return dis; }
void SESAME::DPNode::SetDis(double dis) { DPNode::dis = dis; }

SESAME::DPNode::DPNode(SESAME::PointPtr &p, double time) {
  this->cid = id++;
  this->rho = 1;
  this->delta = FLT_MAX;
  this->dep = nullptr;
  this->center = p->copy();
  this->lastTime = time;
  this->active = false;
  this->Cid = 0;
  this->inactiveTime = 0;
  this->dis = 0;
}

/**
 * Cluster
 */
int SESAME::Cluster::GetLabel() { return label; }
void SESAME::Cluster::SetLabel(int label) { Cluster::label = label; }
std::unordered_set<SESAME::DPNodePtr> &SESAME::Cluster::GetCells() {
  return cells;
}
void SESAME::Cluster::SetCells(std::unordered_set<DPNodePtr> &cells) {
  Cluster::cells = cells;
}
SESAME::Cluster::Cluster() { this->label = -1; }
SESAME::Cluster::Cluster(int label) { this->label = label; }
void SESAME::Cluster::add(DPNodePtr &node) {
  assert(node);
  this->cells.insert(node);
}
void SESAME::Cluster::remove(DPNodePtr &node) { this->cells.erase(node); }

void SESAME::DPNode::insert(double time) {
  this->rho++;
  this->lastTime = time;
}
void SESAME::DPNode::add(double coef, double time) {
  this->rho = coef * this->rho + 1;
  this->lastTime = time;
}
void SESAME::DPNode::addSuccessor(SESAME::DPNodePtr &node) {
  this->sucs.insert(node);
}
void SESAME::DPNode::removeSuccessor(SESAME::DPNodePtr &node) {
  this->sucs.erase(node);
}
bool SESAME::DPNode::hasSuccessor() { return !sucs.empty(); }
double SESAME::DPNode::getDisTo(SESAME::DPNodePtr &node) {
  double distance = 0;
  for (int i = 0; i < this->center->getDimension(); i++) {
    distance = distance + pow(node->center->getFeatureItem(i) -
                                  this->center->getFeatureItem(i),
                              2);
  }
  return sqrt(distance);
}
SESAME::DPNodePtr SESAME::DPNode::copy() {
  return std::make_shared<DPNode>(*this);
}
