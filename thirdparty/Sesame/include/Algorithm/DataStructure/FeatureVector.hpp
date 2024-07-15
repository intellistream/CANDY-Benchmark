//
// Created by tuidan on 2021/8/24.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_FEATUREVECTOR_H_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_FEATUREVECTOR_H_

#include "Algorithm/DataStructure/GenericFactory.hpp"

#include <iostream>
#include <memory>
#include <vector>

namespace SESAME {
class CF;
typedef std::shared_ptr<CF> CFPtr;
class CF {
private:
  // N是子类中节点的数目，LS是N个节点的线性和，SS是N个节点的平方和
  int NumberOfNodes;
  int index;

public:
  std::vector<double> LS;
  std::vector<double> SS;
  CF();
  ~CF();
  int getN() const;
  void setN(int n);
  std::vector<double> getLS() const;
  std::vector<double> getSS() const;
  double getLSItem(int index) const;
  double getSSItem(int index) const;
  void setLS(std::vector<double> &newLs);
  void setSS(std::vector<double> &newSs);
  SESAME::CFPtr copy();
  int getIndex();
  void setIndex(int id);
};

template <typename T>
concept NodeConcept = requires(T t) {
  t->Centroid();
  t->cf.num;
  t->index;
  t->Update(GenericFactory::New<Point>(0));
  t->Scale(1.0);
};

template <NodeConcept T>
std::vector<std::vector<double>> CalcAdjMatrix(const std::vector<T> &nodes) {
  const int n = nodes.size();
  std::vector<std::vector<double>> adjMatrix(n, std::vector<double>(n, 0.0));
  std::vector<PointPtr> centroids(n);
  for (int i = 0; i < n; ++i) {
    centroids[i] = nodes[i]->Centroid();
  }
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      auto distance = centroids[i]->L1Dist(centroids[j]);
      adjMatrix[i][j] = distance, adjMatrix[j][i] = distance;
    }
  }
  return adjMatrix;
}

template <NodeConcept T>
auto CalcClosestNode(const std::vector<T> &nodes, PointPtr point) {
  double minDist = std::numeric_limits<double>::max();
  T node = nullptr;
  for (auto child : nodes) {
    auto centroid = child->Centroid();
    auto distance = centroid->L2Dist(point);
    if (distance < minDist) {
      minDist = distance;
      node = child;
    }
  }
  return std::make_pair(node, minDist);
}

template <NodeConcept T> double CalcClusterL1Dist(T a, T b) {
  auto ca = a->Centroid(), cb = b->Centroid();
  return ca->L1Dist(cb);
}

template <NodeConcept T> double CalcClusterL2Dist(T a, T b) {
  auto ca = a->Centroid(), cb = b->Centroid();
  return ca->L2Dist(cb);
}

struct ClusteringFeatures {
  // 原CF结构体，num是子类中节点的数目，LS是N个节点的线性和，SS是N个节点的平方和
  int num = 0;
  std::vector<double> ls, ss;
  ClusteringFeatures(int d = 0)
      : ls(std::vector<double>(d, 0.0)), ss(std::vector<double>(d, 0.0)) {}
};

} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_FEATUREVECTOR_H_
