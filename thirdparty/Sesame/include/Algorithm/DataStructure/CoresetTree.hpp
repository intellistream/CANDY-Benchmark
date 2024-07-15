// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 19/07/2021.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_CORESETTREE_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_CORESETTREE_HPP_

#include "Algorithm/DataStructure/FeatureVector.hpp"
#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/Param.hpp"
#include "Utils/Random.hpp"

#include <memory>
#include <vector>

namespace SESAME {

class CoresetTree : public std::enable_shared_from_this<CoresetTree> {
public:
  struct Node;
  struct Bucket;
  using NodePtr = std::shared_ptr<Node>;
  using TreePtr = std::shared_ptr<CoresetTree>;
  using Points = std::shared_ptr<std::vector<PointPtr>>;

private:
  const SesameParam &param;
  std::vector<PointPtr> samples;
  bool has_sampled = false;
  Random r;
  NodePtr root = nullptr;
  std::vector<Bucket> buckets;
  size_t num_buckets = 0;
  std::vector<PointPtr> centers;
  std::vector<NodePtr> clusters_;
  std::vector<PointPtr> Union(const std::vector<PointPtr> &a,
                              const std::vector<PointPtr> &b);
  NodePtr Select(NodePtr);
  PointPtr ChooseCenter(NodePtr);
  void Split(NodePtr, PointPtr, int);
  std::vector<NodePtr> Points2Nodes(Points);

public:
  CoresetTree(const SesameParam &param);
  void Init();
  NodePtr Insert(PointPtr input);
  NodePtr Insert(NodePtr node);
  void Remove(NodePtr node);
  std::vector<NodePtr> &clusters();

public:
  struct Bucket {
    Points base, spill;
    Bucket()
        : base(std::make_shared<std::vector<PointPtr>>()),
          spill(std::make_shared<std::vector<PointPtr>>()) {}
  };
  struct Node : std::enable_shared_from_this<Node> {
    size_t timestamp = 0;
    int index = 0;
    const int dim;
    ClusteringFeatures cf;
    double costs_sum_dist = 0.0, costs_sum_sq_dist = 0.0;
    NodePtr lc = nullptr, rc = nullptr, parent = nullptr;
    PointPtr center;
    std::vector<PointPtr> points;
    Node(TreePtr s, PointPtr p)
        : dim(p->getDimension()), cf(p->getDimension()) {
      Update(p);
    }
    Node(PointPtr p) : dim(p->getDimension()), cf(p->dim), center(p) {}
    PointPtr Centroid() {
      auto c = GenericFactory::New<Point>(dim);
      for (int i = 0; i < dim; ++i)
        c->setFeatureItem(cf.ls[i] / cf.num, i);
      return c;
    }
    PointPtr Center() { return center; }
    void Update(PointPtr point) {
      cf.num += point->sgn;
      double d = point->L2Dist(Centroid());
      costs_sum_dist += d * point->sgn;
      costs_sum_sq_dist += d * d * point->sgn;
      for (int i = 0; i < dim; ++i) {
        auto val = point->getFeatureItem(i);
        cf.ls[i] += val * point->sgn;
        cf.ss[i] += (val * val) * point->sgn;
      }
      points.push_back(point);
    }
    void Scale(double scale) {
      costs_sum_dist *= scale;
      costs_sum_sq_dist *= scale * scale;
      for (int i = 0; i < dim; ++i) {
        cf.ls[i] *= scale;
        cf.ss[i] *= scale * scale;
      }
    }
    bool IsLeaf() { return lc == nullptr && rc == nullptr; }
  };
};

} // namespace SESAME

#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_CORESETTREE_HPP_
