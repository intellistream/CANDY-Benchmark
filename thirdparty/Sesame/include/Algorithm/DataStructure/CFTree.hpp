//
// Created by tuidan on 2021/8/24.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_CFTREE_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_CFTREE_HPP_

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <ranges>
#include <vector>

#include "Algorithm/DataStructure/FeatureVector.hpp"
#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/Param.hpp"

namespace SESAME {
// define the share point of the class object
class CFNode;
class CFTree;
typedef std::shared_ptr<CFNode> NodePtr;
typedef std::shared_ptr<CFTree> CFTreePtr;

class CFTree {
private:
  int max_in_nodes;   // max CF number of each internal node
  int max_leaf_nodes; // max CF number of each leaf node
  double
      distance_threshold; // threshold radius of each sub cluster in leaf nodes
public:
  CFTree(const SesameParam &param);
  CFTree(int b, int l, double t);
  ~CFTree();
  int getB() const;
  int getL() const;
  double getT() const;
  void setB(int b);
  void setL(int l);
  void setT(double t);
};

class CFNode {
private:
  CFPtr curCF;
  bool isLeaf;
  bool outlier;
  std::vector<NodePtr> children;
  NodePtr parent;
  int index;

public:
  CFNode();
  ~CFNode();
  CFPtr getCF();
  void setCF(CFPtr &cf);
  NodePtr getParent();
  int getIndex() const;
  std::vector<NodePtr> getChildren();
  void removeChild(NodePtr &child);
  NodePtr copy();
  bool getIsLeaf();
  void setIsLeaf(bool leaf);
  void setNode(CFPtr &Node);
  void setIndex(int Index);
  void setParent(NodePtr &Parent);
  void setChild(NodePtr &child);
  void setChildren(std::vector<NodePtr> children);
  void clearParents();
  void setOutlier(bool flag);
  bool getOutlier();
  std::string Prefix(int d) {
    std::string prefix = "";
    while (d--) {
      prefix += "- ";
    }
    return prefix;
  }
  std::string Serialize(int d = 0) {
    std::string str = Prefix(d) + std::to_string(index) + ":" +
                      std::to_string(curCF->getN()) + "\n";
    return str;
  }
};

class ClusteringFeaturesTree
    : public std::enable_shared_from_this<ClusteringFeaturesTree> {
private:
  const int max_in_nodes;   // max CF number of each internal node
  const int max_leaf_nodes; // max CF number of each leaf node
  const double
      distance_threshold; // threshold radius of each sub cluster in leaf nodes
  const int dim;
  int leafMask = 0;

public:
  struct Node;
  using NodePtr = std::shared_ptr<Node>;
  using TreePtr = std::shared_ptr<ClusteringFeaturesTree>;
  ClusteringFeaturesTree(const SesameParam &param);
  ~ClusteringFeaturesTree();
  void Init();
  NodePtr Insert(PointPtr point);
  NodePtr Insert(NodePtr node);
  void Remove(NodePtr node);
  void ForEach(std::function<void(NodePtr)> func);
  std::string Serialize();
  std::vector<NodePtr> &clusters();
  NodePtr root() { return root_; }

private:
  template <typename T> NodePtr backwardEvolution(NodePtr node, T point);
  NodePtr root_;
  std::vector<NodePtr> clusters_;

public:
  struct Node : std::enable_shared_from_this<Node> {
    size_t timestamp = 0;
    NodePtr parent = nullptr;
    std::vector<NodePtr> children;
    int index = 0;
    const int dim;
    ClusteringFeatures cf;
    TreePtr tree;

    Node(TreePtr tree, int d = 0) : tree(tree), dim(d), cf(d) {}
    Node(TreePtr tree, PointPtr p) : Node(tree, p->getDimension()) {
      Update(p);
    }
    ~Node() = default;
    void RemoveChild(NodePtr child) {
      const auto [first, last] =
          std::ranges::remove_if(children, [&](auto &c) { return c == child; });
      children.erase(first, last);
    }
    bool IsLeaf() const { return children.empty(); }
    void AddChild(NodePtr child) {
      children.push_back(child);
      child->parent = shared_from_this();
    }
    void ClearParents() {
      if (parent != nullptr)
        parent->index = -1;
    }
    void Update(PointPtr point) {
      cf.num += point->sgn;
      for (int i = 0; i < dim; ++i) {
        auto val = point->getFeatureItem(i);
        cf.ls[i] += val * point->sgn;
        cf.ss[i] += (val * val) * point->sgn;
      }
      if (cf.num == 0) {
        if (tree != nullptr)
          tree->Remove(shared_from_this());
      }
    }
    void Update(NodePtr node) {
      cf.num += node->cf.num;
      for (int i = 0; i < dim; ++i) {
        cf.ls[i] += node->cf.ls[i];
        cf.ss[i] += node->cf.ss[i] * node->cf.ss[i];
      }
    }
    void Scale(double scale) {
      for (int i = 0; i < dim; ++i) {
        cf.ls[i] *= scale;
        cf.ss[i] *= scale * scale;
      }
      // auto a = cf.ls.data(), b = cf.ss.data();
      // auto factor1 = _mm256_set1_pd(scale), factor2 = _mm256_set1_pd(scale *
      // scale); for (size_t i = 0; i < dim; i += 4) {
      //   _mm256_mul_pd(_mm256_loadu_pd(a + i), factor1);
      //   _mm256_mul_pd(_mm256_loadu_pd(b + i), factor2);
      // }
    }
    template <typename T> void Update(T point, bool all) {
      Update(point);
      if (parent != nullptr && all) {
        parent->Update(point, all);
      }
    }
    PointPtr Centroid() {
      assert(cf.num);
      auto c = GenericFactory::New<Point>(dim);
      c->setIndex(-1);
      c->setClusteringCenter(-1);
      for (int i = 0; i < dim; ++i)
        c->setFeatureItem(cf.ls[i] / cf.num, i);
      return c;
    }
    std::string Prefix(int d) {
      std::string prefix = "";
      while (d--) {
        prefix += "- ";
      }
      return prefix;
    }
    std::string Serialize(int d = 0) {
      std::string str = Prefix(d) + std::to_string(index) + ":" +
                        std::to_string(cf.num) + "\n";
      return str;
    }
  };
};

class ClusteringFeaturesList {
private:
  const int dim;
  const double distance_threshold;

public:
  struct Node;
  using NodePtr = std::shared_ptr<Node>;
  using ListPtr = std::shared_ptr<ClusteringFeaturesList>;
  ClusteringFeaturesList(const SesameParam &param);
  ~ClusteringFeaturesList();
  NodePtr Insert(PointPtr point);
  NodePtr Insert(NodePtr node);
  void ForEach(std::function<void(NodePtr)> func) {
    std::ranges::for_each(clusters_, func);
  }
  void Init() {}
  std::vector<NodePtr> &clusters();
  void Remove(NodePtr node);

private:
  std::vector<NodePtr> clusters_;

public:
  struct Node : std::enable_shared_from_this<Node> {
    size_t timestamp = 0;
    int index = 0;
    const int dim;
    ClusteringFeatures cf;

    Node(int d = 0) : dim(d), cf(d) {}
    Node(PointPtr p) : Node(p->getDimension()) { Update(p); }
    Node(ListPtr l, PointPtr p) : Node(p) {}
    ~Node() = default;
    void Update(PointPtr point) {
      cf.num += point->sgn;
      for (int i = 0; i < dim; ++i) {
        auto val = point->getFeatureItem(i);
        cf.ls[i] += val;
        cf.ss[i] += val * val;
      }
    }
    void Update(NodePtr node) {
      cf.num += node->cf.num;
      for (int i = 0; i < dim; ++i) {
        cf.ls[i] += node->cf.ls[i];
        cf.ss[i] += node->cf.ss[i] * node->cf.ss[i];
      }
    }
    template <typename T> void Update(T point, bool all) { Update(point); }
    void Scale(double scale) {
      for (int i = 0; i < dim; ++i) {
        cf.ls[i] *= scale;
        cf.ss[i] *= scale * scale;
      }
      // auto a = cf.ls.data(), b = cf.ss.data();
      // auto factor1 = _mm256_set1_pd(scale), factor2 = _mm256_set1_pd(scale *
      // scale); for (size_t i = 0; i < dim; i += 4) {
      //   _mm256_mul_pd(_mm256_loadu_pd(a + i), factor1);
      //   _mm256_mul_pd(_mm256_loadu_pd(b + i), factor2);
      // }
    }
    PointPtr Centroid() {
      auto c = GenericFactory::New<Point>(dim);
      c->setIndex(-1);
      c->setClusteringCenter(-1);
      for (int i = 0; i < dim; ++i)
        c->setFeatureItem(cf.ls[i] / cf.num, i);
      return c;
    }
  };
};

} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_CFTREE_HPP_
