// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 19/07/2021.
//

#include "Algorithm/DataStructure/CoresetTree.hpp"

#include <algorithm>
#include <cmath>

using namespace std;
using namespace SESAME;

CoresetTree::CoresetTree(const param_t &param)
    : param(param), r(param.seed),
      num_buckets(
          log((double)param.num_points / param.coreset_size) / log(2.0) + 2),
      buckets(log((double)param.num_points / param.coreset_size) / log(2.0) +
              2) {}

void CoresetTree::Init() {}

vector<PointPtr> CoresetTree::Union(const vector<PointPtr> &a,
                                    const vector<PointPtr> &b) {
  auto n = a.size() + b.size();
  vector<PointPtr> c(n, GenericFactory::New<Point>(param.dim));
  int choosen = 0;
  int j = r.random_uniform(0, n - 1);
  if (j < a.size()) {
    c[choosen] = a[j];
  } else {
    j = j - a.size();
    c[choosen] = b[j];
  }
  root = GenericFactory::New<Node>(c[choosen++]);
  // clusters_ = {root};
  for (auto &i : a) {
    root->Update(i);
  }
  for (auto &i : b) {
    root->Update(i);
  }
  while (choosen < n) {
    if (root->costs_sum_sq_dist > 0.0) {
      auto leaf = Select(root);
      auto center = ChooseCenter(leaf);
      Split(leaf, center, choosen);
      c[choosen] = center;
    } else {
      c[choosen] = root->center->copy();
      c[choosen]->index = -1;
    }
    ++choosen;
  }
  return c;
}

CoresetTree::NodePtr CoresetTree::Select(NodePtr node) {
  while (!node->IsLeaf()) {
    if (node->lc->costs_sum_sq_dist == 0.0 &&
        node->rc->costs_sum_sq_dist == 0.0) {
      if (node->lc->cf.num == 0) {
        node = node->rc;
      } else if (node->rc->cf.num == 0) {
        node = node->lc;
      } else {
        node = r.bernoulli(0.5) ? node->lc : node->rc;
      }
    } else {
      node = r.bernoulli(node->lc->costs_sum_sq_dist / node->costs_sum_sq_dist)
                 ? node->lc
                 : node->rc;
    }
  }
  return node;
}

PointPtr CoresetTree::ChooseCenter(CoresetTree::NodePtr node) {
  const int times = 3;
  double min_cost = node->costs_sum_sq_dist;
  PointPtr best_center = node->points[0];
  PointPtr centroid = node->Centroid();
  for (int j = 0; j < times; ++j) {
    double sum = 0.0;
    double random = r.random_uniform(0.0, 1.0);
    for (auto &p : node->points) {
      double dist = p->L2Dist(node->center);
      sum += dist / node->costs_sum_sq_dist;
      if (sum >= random) {
        double cost = min(centroid->L2Dist(node->center), centroid->L2Dist(p));
        if (cost < min_cost) {
          min_cost = cost;
          best_center = p;
        }
        break;
      }
    }
  }
  return best_center;
}

void CoresetTree::Split(CoresetTree::NodePtr parent, PointPtr center,
                        int index) {
  NodePtr lc = GenericFactory::New<Node>(center),
          rc = GenericFactory::New<Node>(center);
  // clusters_.push_back(lc), clusters_.push_back(rc);
  lc->center = parent->center;
  lc->parent = parent;
  rc->center = parent->center;
  rc->parent = parent;
  parent->lc = lc, parent->rc = rc;
  for (auto &p : parent->points) {
    p->clu_id = index;
    if (p->L2Dist(center) < p->L2Dist(parent->center)) {
      rc->Update(p);
    } else {
      lc->Update(p);
    }
  }
  while (parent != nullptr) {
    parent->costs_sum_sq_dist =
        parent->lc->costs_sum_sq_dist + parent->rc->costs_sum_sq_dist;
    parent->costs_sum_dist =
        parent->lc->costs_sum_dist + parent->rc->costs_sum_dist;
    parent = parent->parent;
  }
}

CoresetTree::NodePtr CoresetTree::Insert(PointPtr input) {
  buckets[0].base->push_back(input);
  if (buckets[0].base->size() == param.coreset_size) {
    int cur = 0, next = 1;
    if (buckets[next].base->size() == 0) {
      buckets[next].base = buckets[cur].base;
      buckets[cur].base = GenericFactory::New<vector<PointPtr>>();
    } else {
      buckets[next].spill = buckets[cur].base;
      buckets[cur].base = GenericFactory::New<vector<PointPtr>>();
      ++cur, ++next;
      while (buckets[next].base->size() == param.coreset_size) {
        *buckets[next].spill.get() =
            Union(*buckets[cur].base.get(), *buckets[cur].spill.get());
        ++cur, ++next;
      }
      *buckets[next].spill.get() =
          Union(*buckets[cur].base.get(), *buckets[cur].spill.get());
    }
  }
  return root;
}

CoresetTree::NodePtr CoresetTree::Insert(NodePtr node) {
  clusters_.push_back(node);
  return node;
}

void CoresetTree::Remove(NodePtr node) {
  auto it = find(clusters_.begin(), clusters_.end(), node);
  if (it != clusters_.end()) {
    clusters_.erase(it);
  }
}

vector<CoresetTree::NodePtr> &CoresetTree::clusters() {
  if (!clusters_.empty())
    return clusters_;
  Points points = nullptr;
  if (buckets[num_buckets - 1].base->size() == param.coreset_size) {
    points = buckets[num_buckets - 1].base;
  } else {
    int i = 0;
    for (; i < num_buckets; ++i) {
      if (buckets[i].base->size() == param.coreset_size) {
        points = buckets[i].base;
        break;
      }
    }
    for (int j = i + 1; j < num_buckets; ++j) {
      if (buckets[j].base->size() != 0) {
        *buckets[j].spill.get() = Union(*buckets[j].base.get(), *points.get());
        points = buckets[j].spill;
      }
    }
  }
  return clusters_ = Points2Nodes(points);
}

vector<CoresetTree::NodePtr>
CoresetTree::Points2Nodes(CoresetTree::Points points) {
  vector<CoresetTree::NodePtr> nodes;
  if (points == nullptr) {
    return nodes;
  }
  for (auto &p : *points.get()) {
    auto node = GenericFactory::New<Node>(p);
    node->Update(p);
    nodes.push_back(node);
  }
  return nodes;
}
