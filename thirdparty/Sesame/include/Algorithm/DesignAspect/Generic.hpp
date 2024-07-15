// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by wzru on 2022/3/13.
//

#ifndef SESAME_INCLUDE_ALGORITHM_GENERIC_HPP_
#define SESAME_INCLUDE_ALGORITHM_GENERIC_HPP_

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Algorithm/Algorithm.hpp"
#include "Algorithm/DataStructure/CoresetTree.hpp"
#include "Algorithm/DataStructure/GenericFactory.hpp"
#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/OutlierDetection/OutlierDetection.hpp"
#include "Algorithm/Param.hpp"
#include "Sinks/DataSink.hpp"
#include "Utils/BenchmarkUtils.hpp"

namespace SESAME {
template <typename W, typename D, typename O, typename R>
concept StreamClusteringConcept = requires {
  requires requires(W w, PointPtr p) {
    { w.Add(p) } -> std::same_as<bool>;
  };
  requires requires(D d, PointPtr p) {
    { d.Insert(p) } -> std::same_as<typename D::NodePtr>;
  };
  requires requires(O o, PointPtr p, typename D::NodePtr n,
                    std::vector<typename D::NodePtr> &vn) {
    { o.Check(p, vn) } -> std::same_as<bool>;
    { o.Check(n) } -> std::same_as<bool>;
  };
};

template <typename W, typename D, typename O, typename R>
  requires StreamClusteringConcept<W, D, O, R>
class StreamClustering : public Algorithm {
public:
  StreamClustering(const param_t &param);
  ~StreamClustering();
  void Init();
  void Insert(PointPtr);
  void RunOnline(PointPtr input);
  void RunOffline(DataSinkPtr ptr);
  void store(std::string output_file, int dim, std::vector<PointPtr> results);
  void OutputOnline(std::vector<PointPtr> &centers) override;

private:
  using Node = typename D::Node;
  using NodePtr = typename D::NodePtr;
  std::shared_ptr<W> w;
  std::shared_ptr<D> d;
  std::shared_ptr<O> o;
  std::shared_ptr<R> r;

  std::vector<NodePtr> outliers_;
  std::unordered_map<PointPtr, NodePtr> point_map_;
  std::unordered_map<NodePtr, std::unordered_set<PointPtr>> node_map_;
  std::vector<PointPtr> online_centers;
  size_t cluster_size_ = 0, outlier_size_ = 0;
  NodePtr InsertOutliers(PointPtr point);
};

template <typename W, typename D, typename O, typename R>
  requires StreamClusteringConcept<W, D, O, R>
StreamClustering<W, D, O, R>::~StreamClustering() {}

template <typename W, typename D, typename O, typename R>
  requires StreamClusteringConcept<W, D, O, R>
StreamClustering<W, D, O, R>::StreamClustering(const param_t &cmd_params) {
  param = cmd_params;
}

template <typename W, typename D, typename O, typename R>
  requires StreamClusteringConcept<W, D, O, R>
void StreamClustering<W, D, O, R>::Init() {
  w = GenericFactory::New<W>(param);
  d = GenericFactory::New<D>(param);
  o = GenericFactory::New<O>(param);
  r = GenericFactory::New<R>(param);
  d->Init();
  sum_timer.Tick();
}

template <typename W, typename D, typename O, typename R>
  requires StreamClusteringConcept<W, D, O, R>
void StreamClustering<W, D, O, R>::Insert(PointPtr p) {
  d->Insert(p);
}

template <typename W, typename D, typename O, typename R>
  requires StreamClusteringConcept<W, D, O, R>
void StreamClustering<W, D, O, R>::RunOnline(PointPtr input) {
  constexpr bool has_delete = requires(W &w) { w.Delete(); };
  constexpr bool has_update = requires(W &w, NodePtr node) { w.Update(node); };
  constexpr bool buffer_enabled = O::buffer_enabled;
  constexpr bool timer_enabled = O::timer_enabled;
  constexpr bool no_outlier_detection = std::is_same<O, NoDetection>::value;
  constexpr bool is_coreset_tree = std::is_same<D, CoresetTree>::value;
  if (w->Add(input)) {
    NodePtr node;
    out_timer.Tick();
    bool out = false;
    if constexpr (!no_outlier_detection) {
      out = o->Check(input, d->clusters());
    }
    out_timer.Tock();
    if (out) { // outlier
      out_timer.Tick();
      input->outlier = true;
      if constexpr (buffer_enabled) {
        node = InsertOutliers(input);
        out_timer.Tick();
        out = o->Check(node);
        out_timer.Tock();
        if (!out) {
          const auto [first, last] = std::ranges::remove_if(
              outliers_, [&](const NodePtr &n) { return n == node; });
          outliers_.erase(first, last);
          ds_timer.Tick();
          auto newNode = d->Insert(node);
          ds_timer.Tock();
          if constexpr (has_delete) {
            for (auto &p : node_map_[node]) {
              p->outlier = false;
              node_map_[newNode].insert(p);
              point_map_[p] = newNode;
            }
            node_map_.erase(node);
          }
          node = newNode;
        }
      }
      out_timer.Tock();
    } else {
      ds_timer.Tick();
      node = d->Insert(input);
      if (node)
        node->timestamp = input->index;
      ds_timer.Tock();
    }
    if constexpr (has_delete) {
      node_map_[node].insert(input);
      point_map_[input] = node;
    }
    if constexpr (!is_coreset_tree) {
      if (input->index % param.time_window == 0) {
        out_timer.Tick();
        auto &cls = d->clusters();
        std::vector<NodePtr> del;
        for (auto &node : cls) {
          if (o->TimerCheck(input, node)) {
            if constexpr (buffer_enabled) {
              outliers_.push_back(node);
            } else if constexpr (has_delete) {
              for (auto &p : node_map_[node]) {
                point_map_.erase(p);
              }
              node_map_.erase(node);
            }
            del.push_back(node);
          }
        }
        for (auto &node : del) {
          d->Remove(node);
        }
        out_timer.Tock();
      }
    }
  } else {
    if constexpr (is_coreset_tree) {
      out_timer.Tick();
      auto &cls = d->clusters();
      for (auto &node : cls) {
        if (o->TimerCheck(input, node)) {
          if constexpr (buffer_enabled) {
            outliers_.push_back(node);
          }
          d->Remove(node);
        }
      }
      if constexpr (timer_enabled && buffer_enabled) {
        std::vector<NodePtr> del;
        for (auto &out : outliers_) {
          if (o->TimerCheck(input, out)) {
            del.push_back(out);
          }
        }
        for (auto &out : del) {
          std::ranges::remove_if(outliers_,
                                 [&](const NodePtr &n) { return n == out; });
        }
      }
      out_timer.Tock();
    }
    win_timer.Tick();
    OutputOnline(online_centers);
    d = GenericFactory::New<D>(param);
    d->Init();
    outliers_ = {};
    win_timer.Tock();
  }
  if constexpr (has_update) {
    win_timer.Tick();
    if (w->Update()) {
      d->ForEach([&](NodePtr node) { w->Update(node); });
      if constexpr (buffer_enabled) {
        std::ranges::for_each(outliers_,
                              [&](NodePtr node) { w->Update(node); });
      }
    }
    win_timer.Tock();
  }
  if constexpr (has_delete) {
    win_timer.Tick();
    PointPtr point = w->Delete();
    if (point != nullptr) {
      assert(point_map_.contains(point));
      auto node = point_map_[point];
      if (node != nullptr) {
        node->Update(point->Reverse(), true);
        point_map_.erase(point);
        node_map_[node].erase(point);
        if constexpr (buffer_enabled) {
          if (node->cf.num == 0) {
            // TODO
            const auto [first, last] = std::ranges::remove_if(
                outliers_, [&](const NodePtr &n) { return n->cf.num == 0; });
            outliers_.erase(first, last);
          }
        }
      }
    }
    win_timer.Tock();
  }
  lat_timer.Add(input->toa);
}

template <typename W, typename D, typename O, typename R>
  requires StreamClusteringConcept<W, D, O, R>
void StreamClustering<W, D, O, R>::RunOffline(DataSinkPtr ptr) {
  on_timer.Add(sum_timer.start);
  ref_timer.Tick();
  OutputOnline(online_centers);
  std::cout << "cluster_size: " << cluster_size_ << std::endl;
  std::cout << "outlier_size: " << outlier_size_ << std::endl;
  r->Run(param, online_centers, ptr);
  ref_timer.Tock();
  sum_timer.Tock();
}

template <typename W, typename D, typename O, typename R>
  requires StreamClusteringConcept<W, D, O, R>
StreamClustering<W, D, O, R>::NodePtr
StreamClustering<W, D, O, R>::InsertOutliers(PointPtr point) {
  if (outliers_.empty()) {
    auto node = GenericFactory::New<Node>(d, point);
    node->index = 0;
    node->timestamp = point->index;
    outliers_.push_back(node);
    return node;
  } else {
    auto closest = CalcClosestNode(outliers_, point);
    if (closest.second < param.distance_threshold) {
      closest.first->Update(point);
      closest.first->timestamp = point->index;
      return closest.first;
    } else {
      auto node = GenericFactory::New<Node>(d, point);
      node->index = outliers_.size();
      outliers_.push_back(node);
      node->timestamp = point->index;
      return node;
    }
  }
}

template <typename W, typename D, typename O, typename R>
  requires StreamClusteringConcept<W, D, O, R>
void StreamClustering<W, D, O, R>::OutputOnline(
    std::vector<PointPtr> &centers) {
  // std::cerr << "Generic OutputOnline: " << d->clusters().size() << std::endl;
  auto clusters = d->clusters();
  cluster_size_ += clusters.size();
  outlier_size_ += outliers_.size();
  for (int i = 0; i < clusters.size(); i++) {
    auto centroid = GenericFactory::New<Point>(param.dim, i);
    for (int j = 0; j < param.dim; j++) {
      centroid->feature[j] = clusters[i]->cf.ls[j] / clusters[i]->cf.num;
    }
    centers.push_back(centroid);
  }
  for (int i = 0; i < outliers_.size(); ++i) {
    auto centroid = GenericFactory::New<Point>(param.dim, i);
    for (int j = 0; j < param.dim; j++) {
      centroid->feature[j] = outliers_[i]->cf.ls[j] / outliers_[i]->cf.num;
    }
    centers.push_back(centroid);
  }
}

} // namespace SESAME

#endif // SESAME_INCLUDE_ALGORITHM_ALGORITHM_HPP_
