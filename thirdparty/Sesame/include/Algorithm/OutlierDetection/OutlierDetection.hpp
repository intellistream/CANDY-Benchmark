//
// Created by 1124a on 2021/11/2.
//

#ifndef SESAME_INCLUDE_ALGORITHM_OUTLIERDETECTION_OUTLIERDETECTION_HPP_
#define SESAME_INCLUDE_ALGORITHM_OUTLIERDETECTION_OUTLIERDETECTION_HPP_

#include <memory>

#include "Algorithm/DataStructure/CFTree.hpp"
#include "Algorithm/Param.hpp"

namespace SESAME {

template <bool B, bool T> class OutlierDetection {
  const int outlier_cap_;
  const size_t interval_;

public:
  static constexpr bool buffer_enabled = B;
  static constexpr bool timer_enabled = T;

  OutlierDetection(const SesameParam &param)
      : outlier_cap_(param.outlier_cap), interval_(param.clean_interval) {}
  template <NodeConcept N> bool Check(PointPtr point, std::vector<N> &nodes) {
    return false;
  }
  template <NodeConcept N> bool Check(N node) {
    if (node != nullptr)
      return node->cf.num < outlier_cap_;
    return false;
  }
  template <NodeConcept N> bool TimerCheck(PointPtr input, N node) {
    if (node == nullptr)
      return false;
    bool is_outlier = node->cf.num < outlier_cap_;
    if constexpr (timer_enabled) {
      if (input->index - node->timestamp < interval_)
        is_outlier = false;
    }
    return is_outlier;
  }
};

template <bool B, bool T> class DistanceDetection {
private:
  const double outlier_distance_threshold_;
  const int outlier_cap_;
  const size_t interval_;
  size_t cnt_ = 0;

public:
  static constexpr bool buffer_enabled = B;
  static constexpr bool timer_enabled = T;

  DistanceDetection(const SesameParam &param)
      : outlier_distance_threshold_(param.outlier_distance_threshold),
        outlier_cap_(param.outlier_cap), interval_(param.time_interval) {}
  template <NodeConcept N> bool Check(PointPtr point, std::vector<N> &nodes) {
    if (nodes.empty())
      return false;
    auto dist = CalcClosestNode(nodes, point).second;
    return dist > outlier_distance_threshold_;
  }
  template <NodeConcept N> bool Check(N node) {
    if (node != nullptr)
      return node->cf.num < outlier_cap_;
    return false;
  }
  template <NodeConcept N> bool TimerCheck(PointPtr input, N node) {
    if (node == nullptr)
      return false;
    bool is_outlier = node->cf.num < outlier_cap_;
    if constexpr (timer_enabled) {
      if (input->index - node->timestamp < interval_)
        is_outlier = false;
    }
    return is_outlier;
  }
};

template <bool B, bool T> class DensityDetection {
  const double neighbor_distance_, outlier_density_threshold_;
  const int outlier_cap_;
  const size_t interval_;
  size_t cnt_ = 0;

public:
  static constexpr bool buffer_enabled = B;
  static constexpr bool timer_enabled = T;

  DensityDetection(const SesameParam &param)
      : neighbor_distance_(param.neighbor_distance),
        outlier_density_threshold_(param.outlier_density_threshold),
        outlier_cap_(param.outlier_cap), interval_(param.time_interval) {}
  template <NodeConcept N> bool Check(PointPtr point, std::vector<N> &nodes) {
    std::vector<N> neighborNodes;
    int neighborDensity = 0, neighborNeighborDensity = 0;
    for (auto node : nodes) {
      auto dist = point->L2Dist(node->Centroid());
      if (dist < neighbor_distance_) {
        neighborNodes.push_back(node);
        neighborDensity += node->cf.num;
      }
    }
    for (auto neighbor : neighborNodes) {
      for (auto node : nodes) {
        if (CalcClusterL2Dist(node, neighbor) < neighbor_distance_) {
          neighborNeighborDensity += node->cf.num;
        }
      }
    }
    if (neighborNeighborDensity == 0)
      return false;
    else
      return (double)neighborDensity / neighborNeighborDensity <
             outlier_density_threshold_;
  }
  template <NodeConcept N> bool Check(N node) {
    if (node != nullptr)
      return node->cf.num < outlier_cap_;
    return false;
  }
  template <NodeConcept N> bool TimerCheck(PointPtr input, N node) {
    if (node == nullptr)
      return false;
    bool is_outlier = node->cf.num < outlier_cap_;
    if constexpr (timer_enabled) {
      if (input->index - node->timestamp < interval_)
        is_outlier = false;
    }
    return is_outlier;
  }
};

class NoDetection {
public:
  static constexpr bool buffer_enabled = false;
  static constexpr bool timer_enabled = false;
  NoDetection(const SesameParam &param) {}
  template <NodeConcept N> bool Check(PointPtr point, std::vector<N> &nodes) {
    return false;
  }
  template <NodeConcept N> bool Check(N node) { return false; }
  template <NodeConcept N> bool TimerCheck(PointPtr input, N node) {
    return false;
  }
};

} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_OUTLIERDETECTION_OUTLIERDETECTION_HPP_
