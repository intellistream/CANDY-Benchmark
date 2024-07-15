// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by wzru on 2022/3/13.
//

#ifndef SESAME_INCLUDE_ALGORITHM_PARAM_HPP_
#define SESAME_INCLUDE_ALGORITHM_PARAM_HPP_

#include "Utils/BenchmarkUtils.hpp"

namespace SESAME {

using StreamClusteringParam = param_t;

// class StreamClusteringParam {
// public:
//   // General Params
//   int num_points;
//   int dim;

//   // CF-Tree Params
//   int max_in_nodes;     // B
//   int max_leaf_nodes;         // L
//   double distance_threshold; // T

//   // KMeans / KMeans++ Params
//   int num_clusters;
//   bool KMeansPP;

//   // DBSCAN Params
//   unsigned int min_points; // minimum point of core point in DBSCAN
//   double epsilon; // maximum distance if point belongs to the density area of
//                   // core point

//   // Landmark Window
//   int landmark;
//   // Sliding Window
//   int sliding;
//   // Damped Window
//   double alpha, lambda;

//   double outlier_distance_threshold;
//   int outlier_cap;
//   double neighbor_distance, outlier_density_threshold;
// };

} // namespace SESAME

#endif