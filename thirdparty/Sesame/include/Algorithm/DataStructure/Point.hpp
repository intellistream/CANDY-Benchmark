// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 19/07/2021.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_POINT_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_POINT_HPP_

#include "Utils/Types.hpp"

#include <iostream>
#include <memory>
#include <vector>

namespace SESAME {
struct Point;
typedef std::shared_ptr<Point> PointPtr;

struct Point {
  uint64 index;      // 1,2,3,4,5....
  fp64 weight = 1.0; // considering the outdated effect
  fp64 cost = 0.0;
  fp64 min_dist;
  fp64 knn = 0.0, conn = 1.0;
  bool outlier = false;
  int8 sgn = 1;      // the distance to the nearest data point
  int32 clu_id = -1; // using index to identify
  uint32 dim = 0;    // feature Length
  clock_t toa;       // time of arrival
  uint64 timestamp;  // the time stamp of the data point
  std::vector<feature_t> feature;
  Point(uint32 dim = 0, uint64 index = 0, feature_t *feature = nullptr);
  fp64 *data() { return feature.data(); }
  PointPtr copy();
  int getIndex() const;
  void setIndex(int index);
  double getWeight() const;
  void setWeight(double weight);
  double getFeatureItem(int index) const;
  void setFeatureItem(double feature, int index);
  int getClusteringCenter() const;
  void setClusteringCenter(int index);
  int getDimension() const;
  int getFeatureLength();
  double getMinDist() const;
  void setMinDist(double min_dist);
  bool getOutlier();
  void setOutlier(bool flag);
  double L2Dist(PointPtr centroid);
  double L1Dist(PointPtr centroid);
  PointPtr Reverse();
  std::string Serialize();
  void Debug();
};
} // namespace SESAME

#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_POINT_HPP_
