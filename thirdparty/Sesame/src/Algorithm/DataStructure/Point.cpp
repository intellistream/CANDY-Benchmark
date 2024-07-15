// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 19/07/2021.
//

#include "Algorithm/DataStructure/Point.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace SESAME {

Point::Point(uint32 dim, uint64 index, feature_t *feature)
    : dim(dim), index(index), feature((dim + 3) / 4 * 4, 0.0) {
  if (feature != nullptr) {
    memcpy(this->feature.data(), feature, dim * sizeof(feature_t));
  }
}

int Point::getIndex() const { return this->index; }

void Point::setIndex(int index) { this->index = index; }

double Point::getWeight() const { return this->weight; }

void Point::setWeight(double weight) { this->weight = weight; }

double Point::getFeatureItem(int index) const { return this->feature[index]; }

void Point::setFeatureItem(double feature, int index) {
  this->feature[index] = feature;
}

int Point::getClusteringCenter() const { return this->clu_id; }

void Point::setClusteringCenter(int index) { this->clu_id = index; }

/**
 * @param source
 */
PointPtr Point::copy() { return std::make_shared<Point>(*this); }

int Point::getDimension() const { return this->dim; }

int Point::getFeatureLength() { return (int)this->feature.size(); }

double Point::getMinDist() const { return min_dist; }

void Point::setMinDist(double min_dist) { min_dist = min_dist; }

double Point::L1Dist(PointPtr centroid) {
  static const auto mask = _mm256_set1_pd(0x7fffffff);
  double sum = 0;
  const int dim = getDimension();
  auto a = feature.data(), b = centroid->feature.data();
  auto sum_v = _mm256_setzero_pd();
  for (size_t i = 0; i < dim; i += 4) {
    __m256d diff =
        _mm256_sub_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i));
    auto abs_v = _mm256_and_pd(diff, mask);
    sum_v = _mm256_add_pd(sum_v, abs_v);
  }
  double v[4];
  _mm256_storeu_pd(v, sum_v);
  sum += v[0] + v[1] + v[2] + v[3];
  return sum;
}

double Point::L2Dist(PointPtr centroid) {
  double sum = 0.0;
  const int dim = getDimension();
  auto a = feature.data(), b = centroid->feature.data();
  auto sum_v = _mm256_setzero_pd();
  for (size_t i = 0; i < dim; i += 4) {
    __m256d diff =
        _mm256_sub_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i));
    __m256d square = _mm256_mul_pd(diff, diff);
    sum_v = _mm256_add_pd(sum_v, square);
  }
  double v[4];
  _mm256_storeu_pd(v, sum_v);
  sum += v[0] + v[1] + v[2] + v[3];
  //   for (int i = 0; i < dim; i++) {
  // #ifndef NDEBUG
  //     assert(std::isnan(centroid->getFeatureItem(i)) == false);
  //     assert(std::isnan(getFeatureItem(i)) == false);
  // #endif
  //     auto val = a[i] - b[i];
  //     sum += val * val;
  //   }
  return sqrt(sum);
}

void SESAME::Point::setOutlier(bool flag) { this->outlier = flag; }

bool SESAME::Point::getOutlier() { return this->outlier; }

PointPtr Point::Reverse() {
  auto res = copy();
  res->sgn = -res->sgn;
  return res;
}

std::string Point::Serialize() {
  std::string str = "#" + std::to_string(index) + " " + std::to_string(dim);
  for (int i = 0; i < dim; i++) {
    str += "," + std::to_string(feature.at(i));
  }
  return str;
}
void Point::Debug() { std::cerr << Serialize() << std::endl; }

} // namespace SESAME
