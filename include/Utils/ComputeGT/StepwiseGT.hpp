#ifndef UTILS_COMPUTE_GT_STEPWISE_GT_HPP
#define UTILS_COMPUTE_GT_STEPWISE_GT_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <queue>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <mkl.h>

namespace COMPUTE_GT {

enum Metric {
  L2 = 0,
  INNER_PRODUCT = 1,
  COSINE = 2,
  FAST_L2 = 3
};

void computeL2sq(float* pointsL2sq, const float* matrix, const int64_t numPoints, const uint64_t dim);

void distSqToPoints(const size_t dim, float* distMatrix, size_t npoints, const float* points,
                      const float* pointsL2sq, size_t nqueries, const float* queries,
                      const float* queriesL2sq, float* onesVec = nullptr);

void innerProdToPoints(const size_t dim, float* distMatrix, size_t npoints, const float* points,
                          size_t nqueries, const float* queries, float* onesVec = nullptr);

void exactKnn(const size_t dim, const size_t k, size_t* closestPoints, float* distClosestPoints,
                size_t npoints, float* pointsIn, size_t nqueries, float* queriesIn,
                Metric metric);

template <typename T>
void loadBinAsFloat(const char* filename, float*& data, size_t& npts, size_t& ndims, int partNum);

void saveGTVectorsAsFile(const std::string& filename, int32_t* data, float* distances,
                          size_t npts, size_t ndims);

void stepwiseGTCalc(const std::string& baseFile, const std::string& queryFile,
                      const std::string& gtFile, size_t k, Metric metric,
                      size_t batchSize);
}

#endif // UTILS_COMPUTE_GT_STEPWISE_GT_HPP
