//
// Created by tuidan on 2021/10/11.
//
#include <Evaluation/Euclidean.hpp>
#include <Utils/Logger.hpp>
#include <cmath>

double getDistance(SESAME::PointPtr inputs, SESAME::PointPtr centres, int dim) {
  int i;
  double f = 0.0;
  for (i = 0; i < dim; i++) {
    f += pow((inputs->getFeatureItem(i) - centres->getFeatureItem(i)), 2);
  }
  return f;
}

void SESAME::Euclidean::euclideanCost(int numberOfPoints, int numberOfCenters,
                                      int dimention,
                                      const std::vector<PointPtr> &inputs,
                                      const std::vector<PointPtr> &results) {
  double mincost = 0.0;
  double kmeansoverall = 0.0;
  double distance = 0.0;
  for (int i = 0; i < numberOfPoints; i++) {
    mincost = getDistance(inputs[i], results[0], dimention);
    for (int j = 0; j < numberOfCenters;
         j++) { // looking for the nearest center
      distance = getDistance(inputs[i], results[j], dimention); //
      if (distance < mincost)
        mincost = distance;
    }
  }
  kmeansoverall += mincost;
  SESAME_DEBUG("EuclideanCost:" << kmeansoverall);
} // EuclideanCost, defined in StreamKM++: A clustering algorithm for data
  // streams