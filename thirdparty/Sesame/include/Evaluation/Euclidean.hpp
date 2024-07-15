//
// Created by tuidan on 2021/10/11.
//

#ifndef SESAME_INCLUDE_EVALUATION_EUCLIDEAN_HPP_
#define SESAME_INCLUDE_EVALUATION_EUCLIDEAN_HPP_
#include <Algorithm/DataStructure/Point.hpp>
#include <vector>
namespace SESAME {

class Euclidean {
public:
  static void
  euclideanCost(int numberOfPoints, int numberOfCenters, int dimention,
                const std::vector<PointPtr> &inputs,
                const std::vector<PointPtr>
                    &results); // EuclideanCost, defined in StreamKM++: A
                               // clustering algorithm for data streams
};

} // namespace SESAME
#endif // SESAME_INCLUDE_EVALUATION_EUCLIDEAN_HPP_
