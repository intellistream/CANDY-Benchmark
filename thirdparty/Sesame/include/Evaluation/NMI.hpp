//
// Created by tuidan on 2023/1/3.
//

#ifndef SESAME_INCLUDE_EVALUATION_NMI_HPP_
#define SESAME_INCLUDE_EVALUATION_NMI_HPP_
#include <Algorithm/DataStructure/Point.hpp>
#include <vector>
namespace SESAME {

class NMI {
public:
  static double Evaluate(const std::vector<PointPtr> &inputs,
                         const std::vector<PointPtr> &predicts, int gt_size,
                         int predict_size);
};
} // namespace SESAME
#endif // SESAME_INCLUDE_EVALUATION_NMI_HPP_
