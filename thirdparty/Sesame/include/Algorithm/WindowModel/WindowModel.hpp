// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 26/07/2021.
//

#ifndef SESAME_SRC_ALGORITHM_WINDOWMODEL_WINDOW_HPP_
#define SESAME_SRC_ALGORITHM_WINDOWMODEL_WINDOW_HPP_

#include "Algorithm/DataStructure/FeatureVector.hpp"
#include "Algorithm/DataStructure/GenericFactory.hpp"
#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/Param.hpp"

#include <cmath>
#include <queue>

namespace SESAME {
class WindowModel;
typedef std::shared_ptr<WindowModel> WindowPtr;

class WindowModel {};

class Landmark : WindowModel {
private:
  int landmark_;

public:
  Landmark(const SesameParam &param) : landmark_(param.landmark) {}
  bool Add(PointPtr input) {
    return input->index == 0 || input->index % landmark_ != 0;
  }
};

class Sliding : WindowModel {
private:
  int sliding_;
  std::queue<PointPtr> queue_;

public:
  Sliding(const SesameParam &param) : sliding_(param.sliding) {}
  bool Add(const PointPtr input) {
    queue_.push(input);
    return true;
  }
  PointPtr Delete() {
    if (queue_.size() > sliding_) {
      auto ret = queue_.front();
      queue_.pop();
      return ret;
    }
    return nullptr;
  }
};

class Damped : WindowModel {
private:
  const double alpha_, lambda_;
  const int buf_size_;
  int cnt_ = 0;

public:
  Damped(const SesameParam &param)
      : alpha_(param.alpha), lambda_(param.lambda), buf_size_(param.buf_size) {}
  bool Add(const PointPtr input) {
    ++cnt_;
    return true;
  }
  bool Update() {
    if (cnt_ >= buf_size_) {
      cnt_ = 0;
      return true;
    }
    return false;
  }
  template <NodeConcept T> void Update(T node) {
    node->Scale(pow(alpha_, -lambda_));
  }
};

} // namespace SESAME

#endif // SESAME_SRC_ALGORITHM_WINDOWMODEL_WINDOW_HPP_
