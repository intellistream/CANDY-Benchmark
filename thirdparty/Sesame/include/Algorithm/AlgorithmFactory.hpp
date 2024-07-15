// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 26/07/2021.
//

#ifndef SESAME_SRC_ALGORITHM_ALGORITHMFACTORY_HPP_
#define SESAME_SRC_ALGORITHM_ALGORITHMFACTORY_HPP_

#include <Algorithm/Algorithm.hpp>
#include <Utils/BenchmarkUtils.hpp>

namespace SESAME {
class AlgorithmFactory {
public:
  static SESAME::AlgorithmPtr create(param_t &cmd_params);
};
} // namespace SESAME
#endif // SESAME_SRC_ALGORITHM_ALGORITHMFACTORY_HPP_
