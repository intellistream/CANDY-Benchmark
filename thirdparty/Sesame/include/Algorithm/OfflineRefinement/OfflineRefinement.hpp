// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 19/07/2021.
//
#ifndef SESAME_SRC_ALGORITHM_OFFLINE_REFINEMENT_HPP_
#define SESAME_SRC_ALGORITHM_OFFLINE_REFINEMENT_HPP_

#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/Param.hpp"
#include "Sinks/DataSink.hpp"

#include <vector>

namespace SESAME {
class OfflineRefinement {
public:
  void Run(SesameParam &param, const std::vector<PointPtr> &input,
           DataSinkPtr sinkPtr) {
    for (size_t i = 0; i < input.size(); ++i) {
      input[i]->setClusteringCenter(i);
      sinkPtr->put(input[i]);
    }
  }
};

class NoRefinement : public OfflineRefinement {
public:
  NoRefinement(const SesameParam &param) {}
};

} // namespace SESAME
#endif // SESAME_SRC_ALGORITHM_OFFLINE_CLUSTERING_HPP_