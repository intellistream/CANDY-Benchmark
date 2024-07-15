// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by tuidan on 2021/7/21.
//
#ifndef SESAME_SRC_ALGORITHM_STREAMKM_HPP_
#define SESAME_SRC_ALGORITHM_STREAMKM_HPP_

#include <Algorithm/Algorithm.hpp>
#include <Algorithm/OfflineRefinement/KMeans.hpp>
#include <Algorithm/WindowModel/LandmarkWindow.hpp>
#include <Sinks/DataSink.hpp>
#include <Utils/BenchmarkUtils.hpp>
#include <iostream>

namespace SESAME {

class StreamKMParameter : public SesameParam {
public:
  int windowSize;
  int seed;
  int num_clusters;
};

class StreamKM : public Algorithm {
public:
  StreamKMParameter StreamKMParam;

  // initialize
  LandmarkWindowPtr window;
  vector<PointPtr> inputs;           // buffered inputs.
  vector<PointPtr> streamingCoreset; // intermediate results.
  KMeans km;                         // used for offline processing.
  StreamKM(param_t &cmd_params);

  ~StreamKM();

  void Init() override;

  void RunOnline(PointPtr input) override;

  void RunOffline(DataSinkPtr sinkPtr) override;

private:
  void dumpResults(vector<PointPtr> &centers,
                   vector<vector<SESAME::PointPtr>> groups,
                   DataSinkPtr ptr) const;
};
} // namespace SESAME

#endif // SESAME_SRC_ALGORITHM_STREAMKM_HPP_
