// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by tuidan on 2021/7/22.
//

#ifndef SESAME_INCLUDE_ALGORITHM_ALGORITHM_HPP_
#define SESAME_INCLUDE_ALGORITHM_ALGORITHM_HPP_

#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/Param.hpp"
#include "Evaluation/Evaluation.hpp"
#include "Sinks/DataSink.hpp"
#include "Timer/Timer.hpp"

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace SESAME {
class Algorithm;
typedef std::shared_ptr<Algorithm> AlgorithmPtr;

class Algorithm {
public:
  Algorithm() = default;
  virtual ~Algorithm() = default;
  virtual void Init() = 0;
  virtual void RunOnline(SESAME::PointPtr input) = 0;
  virtual void RunOffline(SESAME::DataSinkPtr ptr) = 0;
  void Insert(SESAME::PointPtr input) {};
  virtual void OutputOnline(std::vector<PointPtr> &centers) {};
  void Store(std::string output_file, int dim, std::vector<PointPtr> results);
  Timer win_timer, ds_timer, out_timer, ref_timer, sum_timer, lat_timer,
      on_timer;
  param_t param;
  int cnt = 0;
  std::vector<int64> et;
  PerfRes GetPerf() {
    PerfRes res;
    res.win_us = win_timer.sum / 1000;
    res.ds_us = ds_timer.sum / 1000;
    res.out_us = out_timer.sum / 1000;
    res.ref_us = ref_timer.sum / 1000;
    res.sum_us = sum_timer.sum / 1000;
    if (et.size() == 5) {
      res.on_20 = et[0] / 1e6;
      res.on_40 = et[1] / 1e6;
      res.on_60 = et[2] / 1e6;
      res.on_80 = et[3] / 1e6;
      res.on_100 = et[4] / 1e6;
    }
    res.lat_us = lat_timer.sum / 1e3 / param.num_points;
    res.et_s = on_timer.sum / 1e9;
    res.qps = param.num_points * 1e9 / sum_timer.sum;
    return res;
  }
  void Count() {
    ++cnt;
    if (cnt >= param.num_points * 0.2) {
      auto now = std::chrono::high_resolution_clock::now();
      et.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                       now - sum_timer.start)
                       .count());
      cnt = 1;
    }
  }
};

} // namespace SESAME

#endif // SESAME_INCLUDE_ALGORITHM_ALGORITHM_HPP_
