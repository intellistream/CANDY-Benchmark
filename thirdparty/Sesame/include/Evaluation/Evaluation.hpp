// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by tuidan on 2021/7/20.
//

#ifndef ONLINEMLBENCHMARK_EVALUATIONMETRICS_HPP_
#define ONLINEMLBENCHMARK_EVALUATIONMETRICS_HPP_

#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/Param.hpp"
#include "Utils/UtilityFunctions.hpp"

#include <vector>

namespace SESAME {

struct AccuracyRes {
  double cmm = 0.0, purity = 0.0, nmi = 0.0;
  int num_res = 0;
  AccuracyRes() = default;
  void Evaluate(const param_t &param, const std::vector<PointPtr> &inputs,
                const std::vector<PointPtr> &predicts);
  void Print() {
    std::cout << "num_res: " << num_res << std::endl;
    std::cout << "cmm: " << cmm << std::endl;
    std::cout << "purity: " << purity << std::endl;
    std::cout << "nmi: " << nmi << std::endl;
  }
};

struct PerfRes {
  int64 win_us, ds_us, out_us, ref_us, sum_us;
  double on_20 = 0.0, on_40 = 0.0, on_60 = 0.0, on_80 = 0.0, on_100 = 0.0;
  double lat_us, et_s, qps;
  void Print() {
    std::cout << "win_us: " << win_us << std::endl;
    std::cout << "ds_us: " << ds_us << std::endl;
    std::cout << "out_us: " << out_us << std::endl;
    std::cout << "ref_us: " << ref_us << std::endl;
    std::cout << "sum_us: " << sum_us << std::endl;
    std::cout << "on_20: " << on_20 << std::endl;
    std::cout << "on_40: " << on_40 << std::endl;
    std::cout << "on_60: " << on_60 << std::endl;
    std::cout << "on_80: " << on_80 << std::endl;
    std::cout << "on_100: " << on_100 << std::endl;
    std::cout << "lat_us: " << lat_us << std::endl;
    std::cout << "et_s: " << et_s << std::endl;
    std::cout << "qps: " << qps << std::endl;
  }
};

} // namespace SESAME
#endif // ONLINEMLBENCHMARK_EVALUATIONMETRICS_H
