// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

#include "Evaluation/Evaluation.hpp"
#include "Algorithm/DataStructure/GenericFactory.hpp"
#include "Algorithm/Param.hpp"
#include "Evaluation/CMM.hpp"
#include "Evaluation/Euclidean.hpp"
#include "Evaluation/NMI.hpp"
#include "Evaluation/Purity.hpp"
#include "Timer/Timer.hpp"
#include "Utils/Logger.hpp"
#include "Utils/UtilityFunctions.hpp"

#include <cmath>

namespace SESAME {
void AccuracyRes::Evaluate(const param_t &param,
                           const std::vector<PointPtr> &inputs,
                           const std::vector<PointPtr> &predicts) {
  if (param.run_eval && param.num_res > 0 && param.num_res <= 60000) {
    Timer pur_timer, cmm_timer, nmi_timer;
    std::cerr << "Accuracy:" << std::endl;
    pur_timer.Tick();
    if (param.run_pur) {
      std::cerr << "Purity begin" << std::endl;
      purity =
          Purity::purityCost(inputs, predicts, param.dim, param.time_decay);
    }
    pur_timer.Tock();
    std::cerr << "\033[1;34mPurity: " << round(purity * 10000) / 10000
              << "\033[0m et_s: " << pur_timer.sum / 1e9 << std::endl;
    nmi_timer.Tick();
    if (param.run_nmi) {
      std::cerr << "NMI begin" << std::endl;
      nmi = NMI::Evaluate(inputs, predicts, param.num_clusters, param.num_res);
    }
    nmi_timer.Tock();
    std::cerr << "\033[1;34mNMI: " << round(nmi * 10000) / 10000
              << "\033[0m et_s: " << nmi_timer.sum / 1e9 << std::endl;
    cmm_timer.Tick();
    if (param.run_cmm) {
      std::cerr << "CMM begin" << std::endl;
      CMM eval(param);
      cmm = eval.Evaluate(inputs, predicts);
    }
    cmm_timer.Tock();
    std::cerr << "\033[1;34mCMM: " << round(cmm * 10000) / 10000
              << "\033[0m et_s: " << cmm_timer.sum / 1e9 << std::endl;
  } else {
    std::cerr << "skip evaluation" << std::endl;
  }
}

} // namespace SESAME
