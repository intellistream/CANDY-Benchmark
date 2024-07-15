//
// Created by tuidan on 2023/1/3.
//
#include "Evaluation/NMI.hpp"
#include <cmath>

double SESAME::NMI::Evaluate(const std::vector<PointPtr> &inputs,
                             const std::vector<PointPtr> &predicts, int gt_size,
                             int predict_size) {
  std::vector<int> gt_cap(gt_size, 0);
  std::vector<int> predict_cap(predict_size, 0);
  std::vector<std::vector<int>> mutual(gt_size,
                                       std::vector<int>(predict_size, 0));
  for (int i = 0; i < inputs.size(); i++) {
    auto gtClu_id = inputs[i]->getClusteringCenter() - 1;
    auto predClu_id = predicts[i]->getClusteringCenter();

    // calculate the gt and predict cluster probability
    gt_cap[gtClu_id]++;
    predict_cap[predClu_id]++;

    // calculate the joint probability
    mutual[gtClu_id][predClu_id]++;
  }
  double gtEntropy = 0;
  double predEntropy = 0;
  double mutualInformation = 0;
  for (const auto &gt : gt_cap) {
    double gtProb = (double)gt / (double)inputs.size();
    gtEntropy -= gtProb * log2(gtProb);
  }
  for (const auto &pred : predict_cap) {
    if (pred != 0) {
      double predProb = (double)pred / (double)inputs.size();
      predEntropy -= predProb * log2(predProb);
    }
  }
  for (int i = 0; i < gt_size; i++) {
    for (int j = 0; j < predict_size; j++) {
      double jointProb = (double)mutual[i][j] / (double)inputs.size();
      double gtProb = (double)gt_cap[i] / (double)inputs.size();
      if (jointProb != 0) {
        double predProb = (double)predict_cap[j] / (double)inputs.size();
        mutualInformation += jointProb * log2(jointProb / (gtProb * predProb));
      }
    }
  }
  double nmi = 2 * mutualInformation / (predEntropy + gtEntropy);
  return nmi;
}
