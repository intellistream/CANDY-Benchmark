//
// Created by tuidan on 2021/10/11.
//
#include "Evaluation/Purity.hpp"
#include "Utils/Logger.hpp"
#include "Utils/UtilityFunctions.hpp"

#include <atomic>
#include <cmath>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>

/**
 * @Description: Please note that the order of the cluster index has to be
 * consecutive
 */
void SESAME::Purity::pointToGroup(const std::vector<SESAME::PointPtr> &input,
                                  std::vector<std::vector<PointPtr>> &group) {
  std::unordered_map<int, std::vector<PointPtr>> centerMap;
  for (auto el : input) {
    centerMap[el->getClusteringCenter()].push_back(el);
  }
  for (auto it : centerMap) {
    group.push_back(it.second);
  }
}

/**
 * @Description: count the number of the elements in clusterA contained in
 * clusterB?
 */
double
SESAME::Purity::calculateBelongsFromTwo(std::vector<SESAME::PointPtr> &groupA,
                                        std::vector<SESAME::PointPtr> &groupB) {
  double count = 0;
  std::unordered_set<int> setB;
  for (auto &el : groupB) {
    setB.insert(el->getIndex());
  }
  for (auto &el : groupA) {
    if (setB.contains(el->getIndex())) {
      count += el->getWeight();
    }
  }
  return count;
}

/**
 * @Description: return the max count of the cluster
 */
double
SESAME::Purity::getMaxBelongs(std::vector<SESAME::PointPtr> &singleSample,
                              std::vector<std::vector<PointPtr>> &GT) {
  double max = 0;
  for (auto &re : GT) {
    double temp = calculateBelongsFromTwo(singleSample, re);
    if (max <= temp)
      max = temp;
  }
  return max;
}

double SESAME::Purity::purityCost(const std::vector<SESAME::PointPtr> &inputs,
                                  const std::vector<SESAME::PointPtr> &predicts,
                                  int dim, bool decay) {
  double purity;
  double size = 0;
  for (int i = 0; i < predicts.size(); i++) {
    double w = 1;
    if (decay) { // 分段函数来设置weight
      if (predicts.size() - predicts[i]->getIndex() <= 101) {
        w = 1;
      } else if (predicts[i]->getIndex() < predicts.size() / 100) {
        w = 0;
      } else {
        w = double(predicts[i]->getIndex() - predicts.size() / 100) /
            double(predicts.size() - 100 - predicts.size() / 100);
      }
    } else {
      w = 1;
    }
    predicts[i]->setWeight(w);
    size += w;
  }

  std::vector<std::vector<PointPtr>> GT;
  std::vector<std::vector<PointPtr>> sample;

  pointToGroup(inputs, GT);
  pointToGroup(predicts, sample);

  std::atomic<double> sum = 0;
#pragma omp parallel for
  for (auto &el : sample) {
    sum += getMaxBelongs(el, GT);
  }
  if (!inputs.empty()) {
    purity = sum / size;
  } else {
    purity = 0;
  }
  return purity;
}