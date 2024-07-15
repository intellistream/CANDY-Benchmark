//
// Created by tuidan on 2021/10/11.
//

#include "Evaluation/CMM.hpp"
#include "Algorithm/DataStructure/GenericFactory.hpp"
#include "Algorithm/Param.hpp"
#include "Utils/Logger.hpp"
#include "Utils/UtilityFunctions.hpp"

#include <cassert>
#include <cmath>
#include <map>
#include <numeric>
#include <omp.h>
#include <queue>
#include <set>

namespace SESAME {

/**
 * @Description: CMMPoint
 */
CMMPoint::CMMPoint(int id, long startTime, long time, std::vector<double> &vec,
                   double a, double lambda, int truth) {
  this->id = id;
  this->startTime = startTime;
  this->dim = (int)vec.size();
  this->vec = vec;
  this->weight = pow(a, lambda * (time - startTime));
  this->truth = truth;
}

double CMMPoint::getDisTo(CMMPointPtr &p) {
  double dis = 0;
  double temp = 0;
  for (int i = 0; i < dim; i++) {
    temp = p->vec[i] - vec[i];
    dis += temp * temp;
  }
  return sqrt(dis);
}

double CMMPoint::knnDis(int k, CMMCluster &c) {
  std::vector<CMMPointPtr> list = c.points;
  int size = c.points.size();
  std::vector<double> diss(size);
  for (int i = 0; i < size; i++) {
    CMMPointPtr p = list.at(i);
    if (list.at(i)->id != this->id) {
      diss[i] = getDisTo(list.at(i));
    }
  }
  sort(diss.begin(), diss.end());
  double sum = 0;
  int num = 0;
  for (int i = 0; i < k && i < size; i++) {
    sum += diss[i];
    num++;
  }
  if (sum == 0)
    return 1;
  else
    return sum / num;
}

/**
 * @Description: CMMCluster
 */
CMMCluster::CMMCluster() {}

void CMMCluster::add(CMMPointPtr &p) { this->points.push_back(p); }

void CMMCluster::getConn() {
  int size = (int)points.size();
  std::vector<double> knhPDis(size);
  double sum = 0;
  for (int i = 0; i < size; i++) {
    CMMPointPtr p = points.at(i);
    knhPDis[i] = p->knnDis(CMM_KNN, *this);
    // the paper says that the value of k has only marginal influence on CMM
    // effectiveness. so here we set k = 10
    sum += knhPDis[i];
  }
  knhDis = sum / size;
  for (int i = 0; i < size; i++) {
    if (type == GTCluster) {
      if (knhPDis[i] < knhDis) {
        points.at(i)->conCL = 1;
      } else {
        if (points.size() == 1) {
          points.at(i)->conCL = 1;
        } else {
          points.at(i)->conCL = knhDis / knhPDis[i];
        }
      }
    } else {
      if (knhPDis[i] < knhDis) {
        points.at(i)->con = 1;
      } else {
        points.at(i)->con = knhDis / knhPDis[i];
      }
    }
  }
}

CMMDriver::CMMDriver(int dim, double a, double lambda) {
  this->dim = dim;
  this->a = a;
  this->lambda = lambda;
  // this->k = k;
}
void CMMDriver::load(const std::vector<PointPtr> &inputs,
                     const std::vector<PointPtr> &predicts, int dim,
                     double time) {
  // time ? weight ?
  // convert to the predicted clustering center index
  for (int i = 0; i < inputs.size(); i++) {
    std::vector<double> features;
    for (int j = 0; j < dim; j++) {
      features.push_back(inputs.at(i)->getFeatureItem(j));
    }
    int cl = inputs.at(i)->getClusteringCenter();
    assert(cl != -1);
    CMMPointPtr p = std::make_shared<CMMPoint>(predicts.at(i)->getIndex(),
                                               (long)predicts.at(i)->getIndex(),
                                               time, features, a, lambda, cl);
    if (CL.count(cl)) {
      CL[cl]->add(p);
    } else {
      CMMClusterPtr c = std::make_shared<CMMCluster>();
      c->groundTruth = cl;
      c->type = GTCluster;
      c->add(p);
      CL.insert(std::pair<int, CMMClusterPtr>(cl, c));
      CLlist.push_back(c);
    }
    int ci = predicts.at(i)->getClusteringCenter();
    if (C.count(ci)) {
      C[ci]->add(p);
    } else {
      CMMClusterPtr c = std::make_shared<CMMCluster>();
      ;
      c->groundTruth = ci;
      c->type = Cluster;
      c->add(p);
      C.insert(std::pair<int, CMMClusterPtr>(ci, c));
      Clist.push_back(c);
    }
  }
}
void CMMDriver::voteMap() {
  /*
    here CMM uses the mapping strategy rather than the vote, and in the paper
    the author also says that Mapping clusters based on majority voting cannot
    recognize emerging or disappearing clusters.
  */
  int csize = (int)Clist.size();
  for (int i = 0; i < csize; i++) {
    CMMClusterPtr c = Clist.at(i);
    std::unordered_map<int, double> map;
    std::vector<CMMPointPtr> list = c->points;
    int psize = (int)list.size();
    for (int j = 0; j < psize; j++) {
      CMMPointPtr p = list.at(j);
      int truth = p->truth;
      if (map.count(truth)) {
        map.insert(std::pair<int, double>(truth, map.at(truth) + p->weight));
      } else {
        map.insert(std::pair<int, double>(truth, p->weight));
      }
    }
    int label = 1;
    double max = 0;
    for (auto &m : map) {
      if (m.second > max) {
        max = m.second;
        label = m.first;
        assert(label != -1);
      }
    }
    c->groundTruth = label;
  }
}

void CMMDriver::getFaultSet() {
  int csize = (int)Clist.size();
  for (int i = 0; i < csize; i++) {
    CMMClusterPtr c = Clist.at(i);
    int truth = c->groundTruth;
    std::vector<CMMPointPtr> list = c->points;
    int psize = list.size();
    for (int j = 0; j < psize; j++) {
      CMMPointPtr p = list.at(j);
      if (p->truth != truth) {
        faultSet.push_back(p);
        if (!faultClu.contains(c)) {
          faultClu.insert(c);
          faultClu.insert(CL.at(p->truth));
        }
      }
    }
  }
}

double CMMDriver::compCMM() {
  getFaultSet();
  if (faultSet.empty()) {
    return 1;
  }
  compCon();
  double totalPen = 0;
  double totalCon = 0;
  int faultPsize = (int)faultSet.size();
  for (int i = 0; i < faultPsize; i++) {
    CMMPointPtr p = faultSet.at(i);
    totalPen += p->weight * p->conCL * (1 - p->con);
    totalCon += p->weight * p->conCL;
  }
  if (totalCon == 0) {
    return 0;
  } else
    return 1 - totalPen / totalCon;
}

void CMMDriver::compCon() {
  for (auto &c : faultClu) {
    c->getConn();
  }
}

double CMMDriver::computeWeight(double deltaTime) {
  const double belta = 2;
  const double lamda = 1;
  return pow(belta, lamda * (deltaTime));
}

// template <template <typename Ty> class T>
// void GetKnn(int i, int k, const T<int> &cluster,
//             const std::vector<PointPtr> &inputs, std::deque<double> &dists,
//             std::deque<int> indexes) {
//   for (auto j : cluster) {
//     if (i != j) {
//       double dist = inputs[i]->L2Dist(inputs[j]);
//       if (dists.size() < k || dist < dists.back()) {
//         int index = 0;
//         while (index < dists.size() && dist > dists[index])
//           ++index;
//         dists[index] = dist;
//         indexes[index] = j;
//         if (dists.size() > k) {
//           dists.pop_back();
//           indexes.pop_back();
//         }
//       }
//     }
//   }
// }

void CMM::Cluster::CalcKnn(int k, const std::vector<PointPtr> &inputs) {
  const int n = points.size();
  // std::vector<std::vector<double>> adjDists(n, std::vector<double>(n, 0));
  //   omp_set_dynamic(0);
  //   omp_set_num_threads(std::min(omp_get_num_procs(), 8));
  // #pragma omp parallel for
  // for (int i = 0; i < n; ++i) {
  //   for (int j = i + 1; j < n; ++j) {
  //     adjDists[i][j] = adjDists[j][i] =
  //         inputs[vpoints[i]]->L2Dist(inputs[vpoints[j]]);
  //   }
  // }

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    auto first = std::numeric_limits<double>::max(), second = first;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        // double dist = adjDists[i][j];
        double dist = inputs[vpoints[i]]->L2Dist(inputs[vpoints[j]]);
        if (dist < first) {
          second = first;
          first = dist;
        } else if (dist < second) {
          second = dist;
        }
      }
    }
    double avgKnn = 0.0;
    if (n >= 3) {
      avgKnn = (first + second) / 2;
    } else if (n == 2) {
      avgKnn = first;
    }
    inputs[vpoints[i]]->knn = avgKnn;
    knnMeanAvg += avgKnn;
    knnDevAvg += avgKnn * avgKnn;
  }
  knnMeanAvg = knnMeanAvg.load() / n;
  knnDevAvg = knnDevAvg.load() / n;
  double variance = knnDevAvg - knnMeanAvg * knnMeanAvg;
  if (variance <= 0.0)
    variance = 1e-50;
  knnDevAvg = sqrt(variance);

  // calculate the connectivity of each point
  double upperKnn = knnMeanAvg + knnDevAvg;
  for (int i = 0; i < n; ++i) {
    if (inputs[vpoints[i]]->knn < upperKnn)
      inputs[vpoints[i]]->conn = 1;
    else
      inputs[vpoints[i]]->conn = upperKnn / inputs[vpoints[i]]->knn;
  }
}

double CMM::CalcConn(int i, int j, const std::vector<PointPtr> &inputs) {
  // std::deque<double> dists;
  // std::deque<int> indexes;
  // GetKnn(i, knnNeighbourhood, clusters[j].points, inputs, dists, indexes);
  const auto n = clusters[j].vpoints.size();
  auto first = std::numeric_limits<double>::max(), second = first;
#pragma omp parallel for
  for (int k = 0; k < n; ++k) {
    double dist = inputs[i]->L2Dist(inputs[clusters[j].vpoints[k]]);
    if (dist < first) {
      second = first;
      first = dist;
    } else if (dist < second) {
      second = dist;
    }
  }
  double avgKnn = 0.0;
  if (n >= 2) {
    avgKnn = (first + second) / 2;
  } else {
    avgKnn = first;
  }
  double upperKnn = clusters[inputs[i]->clu_id].knnMeanAvg +
                    clusters[inputs[i]->clu_id].knnDevAvg;
  if (avgKnn < upperKnn) {
    return 1;
  } else {
    // TODO: useExpConnectivity
    return upperKnn / avgKnn;
  }
}

double CMM::CalcConn(int i, const std::vector<PointPtr> &inputs) {
  return CalcConn(i, inputs[i]->clu_id, inputs);
}

/*here we assume that the data comes every 10 seconds
 * */
void CMM::AnalyseGT(const std::vector<PointPtr> &inputs,
                    const std::vector<PointPtr> &predicts,
                    bool enableClassMerge) {
  if (!enableClassMerge) {
    tauConnection = 1.0;
  }
  lambdaConn = -log(lambdaConnRefXValue) / log(2) / lambdaConnX;
  // std::unordered_map<int, int> mapTrueLabelToWorkLabel;
  // std::vec
  for (int i = 0; i < inputs.size(); ++i)
    if (inputs[i]->clu_id != -1) {
      auto clu = inputs[i]->clu_id;
      clusters[clu].Insert(i);
    } else {
      clusters[0].Insert(i);
    }
  std::cerr << "CMM::CalcKnn start" << std::endl;
  for (auto &cluster : clusters) {
    if (cluster.first != 0)
      cluster.second.CalcKnn(knnNeighbourhood, inputs);
  }
  // for (int i = 0; i < inputs.size(); ++i)
  //   if (inputs[i]->clu_id != -1) {
  //     inputs[i]->conn = CalcConn(i, inputs);
  //   }
}

void CMM::CalcMatch(const std::vector<PointPtr> &inputs,
                    const std::vector<PointPtr> &predicts) {
  std::map<int, std::map<int, int>> mapFC, mapGT;
  std::map<int, int> sumsFC;
  for (int i = 0; i < inputs.size(); ++i) {
    auto fc = predicts[i]->clu_id, hc = inputs[i]->clu_id;
    if (fc != -1) {
      if (hc != -1) {
        mapFC[fc][hc]++;
        sumsFC[fc]++;
      }
    }
    if (hc != -1) {
      // TODO: ?
      mapGT[hc][hc]++;
    }
  }
  for (auto &it : mapFC) {
    auto fc = it.first;
    int matchIndex = -1;
    for (auto &jt : it.second) {
      auto hc = jt.first;
      if (jt.second != 0) {
        if (matchIndex == -1) {
          matchIndex = hc;
        } else {
          matchIndex = -1;
          break;
        }
      }
    }

    int minDiff = std::numeric_limits<int>::max();
    if (sumsFC[fc] != 0 && matchIndex == -1) {
      std::vector<int> fitCandidates;
      for (int hc0 = 1; hc0 <= param.num_clusters; ++hc0) {
        int errDiff = 0;
        for (int hc1 = 1; hc1 <= param.num_clusters; ++hc1) {
          double freq_diff = mapFC[fc][hc1] - mapGT[hc0][hc1];
          if (freq_diff > 0) {
            errDiff += freq_diff;
          }
        }
        if (errDiff == 0) {
          fitCandidates.push_back(hc0);
        }
        if (errDiff < minDiff) {
          minDiff = errDiff;
          matchIndex = hc0;
        }
      }
      if (fitCandidates.size() != 0) {
        int bestGTfit = fitCandidates[0];
        for (int i = 1; i < fitCandidates.size(); ++i) {
          if (mapGT[fc][fitCandidates[i]] > mapFC[fc][bestGTfit]) {
            bestGTfit = fitCandidates[i];
          }
        }
        matchIndex = bestGTfit;
      }
    }
    matchMap[fc] = matchIndex;
    // assert(matchIndex != -1);
  }
}

double CMM::MisplacedError(int i, int fc, int hc,
                           const std::vector<PointPtr> &inputs) {
  auto real_hc = matchMap[fc];
  if (real_hc == -1)
    return 1;
  if (real_hc == hc)
    return 0;
  else {
    if (clusters[real_hc].points.contains(i)) {
      return 0.00001;
    } else {
      double weight = 1 - CalcConn(i, real_hc, inputs);
      return weight * inputs[i]->conn;
    }
  }
  // return inputs[i]->conn;
}

double CMM::NoiseError(int i, int fc, int hc,
                       const std::vector<PointPtr> &inputs) {
  auto real_hc = matchMap[fc];
  if (real_hc == -1)
    return 1;
  if (clusters[real_hc].points.contains(i))
    return 0.00001;
  else {
    return 1.0 - CalcConn(i, real_hc, inputs);
  }
}

double CMM::MissedError(int i, int fc, int hc,
                        const std::vector<PointPtr> &inputs) {
  // TODO
  return inputs[i]->conn;
  // for(int i = 1; i<=param.num_clusters; ++i) {
  //   if(matchMap[i]!=-1 && matchMap[i] == hc) {
  //   }
  // }
}

void CMM::CalcError(const std::vector<PointPtr> &inputs,
                    const std::vector<PointPtr> &predicts) {
  int totalErrorCount = 0;
  // int totalRedundancy = 0;
  int trueCoverage = 0;
  int totalCoverage = 0;

  int numNoise = 0;
  double errorNoise = 0;
  double errorNoiseMax = 0;

  double errorMissed = 0;
  // double errorMissedMax = 0;

  double errorMisplaced = 0;
  double errorMisplacedMax = 0;

  double totalError = 0.0;
  double totalErrorMax = 0.0;

  for (int i = 0; i < inputs.size(); ++i) {
    auto fc = predicts[i]->clu_id, hc = inputs[i]->clu_id;
    if (hc == -1) {
      numNoise++;
      errorNoiseMax += inputs[i]->weight * inputs[i]->conn;
    } else {
      errorMissed += inputs[i]->weight * inputs[i]->conn;
      errorMisplacedMax += inputs[i]->weight * inputs[i]->conn;
    }
    totalErrorMax += inputs[i]->weight;
    double err = 0;
    bool flag = false;
    if (fc != -1) {
      flag = true;
      if (hc != -1) {
        if (matchMap[fc] == hc) {
          // true match
        } else {
          err = MisplacedError(i, fc, hc, inputs);
          errorMisplaced += inputs[i]->weight * err;
        }
      } else {
        // noise
        err = NoiseError(i, fc, hc, inputs);
        errorNoise += inputs[i]->weight * err;
      }
    } else {
      if (hc != -1) {
        err = MissedError(i, fc, hc, inputs);
        errorMissed += inputs[i]->weight * err;
      } else {
        // NOISE
      }
    }

    totalError += err * inputs[i]->weight;
    if (err != 0)
      totalErrorCount++;
    if (flag)
      totalCoverage++;
    if (flag && hc != -1)
      trueCoverage++;
  }
  cmm = (totalErrorMax != 0) ? (1 - totalError / totalErrorMax) : 1.0;
}

/*here we assume that the data comes every 10 seconds
 * */
double CMM::Evaluate(const std::vector<PointPtr> &inputs,
                     const std::vector<PointPtr> &predicts) {
#ifndef NDEBUG
  for (int i = 0; i < inputs.size(); i++) {
    assert(inputs[i]->index == predicts[i]->index);
  }
#endif
  AnalyseGT(inputs, predicts, false);
  std::cerr << "CMM::CalcMatch start" << std::endl;
  CalcMatch(inputs, predicts);
  std::cerr << "CMM::CalcError start" << std::endl;
  CalcError(inputs, predicts);
  return cmm;
}

} // namespace SESAME