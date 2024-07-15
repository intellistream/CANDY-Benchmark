//
// Created by tuidan on 2021/9/1.
//

#include <Algorithm/DataStructure/DPTree.hpp>
#include <Utils/Logger.hpp>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <unordered_set>

using namespace std;

/**
 * DPTree
 */
SESAME::DPTree::DPTree() {}
SESAME::DPTree::~DPTree() {}
SESAME::DPTree::DPTree(int num, double CluR) {
  this->num = num;
  std::vector<DPNodePtr> clus(this->num);
  this->Clus = clus;
  this->CluR = CluR;
  this->cluLabel = 0;
}
void SESAME::DPTree::insert(SESAME::DPNodePtr &cc, int opt) {
  cc->SetActive(true);
  Clus[size] = cc;
  size++;

  if (opt == 0) {
    adjustNoOpt(size - 1);
  }
  if (opt == 1) {
    adjustOpt1(size - 1);
  }

  if (opt == 2) {
    adjust(size - 1);
  }

  if (opt == -1) {
    adjustNoDelta(size - 1);
  }

  if (size == num) {
    SESAME_DEBUG("lack of DPTree nodes");
  }
}
void SESAME::DPTree::Init(std::vector<SESAME::DPNodePtr> &clus, int size,
                          double minRho, double minDelta, SESAME::OutPtr &outs,
                          std::unordered_set<ClusterPtr> &clusters) {
  this->minDelta = minDelta;
  Clus[0] = clus[0];
  SESAME::ClusterPtr cluster = std::make_shared<Cluster>(cluLabel++);
  cluster->add(Clus[0]);
  Clus[0]->SetCluster(cluster);
  clusters.insert(cluster);
  int i = 1;

  for (; i < size && clus[i]->GetRho() >= minRho; i++) {
    Clus[i] = clus[i];
    std::unordered_set<DPNodePtr> sucs = Clus[i]->GetDep()->GetSucs();
    sucs.insert(Clus[i]);
    if (Clus[i]->GetDelta() > minDelta) {
      SESAME::ClusterPtr c = std::make_shared<Cluster>(cluLabel++);
      ;
      c->add(Clus[i]);
      Clus[i]->SetCluster(c);
      clusters.insert(c);
    } else {
      Clus[i]->GetDep()->GetCluster()->add(Clus[i]);
      Clus[i]->SetCluster(Clus[i]->GetDep()->GetCluster());
    }
  }
  double maxDelta = 0;
  for (int j = 1; j < size && j < i; j++) {
    if (maxDelta < Clus[j]->GetDelta()) {
      maxDelta = Clus[j]->GetDelta();
    }
  }
  Clus[0]->SetDelta(maxDelta);
  this->size = i;
  for (; i < size; i++) {
    outs->insert(clus[i]);
  }
}
SESAME::DPNodePtr SESAME::DPTree::findNN(SESAME::PointPtr p, double coef,
                                         int opt, double time) {
  int index = 0;
  double dis = 0;
  auto minDis = DBL_MAX;
  for (int i = 0; i < size; i++) {
    Clus[i]->SetRho(Clus[i]->GetRho() * coef);
    dis = p->L2Dist(Clus[i]->GetCenter());
    Clus[i]->SetDis(dis);
    if (dis < minDis) {
      minDis = dis;
      index = i;
    }
  }

  p->setMinDist(minDis);
  auto cc = Clus[index];
  if (minDis <= CluR) {
    Clus[index]->insert(time);
    if (opt == 0) {
      adjustNoOpt(index);
    }
    if (opt == 1) {
      adjustOpt1(index);
    }

    if (opt == 2) {
      adjust(index);
    }

    if (opt == -1) {
      adjustNoDelta(index);
    }
  }
  return cc;
}
void SESAME::DPTree::adjustNoDelta(int index) {
  Clus[0]->SetDelta(DBL_MAX);
  auto clu = Clus[index];

  if (index > 0) {
    for (int i = index - 1; i >= 0; i--) {
      if (clu->GetRho() > Clus[i]->GetRho()) {
        Clus[i + 1] = Clus[i];
        Clus[i] = clu;
      } else {
        break;
      }
    }
  }
}
void SESAME::DPTree::adjustNoOpt(int index) {
  Clus[0]->SetDelta(DBL_MAX);
  auto clu = Clus[index];
  int position = index;

  if (index > 0) {
    for (int i = index - 1; i >= 0; i--) {
      if (clu->GetRho() > Clus[i]->GetRho()) {
        Clus[i + 1] = Clus[i];
        Clus[i] = clu;
        position = i;
      } else {
        break;
      }
    }
  }
  if (Clus[0] == clu) {
    clu->SetDelta(DBL_MAX);
  }

  computeDeltaNoOpt(position);
  computeHeadDelta();
}
void SESAME::DPTree::computeDeltaNoOpt(int index) {
  auto clu = Clus[index];
  if (clu->GetDep() != nullptr) {
    clu->GetDep()->removeSuccessor(clu);
  }
  double dis = 0;
  clu->SetDelta(DBL_MAX);
  for (int i = size - 1; i >= 0; i--) {
    if (i < index) {
      dis = clu->GetCenter()->L2Dist(Clus[i]->GetCenter());
      if (clu->GetDelta() > dis) {
        if (clu->GetDep() != nullptr) {
          clu->GetDep()->removeSuccessor(clu);
        }
        clu->SetDelta(dis);
        clu->SetDep(Clus[i]);
        Clus[i]->addSuccessor(clu);
      }
    }
    if (i > index) {
      if (Clus[i]->GetDelta() > dis) {
        if (Clus[i]->GetDep() != nullptr) {
          Clus[i]->GetDep()->removeSuccessor(Clus[i]);
        }
        Clus[i]->SetDep(clu);
        clu->addSuccessor(Clus[i]);
        Clus[i]->SetDelta(dis);
      }
    }
  }
}
void SESAME::DPTree::adjustOpt1(int index) {
  Clus[0]->SetDelta(DBL_MAX);
  auto clu = Clus[index];
  if (clu->GetDep() != nullptr && clu->GetDep()->GetRho() < clu->GetRho()) {
    clu->GetDep()->removeSuccessor(clu);
    clu->SetDelta(DBL_MAX);
  }
  int position = index;
  double dis = 0;

  if (index > 0) {
    for (int i = index - 1; i >= 0; i--) {
      if (clu->GetRho() > Clus[i]->GetRho()) {
        dis = Clus[i]->getDisTo(clu);
        if (dis <= Clus[i]->GetDelta()) {
          if (Clus[i]->GetDep() != nullptr) {
            Clus[i]->GetDep()->removeSuccessor(Clus[i]);
          }
          Clus[i]->SetDep(clu);
          clu->addSuccessor(Clus[i]);
          Clus[i]->SetDelta(dis);
        }
        Clus[i + 1] = Clus[i];
        Clus[i] = clu;
        position = i;
      } else {
        break;
      }
    }
  }
  if (Clus[0] == clu) {
    clu->SetDelta(DBL_MAX);
    position = 0;
  }

  if (position != 0 &&
      (clu->GetDep() == nullptr || clu->GetRho() > clu->GetDep()->GetRho())) {
    clu->SetDelta(DBL_MAX);

    computeDeltaF1(position);
  }
  computeHeadDelta();
}
void SESAME::DPTree::computeDeltaF1(int index) {
  auto clu = Clus[index];
  if (clu->GetDep() != nullptr) {
    clu->GetDep()->removeSuccessor(clu);
  }
  clu->SetDelta(DBL_MAX);
  if (index == 0) {
    return;
  }
  double dis = 0;

  // with one optimization

  for (int i = index - 1; i >= 0; i--) {
    dis = clu->GetCenter()->L2Dist(Clus[i]->GetCenter());
    if (dis < clu->GetDelta()) {
      clu->SetDep(Clus[i]);
      clu->SetDelta(dis);
    }
  }

  if (clu->GetDep() != nullptr) {
    clu->GetDep()->addSuccessor(clu);
  }
}
void SESAME::DPTree::adjust(int index) {
  Clus[0]->SetDelta(DBL_MAX);
  auto clu = Clus[index];
  if (clu->GetDep() != nullptr && clu->GetDep()->GetRho() < clu->GetRho()) {
    clu->GetDep()->removeSuccessor(clu);
    clu->SetDelta(DBL_MAX);
  }
  int position = index;
  double dis = 0;

  if (index > 0) {
    for (int i = index - 1; i >= 0; i--) {
      if (clu->GetRho() > Clus[i]->GetRho()) {
        if (Clus[i]->GetDelta() > Clus[i]->GetDis() - clu->GetDis()) {
          dis = Clus[i]->getDisTo(clu);
          if (dis < Clus[i]->GetDelta()) {
            if (Clus[i]->GetDep() != nullptr) {
              Clus[i]->GetDep()->removeSuccessor(Clus[i]);
            }
            Clus[i]->SetDep(clu);
            clu->addSuccessor(Clus[i]);
            Clus[i]->SetDelta(dis);
          }
        }
        Clus[i + 1] = Clus[i];
        Clus[i] = clu;
        position = i;
      } else {
        break;
      }
    }
  }
  if (Clus[0] == clu) {
    clu->SetDelta(DBL_MAX);
  }
  if (position != 0 &&
      (clu->GetDep() == nullptr || clu->GetRho() > clu->GetDep()->GetRho())) {
    clu->SetDelta(DBL_MAX);

    computeDelta(position);
  }
  computeHeadDelta();
}
void SESAME::DPTree::computeHeadDelta() {
  auto clu = Clus[0];
  if (clu->GetDep() != nullptr) {
    clu->GetDep()->removeSuccessor(clu);
  }

  double maxValue = 0;
  double secondValue = 0;
  for (int i = 1; i < size; i++) {
    if (maxValue < Clus[i]->GetDelta()) {
      secondValue = maxValue;
      maxValue = Clus[i]->GetDelta();
    } else if (secondValue < Clus[i]->GetDelta()) {
      secondValue = Clus[i]->GetDelta();
    }
  }
  if (maxValue > 3 * secondValue) {
    clu->SetDelta(maxValue);
  } else {
    clu->SetDelta((maxValue + secondValue) / 2);
  }
}
void SESAME::DPTree::computeDelta(int index) {
  auto clu = Clus[index];
  if (clu->GetDep() != nullptr) {
    clu->GetDep()->removeSuccessor(clu);
  }
  clu->SetDelta(DBL_MAX);
  if (index == 0) {
    return;
  }
  double dis = 0;

  for (int i = index - 1; i >= 0; i--) {
    if (clu->GetDelta() > Clus[i]->GetDis() - clu->GetDis()) {
      dis = clu->GetCenter()->L2Dist(Clus[i]->GetCenter());

      if (dis < clu->GetDelta()) {
        clu->SetDep(Clus[i]);
        clu->SetDelta(dis);
      }
    }
  }

  if (clu->GetDep() != nullptr) {
    clu->GetDep()->addSuccessor(clu);
  }
}
void SESAME::DPTree::deleteInact(SESAME::OutPtr &outres, double minRho,
                                 double time) {
  for (int i = size - 1; i > 0; i--) {
    if (Clus[i]->GetRho() < minRho) {
      auto cc = Clus[i];
      Clus[i] = nullptr;
      size--;
      cc->SetActive(false);
      cc->SetInactiveTime(time);
      std::unordered_set<DPNodePtr> cells = cc->GetDep()->GetSucs();
      cells.erase(cc);
      cc->GetCluster()->remove(cc);
      outres->insert(cc);
    } else {
      break;
    }
  }
  if (size > 0 && Clus[0]->GetRho() < minRho) {
    auto cc = Clus[0];
    Clus[0] = nullptr;
    size--;
    cc->SetActive(false);
    cc->SetInactiveTime(time);
    cc->GetCluster()->remove(cc);
    outres->insert(cc);
  }
}
double SESAME::DPTree::computeAlpha(double minDelta) {
  std::vector<double> deltas(size);
  for (int i = 0; i < size; i++) {
    deltas[i] = Clus[i]->GetDelta();
  }
  sort(deltas.begin(), deltas.end());
  double delta1;
  double delta2;
  int m = 0;
  int n = 0;
  double avg;
  double up = 0;
  double down = 0;
  int i = 0;
  for (i = 0; i < size - 1 && deltas[i] < minDelta; i++) {
    n++;
    down += deltas[i];
  }
  delta1 = deltas[i - 1];
  delta2 = deltas[i];
  for (; i < size; i++) {
    m++;
    up += deltas[i];
  }
  avg = up + down;
  avg = avg / (m + n);
  up = up / m;
  down = down / n;
  double alpha = (up * (down - delta1) * (m * up + delta1)) /
                 (avg * avg * (delta1 - up) * (n - 1) +
                  (down - delta1) * up * (m * up + delta1));
  double alpha2 = ((delta2 - down) * up * (m * up - delta2)) /
                  ((delta2 - down) * up * (m * up - delta2) +
                   avg * avg * (up - delta2) * (n + 1));
  if (alpha < alpha2) {
    return (alpha + alpha2) / 2;
  } else {
    return 0;
  }
}
double SESAME::DPTree::adjustMinDelta(double alpha) {
  if (size < 2) {
    return 0;
  }
  std::vector<double> deltas(size);
  for (int i = 0; i < size; i++) {
    deltas[i] = Clus[i]->GetDelta();
  }
  sort(deltas.begin(), deltas.end());
  int m = 0;
  int n = 0;
  double avg = 0;
  double up = 0;
  double down = 0;
  up = deltas[size - 1];
  for (int i = 0; i < size - 1; i++) {
    down += deltas[i];
  }
  n = size - 1;

  m = 1;
  avg = (up + down) / (m + n);
  double score = fun(alpha, up / m, down / n, avg);
  int index = size - 2;
  up += deltas[index];
  m++;
  down -= deltas[index];
  n--;
  double scoredown = fun(alpha, up / m, down / n, avg);
  while (score > scoredown && index > 0) {
    score = scoredown;
    index--;
    up += deltas[index];
    m++;
    down -= deltas[index];
    n--;
    scoredown = fun(alpha, up / m, down / n, avg);
  }
  return (deltas[index + 1] + deltas[index]) / 2;
}
double SESAME::DPTree::fun(double alpha, double upavg, double downavg,
                           double avg) {
  return alpha * (avg / upavg) + (1 - alpha) * (downavg / avg);
}
void SESAME::DPTree::adjustCluster(
    std::unordered_set<SESAME::ClusterPtr> &clusters) {
  std::vector<ClusterPtr> set; //

  if (Clus[0] == nullptr) {
    // SESAME_DEBUG("there is no cluser-cell, r is very small");
    // SESAME_DEBUG("please adjust your r parameter larger");
  } else if (Clus[0]->GetCluster() ==
             nullptr) { // there is new cluster center appearing
    auto cluster = std::make_shared<Cluster>(cluLabel++);
    clusters.insert(cluster);
    cluster->add(Clus[0]);
    Clus[0]->SetCluster(cluster);
    set.push_back(cluster);
  } else {
    set.push_back(Clus[0]->GetCluster());
  }

  for (int i = 1; i < size; i++) {
    if (Clus[i]->GetDelta() >= minDelta) { // clusters split or new cluster
      // appears
      if (Clus[i]->GetDep() == nullptr) {
        SESAME_DEBUG("error contains null dep");
      }
      if (Clus[i]->GetCluster() ==
          Clus[i]->GetDep()->GetCluster()) { // clusters split
        auto c1 = Clus[i]->GetCluster();
        auto c2 = std::make_shared<Cluster>(cluLabel++);
        if (c1 != nullptr) {
          c1->remove(Clus[i]);
        }
        c2->add(Clus[i]);
        Clus[i]->SetCluster(c2);
        clusters.insert(c2);
        set.push_back(c2);
      }
      if (Clus[i]->GetCluster() == nullptr) { // new cluster appears
        auto cluster = std::make_shared<Cluster>(cluLabel++);
        clusters.insert(cluster);
        cluster->add(Clus[i]);
        Clus[i]->SetCluster(cluster);
        set.push_back(cluster);
      } else {
        bool flag = false;
        for (auto c : set) {
          if (c->GetLabel() == Clus[i]->GetCluster()->GetLabel())
            flag = true;
        }
        if (flag) { // this is because when
          // minDelta become smaller
          Clus[i]->GetCluster()->remove(Clus[i]);
          auto cluster = std::make_shared<Cluster>(cluLabel++);
          clusters.insert(cluster);
          cluster->add(Clus[i]);
          Clus[i]->SetCluster(cluster);
          set.push_back(cluster);
        } else {
          set.push_back(Clus[i]->GetCluster());
        }
      }
    } else {
      if (Clus[i]->GetDep() == nullptr) {
        SESAME_DEBUG("error contains null dep");
      }
      if (Clus[i]->GetCluster() != Clus[i]->GetDep()->GetCluster()) {
        auto c1 = Clus[i]->GetCluster();
        auto c2 = Clus[i]->GetDep()->GetCluster();
        if (c1 != nullptr) {
          c1->remove(Clus[i]);
        }
        c2->add(Clus[i]);
        Clus[i]->SetCluster(c2);
      }
    }
  }
}
double SESAME::DPTree::GetLastTime() { return lastTime; }
void SESAME::DPTree::SetLastTime(double last_time) { lastTime = last_time; }
int SESAME::DPTree::GetSize() { return size; }
void SESAME::DPTree::SetSize(int size) { DPTree::size = size; }
int SESAME::DPTree::GetNum() { return num; }
void SESAME::DPTree::SetNum(int num) { DPTree::num = num; }
std::vector<SESAME::DPNodePtr> &SESAME::DPTree::GetClus() { return Clus; }
void SESAME::DPTree::SetClus(std::vector<SESAME::DPNodePtr> &clus) {
  Clus = clus;
}
double SESAME::DPTree::GetA() { return a; }
void SESAME::DPTree::SetA(double a) { DPTree::a = a; }
double SESAME::DPTree::GetLamd() { return lamd; }
void SESAME::DPTree::SetLamd(double lamd) { DPTree::lamd = lamd; }
double SESAME::DPTree::GetCluR() { return CluR; }
void SESAME::DPTree::SetCluR(double clu_r) { CluR = clu_r; }
int SESAME::DPTree::GetCluLabel() { return cluLabel; }
void SESAME::DPTree::SetCluLabel(int clu_label) { cluLabel = clu_label; }
double SESAME::DPTree::GetMinDelta() { return minDelta; }
void SESAME::DPTree::SetMinDelta(double min_delta) { minDelta = min_delta; }
