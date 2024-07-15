//
// Created by tuidan on 2021/9/1.
//

#include <Algorithm/DesignAspect/V10.hpp>
#include <cfloat>

SESAME::V10::V10(param_t &cmd_params) {
  this->param = cmd_params;
  this->V10Param.num_points = cmd_params.num_points;
  this->V10Param.dim = cmd_params.dim;
  this->V10Param.alpha = cmd_params.alpha;
  this->V10Param.lamda = 0;
  this->V10Param.beta = cmd_params.beta;
  this->V10Param.num_cache = cmd_params.num_cache;
  this->V10Param.radius = cmd_params.radius;
  this->V10Param.minDelta = cmd_params.delta;
  this->V10Param.opt = cmd_params.opt;
  this->V10Param.landmark = cmd_params.landmark;
  sum_timer.Tick();
}

void SESAME::V10::Init() {
  this->cache = SESAME::DataStructureFactory::creatCache(
      this->V10Param.num_cache, this->V10Param.alpha, this->V10Param.lamda,
      this->V10Param.radius);
  this->outres = SESAME::DataStructureFactory::createOutlierReservoir(
      this->V10Param.radius, this->V10Param.alpha, this->V10Param.lamda);
  this->dpTree = SESAME::DataStructureFactory::createDPTree(
      this->actCluMaxNum, this->V10Param.radius);
  this->dpTree->SetMinDelta(this->V10Param.minDelta);
}

void SESAME::V10::OutputOnline(std::vector<PointPtr> &output) {
  ref_timer.Tick();
  auto clu = 0;
  for (const auto &center : this->onlineCenters) {
    center->setClusteringCenter(clu++);
    output.push_back(center->copy());
  }
  for (const auto &cluster : this->clusters) {
    std::unordered_set<DPNodePtr> cells = cluster->GetCells();
    for (const auto &cell : cells) {
      PointPtr center = cell->GetCenter();
      center->setClusteringCenter(clu++);
      center->setOutlier(false);
      output.push_back(center->copy());
    }
  }
}

void SESAME::V10::setMinDelta(double minDelta) {
  this->V10Param.minDelta = minDelta;
  this->dpTree->SetMinDelta(minDelta);
}

void SESAME::V10::InitDP(double time) {
  cache->compDeltaRho(time);
  this->minRho = this->V10Param.beta;

  outres->setTimeGap(INT64_MAX);
  cache->getDPTree(this->minRho, this->V10Param.minDelta, dpTree, outres,
                   clusters);
  dpTree->SetLastTime(time);
}

SESAME::DPNodePtr SESAME::V10::streamProcess(SESAME::PointPtr p, int opt,
                                             double time) {
  win_timer.Tick();
  double coef = pow(this->V10Param.alpha,
                    this->V10Param.lamda * (time - dpTree->GetLastTime()));
  dpTree->SetLastTime(time);
  win_timer.Tock();
  ds_timer.Tick();
  auto nn = dpTree->findNN(p, coef, opt, time);
  ds_timer.Tock();
  out_timer.Tick();
  //    if (nn == nullptr || nn->GetDis() > dpTree->GetCluR())
  //    {
  //        nn = outres->insert(p, time);
  //        if (nn->GetRho() > this->minRho)
  //        {
  //            outres->remove(nn);
  //            dpTree->insert(nn, opt);
  //        }
  //    }
  dpTree->deleteInact(outres, this->minRho, time);
  out_timer.Tock();
  return nn;
}
double SESAME::V10::computeAlpha() {
  return dpTree->computeAlpha(this->V10Param.minDelta);
}
double SESAME::V10::adjustMinDelta() {
  return dpTree->adjustMinDelta(this->alpha);
}
void SESAME::V10::delCluster() {
  for (auto it = this->clusters.begin(); it != this->clusters.end();) {
    auto cluster = it->get();
    if (cluster->GetCells().begin() == cluster->GetCells().end()) {
      this->clusters.erase(it++);
    } else {
      it++;
    }
  }
}

SESAME::DPNodePtr SESAME::V10::retrive(SESAME::PointPtr p, int opt,
                                       double time) {
  SESAME::PointPtr curP = p;
  if (!this->V10Param.isInit) {
    auto cc = cache->add(curP, time);
    if (cache->isFull()) {
      // draw decision graph
      InitDP(time);
      this->alpha = computeAlpha(); // TODO: what does it mean?
      this->V10Param.isInit = true;
    }
    return cc;
  } else {
    ds_timer.Tick();
    auto nn = streamProcess(curP, opt, time);
    this->dpTree->adjustCluster(clusters);
    ds_timer.Tock();

    out_timer.Tick();
    delCluster();
    out_timer.Tock();
    return nn;
  }
}

void SESAME::V10::CountNode(const SESAME::DPNodePtr &node, int &num) {
  num = num + 1;
  if (!node->GetSucs().empty()) {
    for (const SESAME::DPNodePtr &el : node->GetSucs()) {
      CountNode(el, num);
    }
  }
}
void SESAME::V10::RunOnline(SESAME::PointPtr input) {
  if (input->getIndex() != 0 and input->getIndex() % V10Param.landmark == 0) {
    for (const auto &cluster : this->clusters) {
      std::unordered_set<DPNodePtr> cells = cluster->GetCells();
      for (const auto &cell : cells) {
        PointPtr center = cell->GetCenter();
        center->setOutlier(false);
        onlineCenters.push_back(center->copy());
      }
    }
    //        for (const auto &out : this->outres->getOutliers())
    //        {
    //            PointPtr center = out->GetCenter();
    //            center->setOutlier(true);
    //            onlineCenters.push_back(center->copy());
    //        }
    DPTreePtr().swap(dpTree);
    OutPtr().swap(outres);
    CachePtr().swap(cache);
    std::unordered_set<ClusterPtr>().swap(clusters);
    Init();
    this->alpha = 0;
    this->minRho = 0;
    this->V10Param.isInit = false;
  }
  double curTime = input->index;
  auto c = retrive(input, this->V10Param.opt, curTime);
  if (input->getIndex() % 100 == 0 && this->V10Param.isInit) {
    ds_timer.Tick();
    setMinDelta(adjustMinDelta());
    this->dpTree->adjustCluster(this->clusters);
    ds_timer.Tock();
    out_timer.Tick();
    this->delCluster();
    out_timer.Tock();
  }
  lat_timer.Add(input->toa);
}
void SESAME::V10::RunOffline(SESAME::DataSinkPtr sinkPtr) {
  on_timer.Add(sum_timer.start);
  ref_timer.Tick();
  auto clu = 0;
  for (const auto &center : this->onlineCenters) {
    center->setClusteringCenter(clu++);
    sinkPtr->put(center->copy());
  }
  for (const auto &cluster : this->clusters) {
    std::unordered_set<DPNodePtr> cells = cluster->GetCells();
    for (const auto &cell : cells) {
      PointPtr center = cell->GetCenter();
      center->setClusteringCenter(clu++);
      center->setOutlier(false);
      sinkPtr->put(center->copy());
    }
  }
  //    for (const auto &out : this->outres->getOutliers())
  //    {
  //        PointPtr center = out->GetCenter();
  //        center->setClusteringCenter(clu++);
  //        center->setOutlier(true);
  //        sinkPtr->put(center->copy());
  //    }
  ref_timer.Tock();
  sum_timer.Tock();
}
