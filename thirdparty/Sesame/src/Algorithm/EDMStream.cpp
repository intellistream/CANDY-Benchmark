//
// Created by tuidan on 2021/9/1.
//

#include <Algorithm/EDMStream.hpp>

SESAME::EDMStream::EDMStream(param_t &cmd_params) {
  this->param = cmd_params;
  this->EDMParam.num_points = cmd_params.num_points;
  this->EDMParam.dim = cmd_params.dim;
  this->EDMParam.alpha = cmd_params.alpha;
  this->EDMParam.lamda = cmd_params.lambda;
  this->EDMParam.beta = cmd_params.beta;
  this->EDMParam.num_cache = cmd_params.num_cache;
  this->EDMParam.radius = cmd_params.radius;
  this->EDMParam.minDelta = cmd_params.delta;
  this->EDMParam.opt = cmd_params.opt;
}

void SESAME::EDMStream::Init() {
  this->alpha = 0;
  this->cache = SESAME::DataStructureFactory::creatCache(
      this->EDMParam.num_cache, this->EDMParam.alpha, this->EDMParam.lamda,
      this->EDMParam.radius);
  this->outres = SESAME::DataStructureFactory::createOutlierReservoir(
      this->EDMParam.radius, this->EDMParam.alpha, this->EDMParam.lamda);
  this->dpTree = SESAME::DataStructureFactory::createDPTree(
      this->actCluMaxNum, this->EDMParam.radius);
  this->dpTree->SetMinDelta(this->EDMParam.minDelta);
  sum_timer.Tick();
}

void SESAME::EDMStream::setMinDelta(double minDelta) {
  this->EDMParam.minDelta = minDelta;
  this->dpTree->SetMinDelta(minDelta);
}

void SESAME::EDMStream::InitDP(double time) {
  cache->compDeltaRho(time);
  SESAME_DEBUG("beta = " << this->EDMParam.beta);
  this->minRho = this->EDMParam.beta /
                 (1 - pow(this->EDMParam.alpha, this->EDMParam.lamda));
  SESAME_DEBUG("minRho = " << this->minRho);

  this->deltaT = (log(1 - pow(this->EDMParam.alpha, this->EDMParam.lamda)) /
                      log(this->EDMParam.alpha) -
                  log(this->EDMParam.beta) / log(this->EDMParam.alpha)) /
                 this->EDMParam.lamda;
  //		double deltaT = 100;
  SESAME_DEBUG("deltaT = " << this->deltaT);
  outres->setTimeGap(this->deltaT);
  cache->getDPTree(this->minRho, this->EDMParam.minDelta, dpTree, outres,
                   clusters);
  SESAME_DEBUG("dpTree size = " << dpTree->GetSize());
  dpTree->SetLastTime(time);
}

SESAME::DPNodePtr SESAME::EDMStream::streamProcess(SESAME::PointPtr p, int opt,
                                                   double time) {
  win_timer.Tick();
  double coef = pow(this->EDMParam.alpha,
                    this->EDMParam.lamda * (time - dpTree->GetLastTime()));
  dpTree->SetLastTime(time);
  win_timer.Tock();
  ds_timer.Tick();
  auto nn = dpTree->findNN(p, coef, opt, time);
  ds_timer.Tock();
  out_timer.Tick();
  if (nn == nullptr || nn->GetDis() > dpTree->GetCluR()) {
    nn = outres->insert(p, time);
    if (nn->GetRho() > this->minRho) {
      outres->remove(nn);
      dpTree->insert(nn, opt);
    }
  }
  dpTree->deleteInact(outres, this->minRho, time);
  out_timer.Tock();
  return nn;
}
double SESAME::EDMStream::computeAlpha() {
  return dpTree->computeAlpha(this->EDMParam.minDelta);
}
double SESAME::EDMStream::adjustMinDelta() {
  return dpTree->adjustMinDelta(this->alpha);
}
void SESAME::EDMStream::delCluster() {
  for (auto it = this->clusters.begin(); it != this->clusters.end();) {
    auto cluster = it->get();
    if (cluster->GetCells().begin() == cluster->GetCells().end()) {
      this->clusters.erase(it++);
    } else {
      it++;
    }
  }
}

SESAME::DPNodePtr SESAME::EDMStream::retrive(SESAME::PointPtr p, int opt,
                                             double time) {
  SESAME::PointPtr curP = p;
  if (!this->EDMParam.isInit) {
    auto cc = cache->add(curP, time);
    if (cache->isFull()) {
      // draw decision graph
      InitDP(time);
      this->alpha = computeAlpha(); // TODO: what does it mean?
      SESAME_DEBUG("alpha = " << this->alpha);
      this->EDMParam.isInit = true;
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

void SESAME::EDMStream::CountNode(const SESAME::DPNodePtr &node, int &num) {
  num = num + 1;
  if (!node->GetSucs().empty()) {
    for (const SESAME::DPNodePtr &el : node->GetSucs()) {
      CountNode(el, num);
    }
  }
}
void SESAME::EDMStream::RunOnline(SESAME::PointPtr input) {
  double curTime = input->index;
  auto c = retrive(input, this->EDMParam.opt, curTime);
  if (input->getIndex() % 100 == 0 && this->EDMParam.isInit) {
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
void SESAME::EDMStream::RunOffline(SESAME::DataSinkPtr sinkPtr) {
  on_timer.Add(sum_timer.start);
  ref_timer.Tick();
  int num = 0;
  int sum = 0;
  auto clu = 0;
  for (const auto &cluster : this->clusters) {
    sum += num;
    num = 0;
    std::unordered_set<DPNodePtr> cells = cluster->GetCells();
    for (const auto &cell : cells) {
      CountNode(cell->copy(), num);
      PointPtr center = cell->GetCenter();
      center->setClusteringCenter(clu++);
      center->setOutlier(false);
      sinkPtr->put(center->copy());
    }
  }
  for (const auto &out : this->outres->getOutliers()) {
    sum += num;
    num = 0;
    CountNode(out->copy(), num);
    PointPtr center = out->GetCenter();
    center->setClusteringCenter(clu++);
    center->setOutlier(true);
    sinkPtr->put(center->copy());
  }
  ref_timer.Tock();
  sum_timer.Tock();
}
