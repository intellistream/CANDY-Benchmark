//
// Created by Zhenyu on 2021/8/30.
//
#include <Algorithm/DBStream.hpp>
#include <Algorithm/WindowModel/WindowFactory.hpp>

/**
 * @Description: initialize user defined parameters,
 * @Param:
 * radius: radius of micro clusters
 * lambda: lambda in decay function
 * clean_interval: time gap of clean up
 * min_weight: the minimum weight of micro cluster to identify noise MCs
 * alpha: intersection factor
 * base: decay function base -- Normally 2
 * @Return: void
 */
SESAME::DBStream::DBStream(param_t &cmd_params) {
  this->param = cmd_params;
  this->dbStreamParams.num_points = cmd_params.num_points;
  this->dbStreamParams.dim = cmd_params.dim;
  this->dbStreamParams.radius = cmd_params.radius;
  this->dbStreamParams.lambda = cmd_params.lambda;
  this->dbStreamParams.clean_interval = cmd_params.clean_interval;
  this->dbStreamParams.min_weight = cmd_params.min_weight;
  this->dbStreamParams.alpha = cmd_params.alpha;
  this->dbStreamParams.base = cmd_params.base;
}
SESAME::DBStream::~DBStream() = default;

/**
 * @Description: initialization of the algorithm,
 * @Param: void
 * @Return: void
 */
void SESAME::DBStream::Init() {
  this->dampedWindow = WindowFactory::createDampedWindow(dbStreamParams.base,
                                                         dbStreamParams.lambda);
  this->pointArrivingTime = 0;
  this->lastCleanTime = 0;
  this->lastArrivingTime = 0;
  this->weakEntry = pow(dbStreamParams.base, (-1) * dbStreamParams.lambda *
                                                 dbStreamParams.clean_interval);
  this->aWeakEntry = weakEntry * dbStreamParams.alpha;

  // std::cout<<"weakEntry"<<weakEntry<<std::endl;
  // std::cout<<"aWeakEntry"<<aWeakEntry<<std::endl;
  this->microClusterIndex = -1;
  connectedRegions =
      ConnectedRegions(dbStreamParams.alpha, dbStreamParams.min_weight);
  sum_timer.Tick();
}
/**
 * @Description: online clustering stage, input data point incrementally and
 * update the MC list and weight adjacency lists,
 * @Param: void
 * @Return: void
 */
void SESAME::DBStream::RunOnline(PointPtr input) {
  if (!this->isInitial) {
    // SESAME_INFO("Start initialize...");
    Init();
    this->isInitial = true;
    update(input);
  } else {
    update(input);
    lastArrivingTime = pointArrivingTime;
    lastArrivingTime0 = pointArrivingTime0;
  }
  lat_timer.Add(input->toa);
}

void SESAME::DBStream::RunOffline(DataSinkPtr sinkPtr) {
  on_timer.Add(sum_timer.start);
  ref_timer.Tick();
  SESAME_INFO("micro clusters " << microClusters.size());
  SESAME_INFO("weightedAdjacencyList  " << weightedAdjacencyList.size());
  //  std::cout<<"micro clusters "<<microClusters.size()<<std::endl;
  //  std::cout<<"weightedAdjacencyList
  //  "<<weightedAdjacencyList.size()<<std::endl;
  connectedRegions.connection(microClusters, weightedAdjacencyList);
  std::vector<PointPtr> points = connectedRegions.ResultsToDataSink();
  for (auto i = 0; i < points.size(); i++) {
    auto res = points[i];
    res->setClusteringCenter(i);
    sinkPtr->put(res);
  }

  // timerMeter.printTime(false,false,true,false);
  ref_timer.Tock();
  sum_timer.Tock();
}
/**
 * @Description: Insert data point into existing MCs,
 * first find the MCs which data point locates in, if finding no MCs,
 * create new MC with this data point, else if finding MCs can accept this data
 * update these MCs and the corresponding Sij in Weighted adjacency list S;
 * After inserting, we check whether moving center of MCs will collapse,
 * if it will, we roll back moving center actions,
 * finally, we clean up the MCs which is less than Wmin
 * @Param: data point
 * @Return: void
 */
void SESAME::DBStream::update(PointPtr dataPoint) {
  win_timer.Tick();
  this->pointArrivingTime = dataPoint->getIndex();
  clock_gettime(CLOCK_REALTIME, &this->pointArrivingTime0);
  double decayFactor =
      dampedWindow->decayFunction(lastArrivingTime, this->pointArrivingTime);
  // TODO this one is using timespec to calculate time
  // double decayFactor0 =
  // dampedWindow->decayFunction(lastArrivingTime,this->pointArrivingTime);
  this->microClusterNN =
      findFixedRadiusNN(dataPoint, decayFactor); // decayFactor
  std::vector<MicroClusterPtr>::size_type sizeNN = microClusterNN.size();
  win_timer.Tock();

  /* *
   * If this point fits in no micro clusters
   * */
  if (microClusterNN.empty()) {
    ds_timer.Tick();
    microClusterIndex++;
    MicroClusterPtr newMicroCluster =
        SESAME::DataStructureFactory::createMicroCluster(
            dbStreamParams.dim, microClusterIndex, dataPoint,
            dbStreamParams.radius);
    microClusters.push_back(newMicroCluster);
    microClusterNN.push_back(newMicroCluster);
    ds_timer.Tock();
  } else {
    // SESAME_INFO("microClusterNN size " << sizeNN);
    for (int i = 0; i < sizeNN; i++) {
      ds_timer.Tick();
      microClusterNN[i]->insert(dataPoint); // just update weight //
      ds_timer.Tock();
      // std::cout<<" cluster "<<microClusterNN[i]->id.front()<<"th weight is
      // "<<microClusterNN[i]->weight<<std::endl;
      for (int j = i + 1; j < sizeNN; j++) {
        ds_timer.Tick();
        MicroClusterPair microClusterPair(microClusterNN[i],
                                          microClusterNN.at(j));
        ds_timer.Tock();
        win_timer.Tick();
        if (weightedAdjacencyList.find(microClusterPair) !=
            weightedAdjacencyList.end()) {
          // SESAME_INFO("update Sij");
          // update existing micro cluster pair in the graph
          int startT = weightedAdjacencyList[microClusterPair]->updateTime;
          double decayValue =
              dampedWindow->decayFunction(startT, this->pointArrivingTime);
          // Timespec
          // timespec startT=
          // weightedAdjacencyList[microClusterPair]->updateTime0; double
          // decayValue0 =
          // dampedWindow->decayFunction(startT,this->pointArrivingTime0);
          weightedAdjacencyList[microClusterPair]->add(this->pointArrivingTime,
                                                       decayValue);
        } else {
          // SESAME_INFO("insert Sij");
          AdjustedWeightPtr adjustedWeight =
              SESAME::DataStructureFactory::createAdjustedWeight(
                  1, this->pointArrivingTime, this->pointArrivingTime0);
          DensityGraph densityGraph(microClusterPair, adjustedWeight);
          weightedAdjacencyList.insert(densityGraph);
        }
        win_timer.Tock();
      }
    }
    ds_timer.Tick();
    if (checkMove(microClusterNN))
      for (const MicroClusterPtr &microCluster : microClusterNN)
        microCluster->move();
    ds_timer.Tock();
  }
  out_timer.Tick();

  // if (((pointArrivingTime-this->lastCleanTime)/CLOCKS_PER_SEC)>=
  // dbStreamParams.clean_interval
  // && dataPoint->getIndex()!=0)
  if ((pointArrivingTime) % dbStreamParams.clean_interval == 0)
  // if(interval/1000L%dbStreamParams.clean_interval==0 )
  {
    cleanUp(pointArrivingTime); // pointArrivingTime

    this->lastCleanTime = this->pointArrivingTime;
  }
  out_timer.Tock();
}

std::vector<SESAME::MicroClusterPtr>
SESAME::DBStream::findFixedRadiusNN(PointPtr dataPoint, double decayFactor) {
  std::vector<SESAME::MicroClusterPtr> result;
  std::vector<SESAME::MicroClusterPtr>::size_type iter;
  // todo this is a test for time
  for (iter = 0; iter < microClusters.size(); iter++) {
    microClusters.at(iter)->decayWeight(decayFactor);
    double distance = microClusters.at(iter)->getDistance(dataPoint);
    // SESAME_INFO("distance " << distance)
    if (distance < dbStreamParams.radius)
      result.push_back(microClusters.at(iter));
  }
  return result;
}

bool SESAME::DBStream::checkMove(
    std::vector<MicroClusterPtr> microClustersList) const {
  if (!microClustersList.empty()) {
    std::vector<MicroClusterPtr>::size_type i, j;
    for (i = 0; i < microClustersList.size(); i++) {
      for (j = i + 1; j < microClustersList.size(); j++) {
        double distance =
            microClustersList.at(i)->getDistance(microClustersList.at(j));
        if (distance < dbStreamParams.radius)
          return false;
      }
    }
  }
  return true;
}

void SESAME::DBStream::cleanUp(int nowTime) {
  std::vector<MicroClusterPtr> removeMicroCluster;
  std::vector<MicroClusterPtr>::size_type iter;
  // Check the current micro Clusters whether they have weak MCs
  // This just test for remove id
  std::vector<int> idList;
  for (iter = 0; iter < microClusters.size(); iter++) {
    if (microClusters.at(iter)->weight <= this->weakEntry) {
      removeMicroCluster.push_back(microClusters.at(iter)->copy());
      idList.push_back(microClusters.at(iter)->id.front());
      microClusters.erase(microClusters.begin() +
                          int(iter)); // Delete this MC from current MC list
    }
  }
  auto iterW = weightedAdjacencyList.begin();
  while (iterW != weightedAdjacencyList.end()) {
    auto val1 = iterW->first.microCluster1->id.front(),
         val2 = iterW->first.microCluster2->id.front();
    auto exist1 = std::find_if(
        removeMicroCluster.begin(), removeMicroCluster.end(),
        [&](const MicroClusterPtr &mc) { return mc->id.front() == val1; });
    auto exist2 = std::find_if(
        removeMicroCluster.begin(), removeMicroCluster.end(),
        [&](const MicroClusterPtr &mc) { return mc->id.front() == val2; });
    if (exist1 != removeMicroCluster.end() ||
        exist2 != removeMicroCluster.end()) {
      iterW = weightedAdjacencyList.erase(iterW);
    } else {
      double decayFactor =
          dampedWindow->decayFunction(iterW->second->updateTime, nowTime);
      // double
      // decayFactor0=dampedWindow->decayFunction(iterW->second->updateTime0,nowTime);
      if (iterW->second->getCurrentWeight(decayFactor) < aWeakEntry)
        iterW = weightedAdjacencyList.erase(iterW);
      else
        iterW++;
    }
  }
  //  SESAME_INFO("CLEAN! now weightedAdjacencyList
  //  size:"<<weightedAdjacencyList.size());
}
