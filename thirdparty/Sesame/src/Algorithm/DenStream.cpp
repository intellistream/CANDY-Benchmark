
//
// Created by 1124a on 2021/8/23.
//
#include "Algorithm/DenStream.hpp"
#include "Algorithm/DataStructure/DataStructureFactory.hpp"
#include "Algorithm/WindowModel/WindowFactory.hpp"

SESAME::DenStream::DenStream(param_t &cmd_params) {
  this->param = cmd_params;
  this->denStreamParams.num_points = cmd_params.num_points;
  this->denStreamParams.dim = cmd_params.dim;
  this->denStreamParams.min_points = cmd_params.min_points;
  this->denStreamParams.epsilon = cmd_params.epsilon;
  this->denStreamParams.base = cmd_params.base;
  this->denStreamParams.lambda = cmd_params.lambda;
  this->denStreamParams.mu = cmd_params.mu;
  this->denStreamParams.beta = cmd_params.beta;
  this->denStreamParams.buf_sizeSize = cmd_params.buf_size;
}
SESAME::DenStream::~DenStream() {}

void SESAME::DenStream::Init(vector<PointPtr> &initData) {
  this->pMicroClusterIndex = -1;
  this->oMicroClusterIndex = -1;
  for (int i = 0; i < denStreamParams.buf_sizeSize; i++) {
    if (initData.at(i)->getClusteringCenter() == noVisited) {
      std::vector<int> pointIndex;
      pMicroClusterIndex++;
      MicroClusterPtr newMicroCluster =
          SESAME::DataStructureFactory::createMicroCluster(denStreamParams.dim,
                                                           pMicroClusterIndex);
      newMicroCluster->Init(initData.at(i), 0);
      pointsNearCorePoint(initData, pointIndex, newMicroCluster);
      if (newMicroCluster->weight <=
          this->minWeight) // TODO need to change minweight
      {
        pMicroClusterIndex--;
        for (int index : pointIndex) {
          initData.at(index)->setClusteringCenter(noVisited);
        }
      } else
        pMicroClusters.push_back(newMicroCluster);
    }
  }
  SESAME_INFO("NOW PMC number is: " << this->pMicroClusterIndex
                                    << " , Init succeed!");
}
void SESAME::DenStream::pointsNearCorePoint(vector<PointPtr> &initData,
                                            std::vector<int> pointIndex,
                                            MicroClusterPtr microCluster) {
  int size = denStreamParams.buf_sizeSize;
  for (int i = 0; i < size; i++) {
    if (initData.at(i)->getClusteringCenter() == noVisited) {
      double dist = microCluster->calCentroidDistance(initData.at(i));
      if (dist < denStreamParams.epsilon) {
        initData[i]->setClusteringCenter(microCluster->id.front());
        microCluster->insert(initData.at(i), 0);
        pointIndex.push_back(i);
      }
    }
  }
}
void SESAME::DenStream::Init() {
  this->dampedWindow = WindowFactory::createDampedWindow(
      denStreamParams.base, denStreamParams.lambda);
  this->dbscan = std::make_shared<DBSCAN>(denStreamParams.min_points,
                                          denStreamParams.epsilon);
  this->startTime = 0;
  this->lastUpdateTime = 0;
  this->pointArrivingTime = 0;
  this->minWeight = denStreamParams.beta * denStreamParams.mu;
  this->Tp = (double)(1 / denStreamParams.lambda) *
             (log(minWeight / (minWeight - 1)) / log(denStreamParams.base));
  if (this->Tp > 1000 || this->Tp <= 0)
    this->Tp = 1;
  sum_timer.Tick();
}
void SESAME::DenStream::RunOnline(PointPtr in) {
  PointPtr input = in;

  if (!this->isInitial) {
    ds_timer.Tick();
    input->setClusteringCenter(noVisited);
    this->initialBuffer.push_back(input);
    if (this->initialBuffer.size() == this->denStreamParams.buf_sizeSize) {
      // Initialize part
      Init(this->initialBuffer);
      this->isInitial = true;
    }
    ds_timer.Tock();
  } else {
    this->pointArrivingTime = input->getIndex();
    merge(input);
    int elapsedTime = this->pointArrivingTime - this->lastUpdateTime;

    if (elapsedTime >= this->Tp) {
      ds_timer.Tick();
      // timerMeter.outlierDetectionAccMeasure();
      // SESAME_INFO("Check "<<elapsedTime);

      for (int iter = 0; iter < pMicroClusters.size(); iter++) {
        if (pMicroClusters.at(iter)->weight < minWeight) {
          pMicroClusters.erase(pMicroClusters.begin() + iter);
          // SESAME_INFO("NOW PMC number is: " << this->pMicroClusterIndex);
        }
      }
      // timerMeter.outlierDetectionEndMeasure();

      if (!oMicroClusters.empty()) {
        for (int iter = 0; iter < oMicroClusters.size(); iter++) {
          double a = -(denStreamParams.lambda) *
                     (pointArrivingTime - oMicroClusters.at(iter)->createTime +
                      this->Tp);
          double b = -denStreamParams.lambda * this->Tp;
          double Xi = (pow(denStreamParams.base, a) - 1) /
                      (pow(denStreamParams.base, b) - 1);
          // SESAME_INFO("NOW Xi  "<<Xi);
          if (oMicroClusters.at(iter)->weight < Xi) {
            oMicroClusters.erase(oMicroClusters.begin() + iter);
            // SESAME_INFO("NOW oMicroClusterIndex number is: " <<
            // oMicroClusters.size());
          }
        }
      }
      ds_timer.Tock();

      this->lastUpdateTime = this->pointArrivingTime;
    }
    this->lastPointTime = this->pointArrivingTime;
  }
  lat_timer.Add(input->toa);
}

void SESAME::DenStream::merge(PointPtr dataPoint) {
  bool index = false;
  if (!this->pMicroClusters.empty()) {
    index = mergeToMicroCluster(dataPoint, this->pMicroClusters);
    //  std::cout<<"Merge into PMC! "<<pMicroClusters.size()<<","<<
    //  index<<","<<std::endl;
  }

  if (!index && !this->oMicroClusters.empty()) {
    // Time measurement inside the mergeToOMicroCluster function
    index = mergeToOMicroCluster(dataPoint, this->oMicroClusters);
    // std::cout<<"Merge into OMC! "<<oMicroClusters.size()<<","<<
    // index<<","<<std::endl;
  }

  out_timer.Tick();
  if (!index) {
    oMicroClusterIndex++;
    MicroClusterPtr newOMicroCluster = DataStructureFactory::createMicroCluster(
        denStreamParams.dim, oMicroClusterIndex);
    newOMicroCluster->Init(dataPoint, 0);
    oMicroClusters.push_back(newOMicroCluster);
  }
  out_timer.Tock();
}

bool SESAME::DenStream::mergeToMicroCluster(
    PointPtr dataPoint, std::vector<MicroClusterPtr> microClusters) {
  ds_timer.Tick();
  bool index = false;
  MicroClusterPtr MC = nearestNeighbor(dataPoint, microClusters);
  ds_timer.Tock();
  win_timer.Tick();
  double decayFactor =
      this->dampedWindow->decayFunction(lastPointTime, pointArrivingTime);
  if (MC != NULL &&
      MC->insert(dataPoint, decayFactor, denStreamParams.epsilon)) {
    index = true;
  }
  win_timer.Tock();
  return index;
}

bool SESAME::DenStream::mergeToOMicroCluster(
    PointPtr dataPoint, std::vector<MicroClusterPtr> microClusters) {
  out_timer.Tick();
  MicroClusterPtr MC = nearestNeighbor(dataPoint, microClusters);
  out_timer.Tock();
  win_timer.Tick();
  double decayFactor =
      this->dampedWindow->decayFunction(lastPointTime, pointArrivingTime);
  if (MC != NULL &&
      MC->insert(dataPoint, decayFactor, denStreamParams.epsilon)) {
    double decayValue = this->dampedWindow->decayFunction(MC->lastUpdateTime,
                                                          pointArrivingTime);
    win_timer.Tock();
    ds_timer.Tick();
    if (MC->weight * decayValue > minWeight) {
      pMicroClusterIndex++;
      MC->resetID(pMicroClusterIndex);
      pMicroClusters.push_back(MC);
      int index = findIndex(oMicroClusters, MC);
      oMicroClusters.erase(oMicroClusters.begin() + index);
      // std::remove(oMicroClusters.begin(), oMicroClusters.end(), MC);
    }
    ds_timer.Tock();
    return true;
  } else {
    win_timer.Tock();
    return false;
  }
}
int SESAME::DenStream::findIndex(std::vector<MicroClusterPtr> &microClusters,
                                 MicroClusterPtr MC) {
  auto ret = std::find(microClusters.begin(), microClusters.end(), MC);
  if (ret != microClusters.end())
    return ret - microClusters.begin();
  return -1;
}

SESAME::MicroClusterPtr
SESAME::DenStream::nearestNeighbor(PointPtr dataPoint,
                                   std::vector<MicroClusterPtr> microClusters) {
  MicroClusterPtr targetMC = NULL;
  double dist = 0, minDist = std::numeric_limits<double>::max();
  for (vector<MicroClusterPtr>::size_type i = 0; i < microClusters.size();
       i++) {
    dist = microClusters.at(i)->calCentroidDistance(dataPoint);
    if (dist < minDist) {
      minDist = dist;
      targetMC = microClusters.at(i);
    }
  }

  return targetMC;
}

void SESAME::DenStream::RunOffline(DataSinkPtr sinkPtr) {
  on_timer.Add(sum_timer.start);
  ref_timer.Tick();
  vector<PointPtr> transformedPoints;
  std::vector<std::vector<PointPtr>> oldGroups;
  microClusterToPoint(pMicroClusters, transformedPoints);
  this->dbscan->run(transformedPoints);
  this->dbscan->produceResult(transformedPoints, sinkPtr);
  for (auto out = this->oMicroClusters.begin();
       out != this->oMicroClusters.end(); ++out) {
    PointPtr center = out->get()->getCenter();
    center->setClusteringCenter(-1);
    center->setOutlier(true);
    sinkPtr->put(center->copy());
  }
  ref_timer.Tock();
  sum_timer.Tock();
}
void SESAME::DenStream::microClusterToPoint(
    std::vector<MicroClusterPtr> &microClusters, vector<PointPtr> &points) {
  for (std::vector<MicroClusterPtr>::size_type i = 0; i < microClusters.size();
       i++) {
    PointPtr point =
        GenericFactory::New<Point>(microClusters.at(i)->centroid.size(), i);
    point->weight = microClusters.at(i)->weight;
    for (int j = 0; j < microClusters.at(i)->centroid.size(); j++)
      point->setFeatureItem(microClusters[i]->centroid[j], j);
    points.push_back(point);
  }
}
