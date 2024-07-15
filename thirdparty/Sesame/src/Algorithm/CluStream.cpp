// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by 1124a on 2021/8/16.
//
#include <Algorithm/CluStream.hpp>
#include <Algorithm/DataStructure/DataStructureFactory.hpp>
#include <Algorithm/WindowModel/WindowFactory.hpp>

#include <iterator>

/**
 * @Description: "offline" init micro clusters using KMeans
 * @param size: The size of initial data objects,
 *  initialData:input initial data
 *@Return: void
 */
SESAME::CluStream::CluStream(param_t &cmd_params) {
  this->param = cmd_params;
  this->CluStreamParam.num_points = cmd_params.num_points;
  this->CluStreamParam.dim = cmd_params.dim;
  this->CluStreamParam.num_clusters = cmd_params.num_online_clusters;
  this->CluStreamParam.num_last_arr = cmd_params.num_last_arr;
  this->CluStreamParam.time_window = cmd_params.time_window;
  this->CluStreamParam.time_interval = cmd_params.time_interval;
  this->CluStreamParam.num_offline_clusters = cmd_params.k;
  this->CluStreamParam.radius = cmd_params.radius;
  this->CluStreamParam.seed = cmd_params.seed;
  this->CluStreamParam.buf_size = cmd_params.buf_size;
  if (this->CluStreamParam.offline_time_window >
      this->CluStreamParam.num_points)
    this->CluStreamParam.offline_time_window = cmd_params.offline_time_window;
  this->pointsFitted = 0;
  this->pointsForgot = 0;
  this->pointsMerged = 0;
}

SESAME::CluStream::~CluStream() {}

void SESAME::CluStream::initOffline(vector<PointPtr> &initData,
                                    vector<PointPtr> &initialData) {
  for (int i = 0; i < CluStreamParam.num_clusters; i++) {
    microClusters.push_back(
        DataStructureFactory::createMicroCluster(CluStreamParam.dim, i));
  }
  std::vector<PointPtr> centers;
  std::vector<std::vector<PointPtr>> oldGroups, newGroups;
  this->kmeans->runKMeans(CluStreamParam.num_clusters, CluStreamParam.buf_size,
                          centers, initData, oldGroups, newGroups,
                          CluStreamParam.seed, true);
  // store the result input output
  this->kmeans->storeResult(oldGroups, initialData);
  for (int i = 0; i < CluStreamParam.buf_size; i++) {
    int clusterId = initialData[i]->getClusteringCenter();
    // SESAME_INFO("the belonging micro cluster id is !"<<clusterId);

    int timestamp = initialData[i]->getIndex() - startTime;
    if (microClusters[clusterId]->weight == 0)
      microClusters[clusterId]->Init(initialData[i], timestamp);
    else
      microClusters[clusterId]->insert(initialData[i], timestamp);
  }
}

/**
 * @Description: incrementally insert data into micro clusters,
 * first determine closest clusters to the data,
 * then judge whether it locates into the maximum boundary of this micro cluster
 * 1. data object fits into closest micro cluster;
 * 2. data does not fit in any cluster,  delete oldest one & create a new
 * cluster;
 * 3.  merge two closest clusters & create a new cluster.
 * @Param: data: input data object
 * @Return: store the output result(with computed clustering center) into ???
 */
void SESAME::CluStream::incrementalCluster(
    PointPtr data) { // 1. Determine closest clusters
  MicroClusterPtr closestCluster;
  double minDistance = doubleMax;
  for (int i = 0; i < this->CluStreamParam.num_clusters; i++) {
    double dist = microClusters[i]->calCentroidDistance(data);
    if (dist < minDistance) {
      closestCluster = microClusters[i]->copy();
      minDistance = dist;
    }
  }
  double radius = calRadius(closestCluster);
  if (minDistance < radius) {
    insertIntoCluster(data, closestCluster);
    return;
  }
  /** 3. Date does not fit  -- free
   * some space to insert a new micro cluster
   * */
  // 3.1 delete oldest one & create a new cluster
  if (!deleteCreateCluster(data)) {
    // 3.2 merge two closest clusters & create a new cluster
    MergeCreateCluster(data);
  }
}

// Calculate and return the value of radius
double SESAME::CluStream::calRadius(MicroClusterPtr closestCluster) {
  double radius;
  if (closestCluster->weight == 1) {
    // Special case: estimate radius by determining the distance to the
    // next closest cluster
    radius = doubleMax;
    dataPoint centroid = closestCluster->getCentroid();
    for (int i = 0; i < this->CluStreamParam.num_clusters; i++) {
      if (microClusters[i]->id == closestCluster->id) {
        continue;
      }
      double dist = distance(microClusters[i]->getCentroid(), centroid,
                             this->CluStreamParam.dim);
      radius = std::min(dist, radius);
    }
  } else
    radius = closestCluster->getRadius(this->CluStreamParam.radius);
  return radius;
}

//   Date fits case
void SESAME::CluStream::insertIntoCluster(PointPtr data,
                                          MicroClusterPtr operateCluster) {
  // SESAME_INFO("This data fits");
  pointsFitted++;

  int timestamp = data->getIndex() - startTime;
  operateCluster->insert(data, timestamp);
}

// Delete the oldest cluster and create new one case
bool SESAME::CluStream::deleteCreateCluster(PointPtr data) {
  // 3.1 Try to forget old micro clusters

  int elapsedTime = data->getIndex();
  int threshold = 0; // Kernels before this can be forgotten
  if (elapsedTime - this->CluStreamParam.time_window >= 0)
    threshold = elapsedTime - this->CluStreamParam.time_window;
  for (int i = 0; i < this->CluStreamParam.num_clusters; i++) {
    if (microClusters[i]->getRelevanceStamp(this->CluStreamParam.num_last_arr) <
        threshold) {
      // SESAME_INFO("Need to delete");
      int newId =
          this->CluStreamParam.num_clusters + pointsForgot + pointsMerged;
      delMicroClusters.push_back(microClusters[i]);
      DataStructureFactory::clearMicroCluster(microClusters[i]);
      microClusters[i] =
          DataStructureFactory::createMicroCluster(CluStreamParam.dim, newId);
      microClusters[i]->Init(std::move(data), elapsedTime);
      pointsForgot++;

      return true;
    }
  }

  return false;
}

// Merge two closest clusters & create a new cluster

void SESAME::CluStream::MergeCreateCluster(PointPtr data) {
  // SESAME_INFO("Micro cluster needs to merge");
  unsigned int closestA = 0;
  unsigned int closestB = 0;
  double minDistance = doubleMax;
  for (int i = 0; i < this->CluStreamParam.num_clusters; i++) { // O(n(n+1)/2)
    dataPoint centroidA = microClusters[i]->getCentroid();
    for (int j = i + 1; j < this->CluStreamParam.num_clusters; j++) {
      double dist = distance(centroidA, microClusters[j]->getCentroid(),
                             this->CluStreamParam.dim);
      if (dist < minDistance) {
        minDistance = dist;
        closestA = i;
        closestB = j;
      }
    }
  }
  int newId = this->CluStreamParam.num_clusters + pointsForgot + pointsMerged;
  microClusters[closestA]->merge(microClusters[closestB]);
  int elapsedTime = data->getIndex();
  DataStructureFactory::clearMicroCluster(microClusters[closestB]);
  microClusters[closestB] =
      DataStructureFactory::createMicroCluster(CluStreamParam.dim, newId);
  microClusters[closestB]->Init(std::move(data), elapsedTime);
  pointsMerged++;
  return;
}

void SESAME::CluStream::microClusterToPoint(
    std::vector<MicroClusterPtr> &microClusters,
    vector<PointPtr> &points) const {
  for (int i = 0; i < this->CluStreamParam.num_clusters; i++) {
    PointPtr point =
        GenericFactory::New<Point>(microClusters[i]->centroid.size(), i);
    point->weight = microClusters[i]->weight;
    for (SESAME::dataPoint::size_type j = 0;
         j < microClusters[i]->centroid.size(); j++)
      point->setFeatureItem(microClusters[i]->centroid[j], j);
    // points;
    points.push_back(point);
  }
}
double SESAME::CluStream::distance(dataPoint a, dataPoint b, int dim) {
  double temp = 0;
  for (int i = 0; i < dim; i++) {
    double diff = b[i] - a[i];
    temp += diff * diff;
  }
  return sqrt(temp);
}

void SESAME::CluStream::Init() {
  this->window = WindowFactory::createLandmarkWindow();
  this->window->pyramidalWindow.time_interval =
      this->CluStreamParam.time_interval;
  this->lastUpdateTime = 0;
  window->initPyramidalWindow(this->window->pyramidalWindow.time_interval);
  sum_timer.Tick();
}

/**
 * @Description: online clustering of Clustream,
 * insert every data object and cluster them incrementally,it has three
 * conditions:
 * 1. data object fits into closest micro cluster;
 * 2. data does not fit in any cluster,  delete oldest one & create a new
 * cluster;
 * 3.  merge two closest clusters & create a new cluster.
 * @Param:dimension: dimension of data object
 * pointNumber: total number of data objects collected in workload
 * timeInterval: time interval of pyramidal window
 * lastArrivingNum: the last arriving  m points in each micro clusters
 * radiusFactor: radius factor of micro cluster
 * size: size of initial data objects
 * input: vector of data streams
 * @Return: store the output result(with computed clustering center) into ???//
 */
void SESAME::CluStream::RunOnline(SESAME::PointPtr input) {
  ds_timer.Tick();
  if (!this->initilized) {
    this->initialInputs.push_back(input->copy());
    this->startTime = initialInputs.at(0)->getIndex();
    if (this->initialInputs.size() == this->CluStreamParam.buf_size) {
      ds_timer.Tock();
      ref_timer.Tick();
      vector<PointPtr> initData; // initialData
      initOffline(this->initialInputs, initData);
      ref_timer.Tock();
      ds_timer.Tick();
      window->pyramidalWindowProcess(startTime, microClusters);
      this->initilized = true;
    }
  } else {
    int interval;
    interval = input->getIndex() - lastUpdateTime;

    // TODO we just assume take a snapshot per 1000 points
    if (interval >= 1000) {
      window->pyramidalWindowProcess(interval, microClusters);
      lastUpdateTime = input->getIndex();
    }
    incrementalCluster(input->copy());
  }
  ds_timer.Tock();
  lat_timer.Add(input->toa);
}

void SESAME::CluStream::RunOffline(SESAME::DataSinkPtr sinkPtr) {
  on_timer.Add(sum_timer.start);
  ref_timer.Tick();
  vector<vector<PointPtr>> groups;
  int elapsedTime = this->CluStreamParam.num_points;
  int landmarkTime = elapsedTime - this->CluStreamParam.offline_time_window;
  if (this->CluStreamParam.offline_time_window > elapsedTime)
    landmarkTime = 0;
  // SESAME_INFO("Start offline...");
  SESAME::SnapshotPtr landmarkSnapshot;
  SESAME::SnapshotPtr subtractMiroCluster;
  // If offline_time_window = 0, Only Observe the end results of micro clusters
  subtractMiroCluster =
      DataStructureFactory::createSnapshot(microClusters, elapsedTime);

  // The offline is to observe a process of data stream clustering
  if (CluStreamParam.offline_time_window != 0) { //
    landmarkSnapshot = SESAME::Snapshot::findSnapshot(
        window->orderSnapShots, landmarkTime, elapsedTime,
        window->pyramidalWindow.currentOrder);

    // SESAME_INFO("Landmark Miro Cluster is...");
    for (int i = 0; i < CluStreamParam.num_clusters; i++) {
      std::stringstream re2;
      std::copy(landmarkSnapshot->microClusters[i]->id.begin(),
                landmarkSnapshot->microClusters[i]->id.end(),
                std::ostream_iterator<int>(re2, " "));
      // SESAME_INFO("The ID is " << re2.str() << "weight is " <<
      // landmarkSnapshot->microClusters[i]->weight);
    }
    if (landmarkSnapshot->elapsedTime == -1)
      landmarkSnapshot = subtractMiroCluster;

    subtractMiroCluster = SESAME::Snapshot::substractSnapshot(
        subtractMiroCluster, landmarkSnapshot,
        this->CluStreamParam.num_clusters);
  }
  // SESAME_INFO("subtract Miro Cluster is...");
  for (int i = 0; i < CluStreamParam.num_clusters; i++) {
    std::stringstream re2;
    std::copy(subtractMiroCluster->microClusters[i]->id.begin(),
              subtractMiroCluster->microClusters[i]->id.end(),
              std::ostream_iterator<int>(re2, " "));
    // SESAME_INFO("The ID is " << re2.str() << "weight is " <<
    // subtractMiroCluster->microClusters[i]->weight);
  }
  vector<PointPtr> TransformedSnapshot;
  microClusterToPoint(subtractMiroCluster->microClusters, TransformedSnapshot);

  // SESAME_INFO("offline Cluster Number " <<
  // this->CluStreamParam.num_offline_clusters << "Total number of p: " <<
  // TransformedSnapshot.size());

  std::vector<PointPtr> centers;
  std::vector<std::vector<PointPtr>> oldGroups, newGroups;

  this->kmeans->Run(param, centers, sinkPtr);
  // this->kmeans->runKMeans(this->CluStreamParam.num_offline_clusters,
  // this->CluStreamParam.num_clusters,centers,
  //                         TransformedSnapshot, oldGroups, newGroups,
  //                         this->CluStreamParam.seed, true);
  // Count overall time

  // store the result input output
  // this->kmeans->produceResult(oldGroups,sinkPtr);
  // timerMeter.printTime(true, true,true,false);
  for (auto out = this->delMicroClusters.begin();
       out != this->delMicroClusters.end(); ++out) {
    PointPtr center = out->get()->getCenter();
    center->setClusteringCenter(-1);
    center->setOutlier(true);
    sinkPtr->put(center);
  }
  ref_timer.Tock();
  sum_timer.Tock();
}
