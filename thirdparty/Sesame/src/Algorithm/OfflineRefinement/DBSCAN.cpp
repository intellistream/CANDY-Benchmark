//
// Created by 1124a on 2021/8/20.
//
#include "Algorithm/OfflineRefinement/DBSCAN.hpp"
#include "Algorithm/Param.hpp"

SESAME::DBSCAN::DBSCAN(unsigned int minPts, float eps) {
  this->min_points = minPts;
  this->epsilon = eps;
  // this->pointSize = size;
  this->clusterID = 0;
}

SESAME::DBSCAN::DBSCAN() {
  this->min_points = 0;
  this->epsilon = 0;
  // this->pointSize = size;
  this->clusterID = 0;
}

SESAME::DBSCAN::~DBSCAN() = default;
void SESAME::DBSCAN::run(std::vector<PointPtr> &input) {
  for (auto &i : input) {
    i->setClusteringCenter(UNCLASSIFIED);
  }
  for (int i = 0; i < input.size(); i++) {
    if (input[i]->getClusteringCenter() == UNCLASSIFIED) {
      if (expandCluster(input, input[i], clusterID) != FAILURE) {
        clusterID += 1;
      }
    }
  }
}

void SESAME::DBSCAN::Run(SesameParam &param, std::vector<PointPtr> &input,
                         SESAME::DataSinkPtr sinkPtr) {
  this->min_points = param.min_points;
  this->epsilon = param.epsilon;
  // this->pointSize = size;
  for (auto &i : input) {
    i->setClusteringCenter(UNCLASSIFIED);
  }
  this->clusterID = 0;
  for (int i = 0; i < input.size(); i++) {
    if (input[i]->getClusteringCenter() == UNCLASSIFIED) {
      if (expandCluster(input, input[i], clusterID) != FAILURE) {
        clusterID += 1;
      }
    }
  }
  produceResult(input, sinkPtr);
}

int SESAME::DBSCAN::expandCluster(std::vector<PointPtr> &input, PointPtr &point,
                                  int clusterID) const {
  std::vector<int> clusterSeeds = calculateCluster(input, point);
  if (clusterSeeds.size() < min_points) {
    point->setClusteringCenter(NOISE);
    return FAILURE;
  } else {
    int index, indexCorePoint = 0;
    for (int iterSeeds = 0; iterSeeds < clusterSeeds.size(); iterSeeds++) {
      index = clusterSeeds.at(iterSeeds);

      input.at(index)->setClusteringCenter(clusterID);
      if (judgeCorePoint(
              input.at(index),
              point)) // check if the seed point in input is the core point?
        indexCorePoint = iterSeeds;
    }
    clusterSeeds.erase(clusterSeeds.begin() + indexCorePoint);

    for (std::vector<int>::size_type i = 0, currentSize = clusterSeeds.size();
         i < currentSize; i++) // ++i or i++?
    {
      std::vector<int> clusterNeighbors =
          calculateCluster(input, input.at(clusterSeeds[i]));
      if (clusterNeighbors.size() >= min_points) {
        for (std::vector<int>::size_type iterNeighbors = 0;
             iterNeighbors < clusterNeighbors.size(); iterNeighbors++) {
          index = clusterNeighbors.at(iterNeighbors);
          if (input.at(index)->getClusteringCenter() == UNCLASSIFIED ||
              input.at(index)->getClusteringCenter() == NOISE) {
            if (input.at(index)->getClusteringCenter() == UNCLASSIFIED) {
              clusterSeeds.push_back(index);
              currentSize = clusterSeeds.size();
            }
            input.at(index)->setClusteringCenter(clusterID);
          }
        }
      }
    }
    return SUCCESS;
  }
}

std::vector<int> SESAME::DBSCAN::calculateCluster(std::vector<PointPtr> &input,
                                                  PointPtr &point) const {
  std::vector<int> clusterIndex;
  for (int i = 0; i < input.size(); i++) {
    if (calculateEluDistance(point, input[i]) <= epsilon)
      clusterIndex.push_back(i);
  }
  return clusterIndex;
}

double SESAME::DBSCAN::calculateEluDistance(PointPtr &point, PointPtr &center) {
  double dist = 0;
  for (int i = 0; i < point->getDimension(); i++) {
    dist += pow(point->getFeatureItem(i) - center->getFeatureItem(i), 2);
  }
  dist = sqrt(dist);
  return dist;
}

bool SESAME::DBSCAN::judgeCorePoint(PointPtr &point, PointPtr &other) {
  bool corePoint = true;
  for (int i = 0; i < point->getDimension(); i++) {
    if (point->getFeatureItem(i) != other->getFeatureItem(i))
      corePoint = false;
  }
  return corePoint;
}

// TODO: whether to output the noise data? In our implementation, we output the
// noise data since it is still a cluster
void SESAME::DBSCAN::produceResult(std::vector<PointPtr> &input,
                                   SESAME::DataSinkPtr sinkPtr) {
  for (auto el : input) {
    if (el->getClusteringCenter() == UNCLASSIFIED or
        NOISE) { // not noise or unclassified
      el->setClusteringCenter(clusterID++);
    }
    sinkPtr->put(el); // point index start from 0
  }
}
