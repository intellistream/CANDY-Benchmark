// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by 1124a on 2021/8/16.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_MICROCLUSTER_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_MICROCLUSTER_HPP_
#include <Algorithm/DataStructure/Point.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
namespace SESAME {

typedef std::vector<double> dataPoint;

class MicroCluster;

typedef std::shared_ptr<MicroCluster> MicroClusterPtr;
class MicroCluster {
public:
  constexpr static const double EPSILON = 0.00005;
  constexpr static const double MIN_VARIANCE = 1e-50; // TODO Need to move ...

  std::vector<int> id;
  dataPoint centroid;
  dataPoint LS;  // the sum of the data values for each dim
  dataPoint SS;  // the sum of the squares of  the data values for each dim
  int LST;       // the sum of the time stamps Ti~... Tin
  int SST;       // the sum of the squares of the time stamps  Til... Tin
  double weight; // number of data point in the clusters
  int dim;
  double radius; // Used in DBStream
  // the parameters below is unique for DenStream
  int createTime;
  int lastUpdateTime;
  bool visited;

  // TODO 1. Need to subtract Base class of CF vector when all cf-vector
  // based-algorithms have been implemented
  //  2.this may need to modify in the future (All algorithms used this,
  //  e.g.DenStream,CluStream,DenStream,DBStream,SWEM =.=)

  MicroCluster(int dim, int id);
  MicroCluster(int dim, int id, PointPtr dataPoint, double radius); // DBStream

  ~MicroCluster();
  void Init(PointPtr datapoint, int timestamp);
  void insert(PointPtr datapoint, int timestamp); // Used in CluStream
  bool insert(PointPtr datapoint, double decayFactor,
              double epsilon);     // DenStream
  void insert(PointPtr datapoint); // DBStream //, double decayFactor
  void merge(MicroClusterPtr other);
  void subtractClusterVector(MicroClusterPtr other);
  void updateId(MicroClusterPtr other);

  void resetID(int index);                          // Used in DenStream
  double getRadius(double decayFactor, bool judge); // Used in DenStream

  double getRelevanceStamp(int num_last_arr) const;
  double getMutime() const;
  double getSigmaTime() const;
  static double getQuantile(double z);
  double getRadius(double radius);
  double getDeviation();
  dataPoint getCentroid();
  PointPtr getCenter();
  double getInclusionProbability(PointPtr datapoint, double radius);
  dataPoint getVarianceVector();
  double calCentroidDistance(PointPtr datapoint);
  bool judgeMerge(MicroClusterPtr other);
  double getDistance(PointPtr datapoint);    // DBStream
  double getDistance(MicroClusterPtr other); // DBStream
  void move();                               // DBStream
  void decayWeight(double decayFactor);
  SESAME::MicroClusterPtr copy();

private:
  double distance;
  static double inverseError(double x);
};

} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_MICROCLUSTER_HPP_
