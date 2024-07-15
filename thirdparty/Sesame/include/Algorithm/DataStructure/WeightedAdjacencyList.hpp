//
// Created by 1124a on 2021/8/30.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_WEIGHTEDADJACENCYLIST_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_WEIGHTEDADJACENCYLIST_HPP_

#include "Algorithm/DataStructure/MicroCluster.hpp"
#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/WindowModel/DampedWindow.hpp"

#include <algorithm>
#include <functional>
#include <unordered_map>

namespace SESAME {
struct MicroClusterPair;
typedef std::shared_ptr<MicroClusterPair> MicroClusterPairPtr;
struct MicroClusterPair {
  MicroClusterPtr microCluster1;
  MicroClusterPtr microCluster2;
  MicroClusterPair(MicroClusterPtr microCluster1,
                   MicroClusterPtr microCluster2) {
    this->microCluster1 = microCluster1->copy();
    this->microCluster2 = microCluster2->copy();
  }
  // bool operator==(const MicroClusterPair &other) const;
};

struct KeyHasher {
  std::size_t operator()(const MicroClusterPair &microClusterPair) const {
    return (std::hash<int>()(microClusterPair.microCluster1->id.front())) ^
           (std::hash<int>()(microClusterPair.microCluster2->id.front()));
  }
};

struct EqualKey {
  bool operator()(const MicroClusterPair &MCPair1,
                  const MicroClusterPair &MCPair2) const {
    if (MCPair1.microCluster1->id.front() ==
            MCPair2.microCluster1->id.front() &&
        MCPair1.microCluster2->id.front() == MCPair2.microCluster2->id.front())
      return true;
    if (MCPair1.microCluster1->id.front() ==
            MCPair2.microCluster2->id.front() &&
        MCPair1.microCluster2->id.front() == MCPair2.microCluster1->id.front())
      return true;
    return false;
  }
};

class AdjustedWeight;
typedef std::shared_ptr<AdjustedWeight> AdjustedWeightPtr;
class AdjustedWeight {
public:
  double weight;
  int updateTime = 0; //
  timespec updateTime0;
  AdjustedWeight(double weight, int pointTime,
                 timespec pointTime0); // clock_t pointTime
  void add(int startTime, double decayValue);
  void add(timespec startTime, double decayValue);
  double getCurrentWeight(double decayFactor);
};

typedef std::unordered_map<MicroClusterPair, AdjustedWeightPtr, KeyHasher,
                           EqualKey>
    WeightedAdjacencyList;
typedef std::pair<MicroClusterPair, AdjustedWeightPtr> DensityGraph;
} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_WEIGHTEDADJACENCYLIST_HPP_
