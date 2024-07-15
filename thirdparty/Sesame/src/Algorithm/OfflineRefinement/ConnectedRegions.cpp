//
// Created by 1124a on 2021/11/23.
//

#include <Algorithm/OfflineRefinement/ConnectedRegions.hpp>
SESAME::ConnectedRegions::ConnectedRegions() {}
SESAME::ConnectedRegions::ConnectedRegions(double alpha, double min_weight) {
  this->alpha = alpha;
  this->min_weight = min_weight;
}
void SESAME::ConnectedRegions::connection(
    std::vector<MicroClusterPtr> &microClusters,

    SESAME::WeightedAdjacencyList &weightedAdjacencyList) {
  WeightedAdjacencyList::iterator iterW;
  for (iterW = weightedAdjacencyList.begin();
       iterW != weightedAdjacencyList.end(); iterW++) {
    //  std::cout<<" cluster 1 weight "<<iterW->first.microCluster1->weight
    //  <<", cluster 2 weight "<<iterW->first.microCluster2->weight<<"weight min
    //  is
    //  "<<min_weight<<std::endl;
    if (iterW->first.microCluster1->weight >= min_weight &&
        iterW->first.microCluster2->weight >= min_weight) {
      double val = 2 * iterW->second->weight /
                   (iterW->first.microCluster1->weight +
                    iterW->first.microCluster2->weight);
      if (val > min_weight) {
        insertIntoGraph(microClusters, iterW->first.microCluster1->id.front(),
                        iterW->first.microCluster2->id.front());

        insertIntoGraph(microClusters, iterW->first.microCluster2->id.front(),
                        iterW->first.microCluster1->id.front());
      } else {
        insertIntoGraph(microClusters, iterW->first.microCluster1->id.front());
        insertIntoGraph(microClusters, iterW->first.microCluster2->id.front());
      }
    }
  }
  findConnectedComponents(microClusters);
}
/**
 * @Description:  insert vertices and entries into connectivity graph when micro
 * cluster pair connectivity value greater than the intersection threshold if
 * the graph has testing micro cluster, add connected strong MC in the
 * corresponding entries else, create new V,E into the graph
 * @Param: connectivity graph, micro cluster 1 and 2
 * @Return: void
 */

void SESAME::ConnectedRegions::insertIntoGraph(
    const std::vector<MicroClusterPtr> &microClusters, int microClusterId,
    int OtherId) {
  if (connecvtivityGraphId.find(microClusterId) != connecvtivityGraphId.end()) {
    connecvtivityGraphId[microClusterId].push_back(OtherId);
  } else {
    auto microCluster = std::find_if(microClusters.begin(), microClusters.end(),
                                     [&](const MicroClusterPtr &mc) {
                                       return mc->id.front() == microClusterId;
                                     });
    (*microCluster)->visited = false;
    std::vector<int> newMicroClusterIdSet;
    newMicroClusterIdSet.push_back(OtherId);
    connecvtivityGraphId.insert(make_pair(microClusterId, OtherId));
  }
}

void SESAME::ConnectedRegions::insertIntoGraph(
    const std::vector<MicroClusterPtr> &microClusters, int microClusterId) {
  if (connecvtivityGraphId.find(microClusterId) == connecvtivityGraphId.end()) {
    // std::cerr << "INSERT micro cluster id is " << microClusterId <<
    // std::endl;
    auto microCluster = std::find_if(microClusters.begin(), microClusters.end(),
                                     [&](const MicroClusterPtr &mc) {
                                       return mc->id.front() == microClusterId;
                                     });
    (*microCluster)->visited = false;
    std::vector<int> newMicroClusterIdSet;
    connecvtivityGraphId.insert(
        make_pair(microClusterId, newMicroClusterIdSet));
  }
}

void SESAME::ConnectedRegions::findConnectedComponents(
    const std::vector<SESAME::MicroClusterPtr> &microClusters) {
  unordered_map<int, std::vector<int>>::iterator iter;
  // This variable just for indicating the id of micro cluster which forming
  // macro clusters
  for (iter = connecvtivityGraphId.begin(); iter != connecvtivityGraphId.end();
       iter++) {
    std::vector<int> idList;
    auto microClusterKey =
        std::find_if(microClusters.begin(), microClusters.end(),
                     [&](const MicroClusterPtr &mc) {
                       return mc->id.front() == iter->first;
                     });
    if (microClusterKey != microClusters.end() &&
        !(*microClusterKey)->visited) {
      std::vector<SESAME::MicroClusterPtr> newCluster;
      newCluster.push_back((*microClusterKey));
      idList.push_back(iter->first);
      for (int iterValue : iter->second) {
        auto microClusterElement =
            std::find_if(microClusters.begin(), microClusters.end(),
                         [&](const MicroClusterPtr &mc) {
                           return mc->id.front() == iterValue;
                         });
        if (microClusterElement != microClusters.end()) {
          if (!(*microClusterElement)->visited) {
            newCluster.push_back((*microClusterElement));
            (*microClusterElement)->visited = true;
            idList.push_back((*microClusterElement)->id.front());
          }
        }
      }
      this->finalClusters.push_back(newCluster);
    }
  }
}

std::vector<SESAME::PointPtr> SESAME::ConnectedRegions::ResultsToDataSink() {
  // SESAME_INFO("Start resize "<<finalClusters.size());

  std::vector<SESAME::PointPtr> points;
  for (auto iter = 0; iter != finalClusters.size();
       iter++) { // initialize pseudo point of macro clusters
    PointPtr point =
        GenericFactory::New<Point>(finalClusters.at(iter).front()->dim, iter);
    // This is just for testing, need to delete
    std::vector<double> centroid(finalClusters.at(iter).front()->dim, 0);
    // TODO maybe wrong ;so dizzy
    for (auto j = 0; j < finalClusters.at(iter).size(); j++) {
      double currentWeight =
          point->getWeight() + finalClusters.at(iter).at(j)->weight;
      point->setWeight(currentWeight);
      for (auto a = 0; a < finalClusters.at(iter).at(j)->dim; a++) {
        if (j == 0)
          point->setFeatureItem(0, a);
        point->setFeatureItem(point->getFeatureItem(a) +
                                  finalClusters.at(iter).at(j)->centroid.at(a),
                              a);
        centroid[a] = point->getFeatureItem(a); // testing
        if (j == finalClusters.at(iter).size() - 1) {
          point->setFeatureItem(
              point->getFeatureItem(a) / finalClusters.at(iter).at(j)->dim, a);
          centroid[a] =
              centroid[a] / finalClusters.at(iter).at(j)->dim; // testing
        }
      }
    }
    points.push_back(point);
    //  std::stringstream results;
    // std::copy(centroid.begin(),centroid.end(),std::ostream_iterator<double>(results,
    // " ")); SESAME_INFO("The NO."<<iter<<" Centroid is "<<results.str());
  }
  return points;
}
