// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 20/07/2021.
//

#include <Algorithm/DataStructure/DataStructureFactory.hpp>
#include <Algorithm/OfflineRefinement/KMeans.hpp>
#include <Algorithm/Param.hpp>
#include <Utils/Logger.hpp>

/**
 * @Description: First step: select k elements randomly in input as initial
 * clustering centers
 */

void SESAME::KMeans::randomSelectCenters(int numberOfCenters, int numberOfInput,
                                         std::vector<PointPtr> &input,
                                         std::vector<PointPtr> &centers) {
  // init_genrand(10);
  // put one point into the result

  std::vector<int> indexs; // for unique
  int id = rand() % numberOfInput;
  indexs.push_back(input[id]->getIndex());
  centers.push_back(input[id]->copy());
  int c = 1; // count the number of element in center
  // run the loop to insert the randomly selected indexes

  while (c < numberOfCenters) {
    id = rand() % numberOfInput;
    if (count(indexs.begin(), indexs.end(), id) == 0) {
      indexs.push_back(input[id]->getIndex());
      centers.push_back(input[id]->copy());
      c++;
    }
  }

  // print information
  if (indexs.size() != numberOfCenters)
    SESAME_INFO("ERROR!!! number of centers in indexs is not right!");
  // TODO : I commented out the following output
  /*
   * cout << "Randomly selected indexes:";
  for (int index : indexs) {
    cout << index << " ";
  }
  cout << endl;
   */
}

/**
 * @Description: select the k initial centers from the function determined by
 * point weight
 */

void SESAME::KMeans::selectCentersFromWeight(int numberOfCenters,
                                             int numberOfInput,
                                             std::vector<PointPtr> &input,
                                             std::vector<PointPtr> &centers) {
  std::vector<int> indexs; // for unique
  indexs.push_back(centers.at(0)->getIndex());
  // cout << "KMeans++ randomly select indexes: ";
  int times = 0;
  while (times < numberOfCenters) {
    double sum = 0;
    vector<PointPtr> leftOver;
    std::vector<double> weightSquare, probability;
    for (int i = 0; i < numberOfInput; i++) {
      if (count(indexs.begin(), indexs.end(), input.at(i)->getIndex()) == 0) {
        leftOver.push_back(input.at(i)->copy());
        double Min = calculateEluDistance(input.at(i), centers.at(0));
        // Min is D(x)
        for (int j = 1; j < centers.size(); j++) {
          if (Min > calculateEluDistance(input.at(i), centers.at(j))) {
            Min = calculateEluDistance(input.at(i), centers.at(j));
          }
        }
        weightSquare.push_back(pow(Min, 2));
        sum += pow(Min, 2);
        // here we only need to store D2.txt(x)
      }
    }

    // calculate the weight of every other point(except centers)
    for (double i : weightSquare) {
      probability.push_back(i / sum);
    }

    double left = 0, right = 0;
    double r = ((double)rand() / (RAND_MAX));
    for (int i = 0; i < leftOver.size(); i++) {
      right += probability.at(i);
      if (r > left && r <= right) {
        // cout << "randomly generate number: " << u(e) << endl;
        if (count(indexs.begin(), indexs.end(), leftOver.at(i)->getIndex()) ==
            0) {
          indexs.push_back(leftOver.at(i)->getIndex());
          centers.push_back(leftOver.at(i)->copy());
          // cout << leftOver.at(i)->getIndex() << " ";
          break;
        }
      }
      left = right;
    }
    times++;
  }
  // cout << endl;
}

/**
 * @Description: Calculate norm2 distance from point to one of the center
 */

double SESAME::KMeans::calculateEluDistance(PointPtr &point, PointPtr &center) {
  double dist = 0;
  for (int i = 0; i < point->getDimension(); i++) {
    dist += pow(point->getFeatureItem(i) - center->getFeatureItem(i), 2);
  }
  dist = sqrt(dist);
  return dist;
}

/**
 * @Description:  Calculate the new clustering center from groups
 */

void SESAME::KMeans::calculateClusterCenter(PointPtr &center,
                                            std::vector<PointPtr> &group) {
  for (int i = 0; i < center->getDimension(); i++) {
    double sum = 0;
    if (!group.empty()) {
      for (auto &j : group) {
        sum += j->getFeatureItem(i);
      }
      center->setFeatureItem(sum / double(group.size()), i);
    }
  }
}

/**
 * @Description: Second Step: group every input points to the nearest centers
 */

void SESAME::KMeans::groupPointsByCenters(
    int numberOfCenters, int numberOfInput, std::vector<PointPtr> &input,
    std::vector<PointPtr> &centers,
    std::vector<std::vector<PointPtr>> &groups) {
  for (int i = 0; i < centers.size(); i++) {
    std::vector<PointPtr> initial;
    groups.push_back(initial);
  }
  int Id;
  for (int i = 0; i < input.size(); i++) {
    double Min = calculateEluDistance(input.at(i), centers.at(0));
    Id = 0; // cluster_id that the point belongs to
    for (int j = 1; j < centers.size(); j++) {
      if (Min > calculateEluDistance(input.at(i), centers.at(j))) {
        Id = j;
        Min = calculateEluDistance(input.at(i), centers.at(j));
      }
    }
    groups[Id].push_back(input.at(i));
  }
}

/**
 * @Description: Third Step: choose new clustering center from group points
 * again
 */

void SESAME::KMeans::adjustClusteringCenters(
    std::vector<PointPtr> &centers,
    std::vector<std::vector<PointPtr>> &groups) {
  for (int i = 0; i < groups.size(); i++) {
    calculateClusterCenter(centers[i], groups[i]);
  }
}

/**
 * @Description: refresh group,replace the old one with the new one and clean up
 * the new group
 */

void SESAME::KMeans::refreshGroup(
    std::vector<std::vector<PointPtr>> &oldGroups,
    std::vector<std::vector<PointPtr>> &newGroups) {
  oldGroups.assign(newGroups.begin(), newGroups.end());
  newGroups.clear();
  std::vector<std::vector<PointPtr>> tmp;
  newGroups.swap(tmp);
}

/**
 * @Description: if the old groups equals the new ones, set flagToStop to true
 * to stop KMeans
 */

void SESAME::KMeans::checkStopStatus(
    bool &flag, std::vector<std::vector<PointPtr>> &oldGroups,
    std::vector<std::vector<PointPtr>> &newGroups) {
  flag = true;
  if (oldGroups.size() == newGroups.size()) {
    for (int i = 0; i < oldGroups.size(); i++) {
      if (oldGroups[i].size() == newGroups[i].size()) {
        for (int j = 1; j < oldGroups[i].size(); j++) {
          if (oldGroups[i][j]->getIndex() != newGroups[i][j]->getIndex()) {
            flag = false;
          }
        }
      } else {
        flag = false;
      }
    }
  } else {
    flag = false;
  }
  if (!flag) {
    // SESAME_INFO("Point cluster need to be adjust, start a new iteration!");
  }
}

/**
 * @Description: store the result into the output (clu_id)
 */

void SESAME::KMeans::storeResult(std::vector<std::vector<PointPtr>> &groups,
                                 std::vector<PointPtr> &output) {
  for (int i = 0; i < groups.size(); i++) {
    for (int j = 0; j < groups[i].size(); j++) {
      groups[i][j]->setClusteringCenter(i);
      output.push_back(groups[i][j]->copy()); // point index start from 0
    }
  }
}
void SESAME::KMeans::produceResult(
    std::vector<std::vector<SESAME::PointPtr>> &groups,
    SESAME::DataSinkPtr sinkPtr) {
  for (int i = 0; i < groups.size(); i++) {
    for (int j = 0; j < groups[i].size(); j++) {
      groups[i][j]->setClusteringCenter(i);
      sinkPtr->put(groups[i][j]->copy()); // point index start from 0
    }
  }
}

/**
 * @Description: if kmeanspp is true, run KMeans++, if false, run KMeans
 */

void SESAME::KMeans::runKMeans(int numberOfCenters, int numberOfInput,
                               std::vector<PointPtr> &centers,
                               std::vector<PointPtr> &input,
                               std::vector<std::vector<PointPtr>> &oldGroups,
                               std::vector<std::vector<PointPtr>> &newGroups,
                               int seed, bool kmeanspp) {
  bool flagToStop = false;
  srand(seed);
  if (kmeanspp) {
    // run the first step in KMeans++
    SESAME_INFO("KMeans++ start!!!");
    randomSelectCenters(1, numberOfInput, input, centers);
    int resetCenter = numberOfCenters - 1;
    selectCentersFromWeight(resetCenter, numberOfInput, input, centers);

    // run the second step in KMeans++
    groupPointsByCenters(numberOfCenters, numberOfInput, input, centers,
                         oldGroups);
  } else {
    SESAME_INFO("KMeans start!!!");
    // run the first step in KMeans
    randomSelectCenters(numberOfCenters, numberOfInput, input, centers);

    // run the second step in KMeans
    groupPointsByCenters(numberOfCenters, numberOfInput, input, centers,
                         oldGroups);
  }

  do {
    // run the third step in KMeans
    adjustClusteringCenters(centers, oldGroups);

    // run the second step in KMeans
    groupPointsByCenters(numberOfCenters, numberOfInput, input, centers,
                         newGroups);

    // check whether to stop
    checkStopStatus(flagToStop, oldGroups, newGroups);

    // refresh the groups, store newGroups in oldGroups and clean newGroups
    refreshGroup(oldGroups, newGroups);

  } while (!flagToStop);
  if (kmeanspp) {
    SESAME_INFO("KMeans++ sourceEnd!!!");
  } else {
    SESAME_INFO("KMeans sourceEnd!!!");
  }
}

void SESAME::KMeans::Run(SESAME::SesameParam &param,
                         vector<PointPtr> &online_centers,
                         SESAME::DataSinkPtr sinkPtr) {
  srand(param.seed);
  if (online_centers.size() <= param.k or param.k < 2) {
    int i = 0;
    for (auto el : online_centers) {
      el->setClusteringCenter(i++);
      sinkPtr->put(el);
    }
  } else {
    bool flagToStop = false;
    int numberOfCenters = param.k;
    int numberOfInput = (int)online_centers.size();
    std::vector<PointPtr> offlineCenters;
    std::vector<std::vector<PointPtr>> oldGroups, newGroups;

    if (param.kmeanspp) {
      // run the first step in KMeans++
      SESAME_INFO("KMeans++ start!!!");
      randomSelectCenters(1, numberOfInput, online_centers, offlineCenters);
      int resetCenter = numberOfCenters - 1;
      selectCentersFromWeight(resetCenter, numberOfInput, online_centers,
                              offlineCenters);

      // run the second step in KMeans++
      groupPointsByCenters(numberOfCenters, numberOfInput, online_centers,
                           offlineCenters, oldGroups);
    } else {
      SESAME_INFO("KMeans start!!!");
      // run the first step in KMeans
      randomSelectCenters(numberOfCenters, numberOfInput, online_centers,
                          offlineCenters);

      // run the second step in KMeans
      groupPointsByCenters(numberOfCenters, numberOfInput, online_centers,
                           offlineCenters, oldGroups);
    }

    do {
      // run the third step in KMeans
      adjustClusteringCenters(offlineCenters, oldGroups);

      // run the second step in KMeans
      groupPointsByCenters(numberOfCenters, numberOfInput, online_centers,
                           offlineCenters, newGroups);

      // check whether to stop
      checkStopStatus(flagToStop, oldGroups, newGroups);

      // refresh the groups, store newGroups in oldGroups and clean newGroups
      refreshGroup(oldGroups, newGroups);

    } while (!flagToStop);
    if (param.kmeanspp) {
      SESAME_INFO("KMeans++ sourceEnd!!!");
    } else {
      SESAME_INFO("KMeans sourceEnd!!!");
    }
    produceResult(oldGroups, sinkPtr);
  }
}

void SESAME::KMeans::Run(SESAME::SesameParam &param,
                         vector<PointPtr> &online_centers,
                         vector<PointPtr> &results) {
  srand(param.seed);
  if (online_centers.size() <= param.k or param.k < 2) {
    int i = 0;
    for (auto el : online_centers) {
      el->setClusteringCenter(i++);
      results.push_back(el);
    }
  } else {
    bool flagToStop = false;
    int numberOfCenters = param.k;
    int numberOfInput = (int)online_centers.size();
    std::vector<PointPtr> offlineCenters;
    std::vector<std::vector<PointPtr>> oldGroups, newGroups;

    if (param.kmeanspp) {
      // run the first step in KMeans++
      SESAME_INFO("KMeans++ start!!!");
      randomSelectCenters(1, numberOfInput, online_centers, offlineCenters);
      int resetCenter = numberOfCenters - 1;
      selectCentersFromWeight(resetCenter, numberOfInput, online_centers,
                              offlineCenters);

      // run the second step in KMeans++
      groupPointsByCenters(numberOfCenters, numberOfInput, online_centers,
                           offlineCenters, oldGroups);
    } else {
      SESAME_INFO("KMeans start!!!");
      // run the first step in KMeans
      randomSelectCenters(numberOfCenters, numberOfInput, online_centers,
                          offlineCenters);

      // run the second step in KMeans
      groupPointsByCenters(numberOfCenters, numberOfInput, online_centers,
                           offlineCenters, oldGroups);
    }

    do {
      // run the third step in KMeans
      adjustClusteringCenters(offlineCenters, oldGroups);

      // run the second step in KMeans
      groupPointsByCenters(numberOfCenters, numberOfInput, online_centers,
                           offlineCenters, newGroups);

      // check whether to stop
      checkStopStatus(flagToStop, oldGroups, newGroups);

      // refresh the groups, store newGroups in oldGroups and clean newGroups
      refreshGroup(oldGroups, newGroups);

    } while (!flagToStop);
    if (param.kmeanspp) {
      SESAME_INFO("KMeans++ sourceEnd!!!");
    } else {
      SESAME_INFO("KMeans sourceEnd!!!");
    }
    for (int i = 0; i < oldGroups.size(); i++) {
      for (int j = 0; j < oldGroups[i].size(); j++) {
        oldGroups[i][j]->setClusteringCenter(i);
        results.push_back(oldGroups[i][j]->copy()); // point index start from 0
      }
    }
  }
}
