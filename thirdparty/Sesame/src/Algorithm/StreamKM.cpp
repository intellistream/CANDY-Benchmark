// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by tuidan on 2021/7/21.
//

#include <Algorithm/DataStructure/DataStructureFactory.hpp>
#include <Algorithm/DataStructure/Point.hpp>
#include <Algorithm/StreamKM.hpp>
#include <Algorithm/WindowModel/WindowFactory.hpp>
#include <Utils/Logger.hpp>
#include <Utils/UtilityFunctions.hpp>

void SESAME::StreamKM::Init() {
  // initial the landmark window
  UtilityFunctions::init_genrand(this->StreamKMParam.seed);
  this->window = WindowFactory::createLandmarkWindow();
  this->window->windowManager.numberOfWindow =
      ceil(log((double)this->StreamKMParam.num_points /
               (double)this->StreamKMParam.windowSize) /
           log(2)) +
      2;
  this->window->windowManager.maxWindowSize = this->StreamKMParam.windowSize;
  this->window->initWindow(this->StreamKMParam.windowSize);
  this->window->tree = GenericFactory::New<LandmarkWindow::CoresetTree>(param);
  SESAME_DEBUG("Created manager with "
               << this->window->windowManager.numberOfWindow
               << " windows of dim: " << this->StreamKMParam.dim);
  sum_timer.Tick();
}

/**
 * @Description: build the landmark Window, insert the data point and construct
 * the coreset tree if the window is full
 * @Param:
 * @Return: although void, but actually we store the output result(with computed
 * clustering center) into this->streamingCoreset
 */
void SESAME::StreamKM::RunOnline(const SESAME::PointPtr input) {
  ds_timer.Tick();
  this->window->insertPoint(input);
  ds_timer.Tock();
  lat_timer.Add(input->toa);
}

/**
 * @Description: we run offline KMeans++ algorithm 5 times using the final m
 * coreset points(this->streamingCoreset)[m>k]
 * @param num_clusters
 * @param coreset_size
 * @param dim
 * @param output
 */
void SESAME::StreamKM::RunOffline(DataSinkPtr sinkPtr) {
  on_timer.Add(sum_timer.start);
  ref_timer.Tick();
  this->window->getCoresetFromManager(
      this->streamingCoreset); // streamingCoreset =
                               // LandmarkWindow::getCoresetFromManager(manager);
  int parNumber = this->streamingCoreset.size();
  for (int i = 0; i < parNumber; i++) {
    for (int j = 0; j < this->StreamKMParam.dim; j++) {
      this->streamingCoreset[i]->setFeatureItem(
          this->streamingCoreset[i]->getFeatureItem(j) /
              this->streamingCoreset[i]->getWeight(),
          j);
    }
  }
  if (param.run_offline) {
    vector<PointPtr> centers;
    vector<vector<PointPtr>> groups;
    std::vector<std::vector<PointPtr>> oldGroups, newGroups;
    this->km.runKMeans(this->streamingCoreset.size(), parNumber, centers,
                       this->streamingCoreset, oldGroups, newGroups,
                       this->StreamKMParam.seed, true);
    // store the result input output
    this->km.produceResult(oldGroups, sinkPtr);
  } else {
    for (int i = 0; i < parNumber; i++) {
      streamingCoreset[i]->setClusteringCenter(i);
      sinkPtr->put(streamingCoreset[i]);
    }
  }
  ref_timer.Tock();
  sum_timer.Tock();
}

SESAME::StreamKM::StreamKM(param_t &cmd_params) {
  this->param = cmd_params;
  this->StreamKMParam.num_points = cmd_params.num_points;
  this->StreamKMParam.num_clusters = cmd_params.k;
  this->StreamKMParam.windowSize = cmd_params.coreset_size;
  this->StreamKMParam.seed = cmd_params.seed;
  this->StreamKMParam.dim = cmd_params.dim;
}
SESAME::StreamKM::~StreamKM() {}

// TODO: convert it as general tree model
void SESAME::LandmarkWindow::CoresetTree::unionTreeCoreset(
    int k, int n_1, int n_2, std::vector<PointPtr> &setA,
    std::vector<PointPtr> &setB, std::vector<PointPtr> &centres) {
  //  SESAME_DEBUG("Computing coreset...");
  // total number of points
  int n = n_1 + n_2;
  // choose the first centre (each point has the same probability of being
  // choosen) stores, how many centres have been choosen yet
  int choosenPoints = 0;
  // only choose from the n-i points not already choosen
  int j = UtilityFunctions::genrand_int31() % (n - choosenPoints);

  // copy the choosen point
  if (j < n_1) {
    centres[choosenPoints] = setA[j]->copy(); // TODO: ???? why re-set setA[j]?
  } else {
    j = j - n_1;
    centres[choosenPoints] = setB[j]->copy(); // TODO: ???? why re-set setB[j]?
  }
  //  struct treeNode *root = (struct treeNode *) malloc(sizeof(struct
  //  treeNode));
  TreeNodePtr root = DataStructureFactory::createTreeNode();
  constructRoot(root, setA, setB, n_1, n_2, centres[choosenPoints],
                choosenPoints);
  choosenPoints = 1;

  // choose the remaining points
  while (choosenPoints < k) {
    if (root->cost > 0.0) {
      TreeNodePtr leaf = selectNode(root);
      PointPtr centre = chooseCentre(leaf);
      split(leaf, centre, choosenPoints);
      centres[choosenPoints] = centre->copy();
    } else {
      // create a dummy point
      centres[choosenPoints] = root->centre->copy();
      int l;
      for (l = 0; l < centres[choosenPoints]->getDimension(); l++) {
        centres[choosenPoints]->setFeatureItem(-1 * 1000000, l);
      }
      centres[choosenPoints]->setIndex(-1);
      centres[choosenPoints]->setWeight(0.0);
    }

    choosenPoints++;
  }

  // free the tree
  freeTree(root);

  // recalculate clustering features
  int i;
  for (i = 0; i < n; i++) {
    if (i < n_1) {
      int index = setA[i]->getClusteringCenter();
      if (centres[index]->getIndex() != setA[i]->getIndex()) {
        centres[index]->setWeight(centres[index]->getWeight() +
                                  setA[i]->getWeight());
        int l;
        for (l = 0; l < setA[i]->getDimension(); l++) {
          if (setA[i]->getWeight() != 0.0) {
            centres[index]->setFeatureItem(
                setA[i]->getFeatureItem(l) + centres[index]->getFeatureItem(l),
                l);
          }
        }
      }
    } else {
      int index = setB[i - n_1]->getClusteringCenter();
      if (centres[index]->getIndex() != setB[i - n_1]->getIndex()) {
        centres[index]->setWeight(centres[index]->getWeight() +
                                  setB[i - n_1]->getWeight());
        int l;
        for (l = 0; l < setB[i - n_1]->getDimension(); l++) {
          if (setB[i - n_1]->getWeight() != 0.0) {
            centres[index]->setFeatureItem(
                setB[i - n_1]->getFeatureItem(l) +
                    centres[index]->getFeatureItem(l),
                l);
          }
        }
      }
    }
  }
}

void SESAME::LandmarkWindow::CoresetTree::freeTree(TreeNodePtr root) {
  while (!treeFinished(root)) {
    if (root->lc == NULL && root->rc == NULL) {
      root = root->parent;
    } else if (root->lc == NULL && root->rc != NULL) {
      // Schau ob rc ein Blatt ist
      if (isLeaf(root->rc)) {
        // Gebe rechtes Kind frei
        root->rc->points.clear();
        DataStructureFactory::clearTreeNode(root->rc);
        root->rc = NULL;
      } else {
        // Fahre mit rechtem Kind fort
        root = root->rc;
      }
    } else if (root->lc != NULL) {
      if (isLeaf(root->lc)) {
        root->lc->points.clear();
        DataStructureFactory::clearTreeNode(root->lc);
        root->lc = NULL;
      } else {
        root = root->lc;
      }
    }
  }
  root->points.clear();
  root.reset();
}

bool SESAME::LandmarkWindow::CoresetTree::treeFinished(TreeNodePtr root) {
  if (root->parent == NULL && root->lc == NULL && root->rc == NULL) {
    return 1;
  } else {
    return 0;
  }
}

bool SESAME::LandmarkWindow::CoresetTree::isLeaf(TreeNodePtr node) {
  if (node->lc == NULL && node->rc == NULL) {
    return 1;
  } else {
    return 0;
  }
}

void SESAME::LandmarkWindow::CoresetTree::constructRoot(
    TreeNodePtr root, std::vector<PointPtr> &setA, std::vector<PointPtr> &setB,
    int n_1, int n_2, PointPtr centre, int centreIndex) {
  // loop counter variable
  int i;
  // the root has no parent and no child nodes in the beginning
  root->parent = NULL;
  root->lc = NULL;
  root->rc = NULL;

  // array with points to the points
  // root->points= new Point[n_1 + n_2];
  root->n = n_1 + n_2;

  for (i = 0; i < root->n; i++) {
    if (i < n_1) {
      root->points.push_back(setA[i]);
      //      root->points[i] = setA[i];
      root->points[i]->setClusteringCenter(centreIndex);
    } else {
      root->points.push_back(setB[i - n_1]);
      root->points[i]->setClusteringCenter(centreIndex);
    }
  }

  // set the centre
  root->centre = centre;

  // calculate costs
  treeNodeTargetFunctionValue(root);
}

void SESAME::LandmarkWindow::CoresetTree::treeNodeTargetFunctionValue(
    TreeNodePtr node) {
  // loop counter variable
  int i;

  // stores the cost
  double sum = 0.0;

  for (i = 0; i < node->n; i++) {
    // stores the distance
    double distance = 0.0;

    // loop counter variable
    int l;

    for (l = 0; l < node->points[i]->getDimension(); l++) {
      // centroid coordinate of the point
      double centroidCoordinatePoint;
      if (node->points[i]->getWeight() != 0.0) {
        centroidCoordinatePoint =
            node->points[i]->getFeatureItem(l) / node->points[i]->getWeight();
      } else {
        centroidCoordinatePoint = node->points[i]->getFeatureItem(l);
      }
      // centroid coordinate of the centre
      double centroidCoordinateCentre;
      if (node->centre->getWeight() != 0.0) {
        centroidCoordinateCentre =
            node->centre->getFeatureItem(l) / node->centre->getWeight();
      } else {
        centroidCoordinateCentre = node->centre->getFeatureItem(l);
      }
      distance += (centroidCoordinatePoint - centroidCoordinateCentre) *
                  (centroidCoordinatePoint - centroidCoordinateCentre);
    }
    sum += distance * node->points[i]->getWeight();
  }
  node->cost = sum;
}

SESAME::TreeNodePtr
SESAME::LandmarkWindow::CoresetTree::selectNode(TreeNodePtr root) {
  // random number between 0 and 1
  double random = UtilityFunctions::genrand_real3();
  while (!isLeaf(root)) {
    if (root->lc->cost == 0 && root->rc->cost == 0) {
      if (root->lc->n == 0) {
        root = root->rc;
      } else if (root->rc->n == 0) {
        root = root->lc;
      } else if (random < 0.5) {
        random = UtilityFunctions::genrand_real3();
        root = root->lc;
      } else {
        random = UtilityFunctions::genrand_real3();
        root = root->rc;
      }
    } else {
      if (random < root->lc->cost / root->cost) {
        root = root->lc;
      } else {
        root = root->rc;
      }
    }
  }

  return root;
}

/**
 * selects a new centre from the treenode (using the kMeans++ distribution)
 * TODO: Why hard-code??
 */
SESAME::PointPtr
SESAME::LandmarkWindow::CoresetTree::chooseCentre(TreeNodePtr node) {
  // TODO: How many times should we try to choose a centre ??
  int times = 3;

  // stores the nodecost if node is split with the best centre
  double minCost = node->cost;
  PointPtr bestCentre = GenericFactory::New<Point>(param.dim);

  for (int j = 0; j < times; j++) {
    // sum of the relativ cost of the points
    double sum = 0.0;
    // random number between 0 and 1
    double random = UtilityFunctions::genrand_real3();

    for (int i = 0; i < node->n; i++) {
      sum += treeNodeCostOfPoint(node, node->points[i]) / node->cost;
      if (sum >= random) {
        if (node->points[i]->getWeight() == 0.0) {
          SESAME_INFO("ERROR: CHOOSEN DUMMY NODE THOUGH OTHER AVAILABLE \n");
          return bestCentre;
        }
        double curCost = treeNodeSplitCost(node, node->centre, node->points[i]);
        if (curCost < minCost) {
          bestCentre = node->points[i];
          minCost = curCost;
        }
        break;
      }
    }
  }
  if (bestCentre->getIndex() == -1) {
    return node->points[0];
  } else {
    return bestCentre;
  }
}

double
SESAME::LandmarkWindow::CoresetTree::treeNodeCostOfPoint(TreeNodePtr node,
                                                         PointPtr p) {
  if (p->getWeight() == 0.0) {
    return 0.0;
  }

  // stores the distance between centre and p
  double distance = 0.0;

  // loop counter variable
  int l;

  for (l = 0; l < p->getDimension(); l++) {
    // centroid coor->inate of the point
    double centroidCoordinatePoint;
    if (p->getWeight() != 0.0) {
      centroidCoordinatePoint = p->getFeatureItem(l) / p->getWeight();
    } else {
      centroidCoordinatePoint = p->getFeatureItem(l);
    }
    // centroid coordinate of the centre
    double centroidCoordinateCentre;
    if (node->centre->getWeight() != 0.0) {
      centroidCoordinateCentre =
          node->centre->getFeatureItem(l) / node->centre->getWeight();
    } else {
      centroidCoordinateCentre = node->centre->getFeatureItem(l);
    }
    distance += (centroidCoordinatePoint - centroidCoordinateCentre) *
                (centroidCoordinatePoint - centroidCoordinateCentre);
  }
  return distance * p->getWeight();
}

/**
 * computes the hypothetical cost if the node would be split with new centers
 * centreA, centreB
 * @param node
 * @param centreA
 * @param centreB
 * @return
 */
double SESAME::LandmarkWindow::CoresetTree::treeNodeSplitCost(
    TreeNodePtr node, PointPtr centreA, PointPtr centreB) {
  // loop counter variable
  int i;
  // stores the cost
  double sum = 0.0;
  for (i = 0; i < node->n; i++) {
    // loop counter variable
    int l;
    // stores the distance between p and centreA
    double distanceA = 0.0;
    for (l = 0; l < node->points[i]->getDimension(); l++) {
      // centroid coordinate of the point
      double centroidCoordinatePoint;
      if (node->points[i]->getWeight() != 0.0) {
        centroidCoordinatePoint =
            node->points[i]->getFeatureItem(l) / node->points[i]->getWeight();
      } else {
        centroidCoordinatePoint = node->points[i]->getFeatureItem(l);
      }
      // centroid coordinate of the centre
      double centroidCoordinateCentre;
      if (centreA->getWeight() != 0.0) {
        centroidCoordinateCentre =
            centreA->getFeatureItem(l) / centreA->getWeight();
      } else {
        centroidCoordinateCentre = centreA->getFeatureItem(l);
      }
      distanceA += (centroidCoordinatePoint - centroidCoordinateCentre) *
                   (centroidCoordinatePoint - centroidCoordinateCentre);
    }
    // stores the distance between p and centreB
    double distanceB = 0.0;
    for (l = 0; l < node->points[i]->getDimension(); l++) {
      // centroid coordinate of the point
      double centroidCoordinatePoint;
      if (node->points[i]->getWeight() != 0.0) {
        centroidCoordinatePoint =
            node->points[i]->getFeatureItem(l) / node->points[i]->getWeight();
      } else {
        centroidCoordinatePoint = node->points[i]->getFeatureItem(l);
      }
      // centroid coordinate of the centre
      double centroidCoordinateCentre;
      if (centreB->getWeight() != 0.0) {
        centroidCoordinateCentre =
            centreB->getFeatureItem(l) / centreB->getWeight();
      } else {
        centroidCoordinateCentre = centreB->getFeatureItem(l);
      }

      distanceB += (centroidCoordinatePoint - centroidCoordinateCentre) *
                   (centroidCoordinatePoint - centroidCoordinateCentre);
    }
    // add the cost of the closest centre to the sum
    if (distanceA < distanceB) {
      sum += distanceA * node->points[i]->getWeight();
    } else {
      sum += distanceB * node->points[i]->getWeight();
    }
  }
  // return the total cost
  return sum;
}

/**
splits the parent node and creates two child nodes (one with the old centre and
one with the new one)
**/
void SESAME::LandmarkWindow::CoresetTree::split(TreeNodePtr parent,
                                                PointPtr newCentre,
                                                int newCentreIndex) {
  // loop counter variable
  int i = 0;

  // 1. Counts how many points belong to the new and how many points belong to
  // the old centre
  int nOld = 0;
  int nNew = 0;
  for (i = 0; i < parent->n; i++) {
    PointPtr centre =
        determineClosestCentre(parent->points[i], parent->centre, newCentre);
    if (centre->getIndex() == newCentre->getIndex()) {
      nNew++;
    } else {
      nOld++;
    }
  }

  // 2. initalizes the arrays for the pointer
  // array for pointer on the points belonging to the old centre
  std::vector<PointPtr> oldPoints; // = new Point[nOld];
                                   //  for(i = 0; i < nOld; i++) {
                                   //
                                   //  }
  // array for pointer on the points belonging to the new centre
  std::vector<PointPtr> newPoints; //= new Point[nNew];

  int indexOld = 0;
  int indexNew = 0;

  for (i = 0; i < parent->n; i++) {
    PointPtr centre =
        determineClosestCentre(parent->points[i], parent->centre, newCentre);
    if (centre->getIndex() == newCentre->getIndex()) {
      newPoints.push_back(parent->points[i]);
      newPoints[indexNew]->setClusteringCenter(newCentreIndex);
      indexNew++;
    } else if (centre->getIndex() == parent->centre->getIndex()) {
      oldPoints.push_back(parent->points[i]);
      indexOld++;
    } else {
      SESAME_INFO("ERROR !!! NO CENTER NEAREST !! \n");
    }
  }

  // left child: old centre
  //  struct TreeNode *lc = (struct TreeNode *) malloc(sizeof(struct TreeNode));
  TreeNodePtr lc = DataStructureFactory::createTreeNode();
  lc->centre = parent->centre;
  lc->points = oldPoints;
  lc->n = nOld;

  lc->lc = NULL;
  lc->rc = NULL;
  lc->parent = parent;

  treeNodeTargetFunctionValue(lc);

  // right child: new centre
  //  struct TreeNode *rc = (struct TreeNode *) malloc(sizeof(struct TreeNode));
  TreeNodePtr rc = DataStructureFactory::createTreeNode();
  rc->centre = newCentre;
  rc->points = newPoints;
  rc->n = nNew;

  rc->lc = NULL;
  rc->rc = NULL;
  rc->parent = parent;

  treeNodeTargetFunctionValue(rc);

  // set childs of the parent node
  parent->lc = lc;
  parent->rc = rc;

  // propagate the cost changes to the parent nodes
  while (parent != NULL) {
    parent->cost = parent->lc->cost + parent->rc->cost;
    parent = parent->parent;
  }
}

/**
 * returns the next centre from A or B.
 * @param point
 * @param centreA
 * @param centreB
 * @return
 */
SESAME::PointPtr SESAME::LandmarkWindow::CoresetTree::determineClosestCentre(
    PointPtr point, PointPtr centreA, PointPtr centreB) {
  // loop counter variable
  int l;

  // stores the distance between p and centreA
  double distanceA = 0.0;

  for (l = 0; l < point->getDimension(); l++) {
    // centroid coordinate of the point
    double centroidCoordinatePoint;
    if (point->getWeight() != 0.0) {
      centroidCoordinatePoint = point->getFeatureItem(l) / point->getWeight();
    } else {
      centroidCoordinatePoint = point->getFeatureItem(l);
    }
    // centroid coordinate of the centre
    double centroidCoordinateCentre;
    if (centreA->getWeight() != 0.0) {
      centroidCoordinateCentre =
          centreA->getFeatureItem(l) / centreA->getWeight();
    } else {
      centroidCoordinateCentre = centreA->getFeatureItem(l);
    }

    distanceA += (centroidCoordinatePoint - centroidCoordinateCentre) *
                 (centroidCoordinatePoint - centroidCoordinateCentre);
  }

  // stores the distance between p and centreB
  double distanceB = 0.0;

  for (l = 0; l < point->getDimension(); l++) {
    // centroid coordinate of the point
    double centroidCoordinatePoint;
    if (point->getWeight() != 0.0) {
      centroidCoordinatePoint = point->getFeatureItem(l) / point->getWeight();
    } else {
      centroidCoordinatePoint = point->getFeatureItem(l);
    }
    // centroid coordinate of the centre
    double centroidCoordinateCentre;
    if (centreB->getWeight() != 0.0) {
      centroidCoordinateCentre =
          centreB->getFeatureItem(l) / centreB->getWeight();
    } else {
      centroidCoordinateCentre = centreB->getFeatureItem(l);
    }

    distanceB += (centroidCoordinatePoint - centroidCoordinateCentre) *
                 (centroidCoordinatePoint - centroidCoordinateCentre);
  }

  // return the nearest centre
  if (distanceA < distanceB) {
    return centreA;
  } else {
    return centreB;
  }
}