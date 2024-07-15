//
// Created by tuidan on 2021/8/24.
//
#include <Algorithm/Birch.hpp>
#include <Algorithm/DataStructure/DataStructureFactory.hpp>

void SESAME::Birch::Init() {
  this->cfTree = DataStructureFactory::createCFTree();
  this->cfTree->setB(BirchParam.max_in_nodes);
  this->cfTree->setL(BirchParam.max_leaf_nodes);
  this->cfTree->setT(BirchParam.distance_threshold);
  this->root = DataStructureFactory::createNode();
  this->root->setIsLeaf(true);
  sum_timer.Tick();
}

void SESAME::Birch::RunOnline(const SESAME::PointPtr input) {
  ds_timer.Tick();
  forwardInsert(input);
  ds_timer.Tock();
  lat_timer.Add(input->toa);
}

void SESAME::Birch::RunOffline(DataSinkPtr sinkPtr) {
  on_timer.Add(sum_timer.start);
  ref_timer.Tick();
  for (int i = 0; i < this->leafNodes.size(); i++) {
    PointPtr centroid = GenericFactory::New<Point>(BirchParam.dim, i);
    for (int j = 0; j < BirchParam.dim; j++) {
      centroid->setFeatureItem(this->leafNodes[i]->getCF()->getLS().at(j) /
                                   this->leafNodes[i]->getCF()->getN(),
                               j);
    }
    centroid->setClusteringCenter(i);
    sinkPtr->put(centroid);
  }
  ref_timer.Tock();
  sum_timer.Tock();
}

SESAME::Birch::Birch(param_t &cmd_params) {
  this->param = cmd_params;
  this->BirchParam.num_points = cmd_params.num_points;
  this->BirchParam.dim = cmd_params.dim;
  this->BirchParam.max_in_nodes = cmd_params.max_in_nodes;
  this->BirchParam.max_leaf_nodes = cmd_params.max_leaf_nodes;
  this->BirchParam.distance_threshold = cmd_params.distance_threshold;
}
SESAME::Birch::~Birch() {}
// when a new point insert into the CF, update the CF N, LS and SS
void SESAME::Birch::updateNLS(SESAME::NodePtr &node, SESAME::PointPtr &point,
                              bool updateAll) {
  SESAME::NodePtr nodeSearch = node;
  while (true) {
    SESAME::CFPtr cf = nodeSearch->getCF();
    vector<double> tmpLS = cf->getLS();
    vector<double> tmpSS = cf->getSS();
    if (tmpLS.empty()) {
      for (int i = 0; i < point->getDimension(); i++) {
        tmpLS.push_back(0);
        tmpSS.push_back(0);
      }
    }
    cf->setN(cf->getN() + 1);
    for (int i = 0; i < point->getDimension(); i++) {
      tmpLS[i] += point->getFeatureItem(i);
      tmpSS[i] += pow(point->getFeatureItem(i), 2);
    }
    cf->setLS(tmpLS);
    cf->setSS(tmpSS);
    if (nodeSearch->getParent() != nullptr && updateAll) {
      nodeSearch = nodeSearch->getParent();
    } else
      break;
  }
}

// centroid index: -1(virtual)
// centroid feature: mean of the feature of cluster points
// centroid cluster: -1
void SESAME::Birch::calculateCentroid(SESAME::CFPtr &cf,
                                      SESAME::PointPtr &centroid) {
  centroid->setIndex(-1);
  centroid->setClusteringCenter(-1);
  auto &ls = cf->LS;
  const auto n = cf->getN();
  for (int i = 0; i < n; ++i)
    centroid->feature[i] = ls[i] / n;
}

// use Manhattan Distance
void SESAME::Birch::pointToClusterDist(SESAME::PointPtr &insertPoint,
                                       SESAME::NodePtr &node, double &dist) {
  dist = 0;
  SESAME::PointPtr centroid = make_shared<SESAME::Point>(BirchParam.dim);
  SESAME::CFPtr curCF = node->getCF();
  calculateCentroid(curCF, centroid);
  dist = insertPoint->L1Dist(centroid);
}

// use Manhattan Distance
double SESAME::Birch::clusterToClusterDist(SESAME::NodePtr &nodeA,
                                           SESAME::NodePtr &nodeB) {
  SESAME::PointPtr centroidA = make_shared<SESAME::Point>(BirchParam.dim);
  SESAME::PointPtr centroidB = make_shared<SESAME::Point>(BirchParam.dim);
  SESAME::CFPtr curCFA = nodeA->getCF();
  SESAME::CFPtr curCFB = nodeB->getCF();
  calculateCentroid(curCFA, centroidA);
  calculateCentroid(curCFB, centroidB);
  return centroidA->L1Dist(centroidB);
}

// select the closest child cluster according to Manhattan Distance
void SESAME::Birch::selectChild(vector<SESAME::NodePtr> &children,
                                SESAME::PointPtr &insertPoint,
                                SESAME::NodePtr &node) {
  double dist = 0;
  double temp = 0;
  pointToClusterDist(insertPoint, children.at(0), dist);
  node = children.at(0);
  for (int i = 1; i < children.size(); i++) {
    pointToClusterDist(insertPoint, children.at(i), temp);
    if (temp < dist) {
      dist = temp;
      node = children.at(i);
    }
  }
}

// calculate the radius of a cluster
double SESAME::Birch::calculateRadius(SESAME::PointPtr &point,
                                      SESAME::PointPtr &centroid) {
  double denominator = 0;
  double radius = 0;
  for (int i = 0; i < point->getDimension(); i++) {
    denominator +=
        pow(centroid->getFeatureItem(i) - point->getFeatureItem(i), 2);
  }
  radius = sqrt(denominator);
  return radius;
}

void SESAME::Birch::calculateCorDistance(vector<vector<double>> &distance,
                                         vector<SESAME::NodePtr> &nodes) {
  const auto n = nodes.size();
  distance = vector<vector<double>>(n, vector<double>(n, 0));
  auto centroids = vector<SESAME::PointPtr>(n);
  for (int i = 0; i < n; i++) {
    centroids[i] = make_shared<SESAME::Point>(BirchParam.dim);
    auto cf = nodes[i]->getCF();
    calculateCentroid(cf, centroids[i]);
  }
  // calculate the correlate distance
  for (int i = 0; i < nodes.size(); i++) {
    SESAME::PointPtr centroidA = centroids[i];
    for (int j = i + 1; j < nodes.size(); j++) {
      SESAME::PointPtr centroidB = centroids[j];
      auto dist = centroidA->L1Dist(centroidB);
      distance[i][j] = dist;
      distance[j][i] = dist;
    }
  }
}

void SESAME::Birch::setCFToBlankNode(SESAME::NodePtr &curNode,
                                     SESAME::PointPtr &point) {
  SESAME::CFPtr curCF = curNode->getCF();
  curCF->setN(curCF->getN() + 1);
  vector<double> newLs;
  vector<double> newSs;
  for (int i = 0; i < point->getDimension(); i++) {
    newLs.push_back(point->getFeatureItem(i));
    newSs.push_back(pow(point->getFeatureItem(i), 2));
  }
  curCF->setSS(newSs);
  curCF->setLS(newLs);
}

void SESAME::Birch::addNodeNLSToNode(SESAME::NodePtr &child,
                                     SESAME::NodePtr &parent) {
  SESAME::CFPtr childCF = child->getCF();
  SESAME::CFPtr parCF = parent->getCF();
  parCF->setN(parCF->getN() + childCF->getN());
  vector<double> newLs;
  vector<double> newSs;
  for (int i = 0; i < childCF->getLS().size(); i++) {
    newLs.push_back(childCF->getLS().at(i) + parCF->getLS().at(i));
    newSs.push_back(childCF->getSS().at(i) + parCF->getSS().at(i));
  }
  parCF->setLS(newLs);
  parCF->setSS(newSs);
}

void SESAME::Birch::initializeCF(SESAME::CFPtr &cf, int dim) {
  vector<double> ls = cf->getLS();
  vector<double> ss = cf->getSS();
  for (int i = 0; i < dim; i++) {
    ls.push_back(0);
    ss.push_back(0);
  }
  cf->setLS(ls);
  cf->setSS(ss);
}

void SESAME::Birch::clearChildParents(vector<SESAME::NodePtr> &children) {
  for (auto child : children) {
    child->clearParents();
  }
}

void SESAME::Birch::forwardInsert(SESAME::PointPtr point) {
  NodePtr curNode = this->root;
  if (curNode->getCF()->getN() == 0) {
    updateNLS(curNode, point, true);
  } else {
    while (1) {
      vector<NodePtr> childrenNode = curNode->getChildren();
      if (curNode->getIsLeaf()) {
        CFPtr curCF = curNode->getCF();
        if (curCF->getN() == 0) {
          initializeCF(curCF, point->getDimension());
        }
        PointPtr centroid = make_shared<Point>(BirchParam.dim);
        calculateCentroid(curCF, centroid);
        if (calculateRadius(point, centroid) <=
            this->cfTree
                ->getT()) { // concept drift detection
                            // whether the new radius is lower than threshold T
          auto a = calculateRadius(point, centroid);
          if (point->getIndex() % 100 == 0) {
            std::cout << a;
          }
          updateNLS(curNode, point, true);

          // means this point could get included in this cluster
          // SESAME_DEBUG("No concept drift occurs(t <= T), insert tha point
          // into the leaf node...");
          break;
          // Normally insert the data point into the tree leafNode without
          // concept drift
        } else {
          auto a = calculateRadius(point, centroid);
          if (point->getIndex() % 100 == 0) {
            std::cout << a;
          }
          // concept drift adaption
          // SESAME_DEBUG("Concept drift occurs(t > T), the current leaf node
          // capacity reaches the threshold T");
          backwardEvolution(curNode, point);
          break;
        }
      } else {
        selectChild(childrenNode, point, curNode);
      }
    }
  }
}

// concept drift adaption
void SESAME::Birch::backwardEvolution(SESAME::NodePtr &curNode,
                                      SESAME::PointPtr &point) {
  if (curNode->getParent() == nullptr) { // means current node is root node
    // SESAME_DEBUG("l <= L, create a new leaf node and insert the point into
    // it(root change)");
    NodePtr newRoot = make_shared<CFNode>();
    newRoot->setIsLeaf(false);
    newRoot->setChild(curNode);
    curNode->setParent(newRoot);

    NodePtr newNode = make_shared<CFNode>();
    newNode->setIsLeaf(true);
    newNode->setParent(newRoot);
    vector<double> curLS = curNode->getCF()->getLS();
    vector<double> curSS = curNode->getCF()->getSS();
    int curN = curNode->getCF()->getN();
    newRoot->getCF()->setLS(curLS);
    newRoot->getCF()->setSS(curSS);
    newRoot->getCF()->setN(curN);
    newRoot->setIndex(this->leafMask++);
    // here we need to remove the old root and add the new one into the
    // leafnodes set
    this->leafNodes.push_back(newRoot);

    // update the parent node
    newRoot->setChild(newNode);
    updateNLS(newNode, point, true);
    this->root = newRoot;
  } else {
    NodePtr parent = curNode->getParent();
    NodePtr newNode = make_shared<CFNode>();
    newNode->setIsLeaf(true);
    newNode->setParent(parent);
    parent->setChild(newNode);
    updateNLS(newNode, point, false);
    if (parent->getChildren().size() < this->cfTree->getL()) {
      // whether the number of CFs(clusters) in the current leaf node is lower
      // thant threshold
      // L
      // SESAME_DEBUG("l <= L, create a new leaf node and insert the point into
      // it");

      // update the parent node
      updateNLS(parent, point, true);
    } else {
      // SESAME_DEBUG("l > L, parent node of the current leaf node capacity
      // reaches the threshold L"); SESAME_DEBUG("split a new parent node from
      // the old one ");
      bool CurNodeIsLeaf = true;
      while (true) {
        NodePtr parParent;
        if (parent->getParent() == nullptr) {
          parParent = make_shared<CFNode>();
          parParent->setIsLeaf(false);
          this->root = parParent;
          CFPtr parCF = parent->getCF();
          parParent->setCF(parCF);
        } else {
          parParent = parent->getParent();
          parParent->removeChild(parent);
        }
        NodePtr newParentA = make_shared<CFNode>();
        NodePtr newParentB = make_shared<CFNode>();
        if (parent->getChildren().at(0)->getIsLeaf()) {
          for (int i = 0; i < this->leafNodes.size(); i++) {
            if (this->leafNodes.at(i)->getIndex() == parent->getIndex()) {
              this->leafNodes.erase(this->leafNodes.begin() + i);
            }
          }
          newParentA->setIndex(++this->leafMask);
          newParentB->setIndex(++this->leafMask);
          this->leafNodes.push_back(newParentA);
          this->leafNodes.push_back(newParentB);
        }

        newParentB->setIsLeaf(false);
        newParentA->setIsLeaf(false);
        newParentB->setParent(parParent);
        newParentA->setParent(parParent);
        parParent->setChild(newParentB);
        parParent->setChild(newParentA);
        CFPtr cfA = newParentA->getCF();
        CFPtr cfB = newParentB->getCF();
        initializeCF(cfA, point->getDimension());
        initializeCF(cfB, point->getDimension());

        vector<NodePtr> broNodes = parent->getChildren();
        vector<vector<double>> corCFDistance;
        calculateCorDistance(corCFDistance, broNodes);

        // choose two farthest CFs as seedA and seedB
        int seedA = 0;
        int seedB = 0;
        double max = 0;
        for (int i = 0; i < broNodes.size(); i++) {
          for (int j = i; j < broNodes.size(); j++) {
            if (max < corCFDistance[i][j]) {
              seedA = i;
              seedB = j;
              max = corCFDistance[i][j];
            }
          }
        }

        // insert the child node into the nearest seed(A / B)
        clearChildParents(broNodes);
        // insert seedA node into parent A
        newParentA->setChild(broNodes[seedA]);
        broNodes[seedA]->setParent(newParentA);
        addNodeNLSToNode(broNodes[seedA], newParentA);
        // insert seedB node into parent B
        newParentB->setChild(broNodes[seedB]);
        broNodes[seedB]->setParent(newParentB);
        addNodeNLSToNode(broNodes[seedB], newParentB);
        // split other nodes into A and B
        for (int i = 0; i < broNodes.size(); i++) {
          if (i != seedA and i != seedB) {
            if (corCFDistance[i][seedA] < corCFDistance[i][seedB]) {
              newParentA->setChild(broNodes[i]);
              addNodeNLSToNode(broNodes[i], newParentA);
              broNodes[i]->clearParents();
              broNodes[i]->setParent(newParentA);
            } else {
              newParentB->setChild(broNodes[i]);
              addNodeNLSToNode(broNodes[i], newParentB);
              broNodes[i]->clearParents();
              broNodes[i]->setParent(newParentB);
            }
          }
        }
        if (CurNodeIsLeaf) {
          updateNLS(parParent, point, true);
        }

        if (parParent->getChildren().size() <= this->cfTree->getB()) {
          // SESAME_DEBUG("b < B, remove the old node and insert the new nodeA
          // and nodeB into the parent node");
          break;
        } else {
          // SESAME_DEBUG("b >= B, parent node of the current interior node
          // capacity reaches the threshold B");
          curNode = curNode->getParent();
          parent = parParent;
          CurNodeIsLeaf = false;
        }
      }
    }
  }
}
