//
// Created by tuidan on 2021/8/24.
//

#include "Algorithm/DataStructure/CFTree.hpp"

#include <experimental/source_location>
#include <limits>
#include <queue>
#include <string_view>
#include <vector>

#include "Algorithm/DataStructure/GenericFactory.hpp"

using namespace std::experimental;

namespace SESAME {
CFTree::CFTree(const SesameParam &param)
    : max_in_nodes(param.max_in_nodes), max_leaf_nodes(param.max_leaf_nodes),
      distance_threshold(param.distance_threshold) {}

CFTree::CFTree(int b = 0, int l = 0, double t = 0.0)
    : max_in_nodes(b), max_leaf_nodes(l), distance_threshold(t) {}

CFTree::~CFTree() {}
int CFTree::getB() const { return this->max_in_nodes; }
int CFTree::getL() const { return this->max_leaf_nodes; }
double CFTree::getT() const { return this->distance_threshold; }
void CFTree::setB(int b) { this->max_in_nodes = b; }
void CFTree::setT(double t) { this->distance_threshold = t; }
void CFTree::setL(int l) { this->max_leaf_nodes = l; }

CFPtr CFNode::getCF() { return this->curCF; }
NodePtr CFNode::getParent() { return this->parent; }
int CFNode::getIndex() const { return this->index; }
std::vector<NodePtr> CFNode::getChildren() { return this->children; }

void CFNode::setNode(CFPtr &Node) { this->curCF = Node; }
void CFNode::setParent(NodePtr &Parent) { this->parent = Parent; }
void CFNode::setIndex(int Index) { this->index = Index; }
void CFNode::setChild(NodePtr &child) { this->children.push_back(child); }
void CFNode::setChildren(std::vector<NodePtr> children) {
  this->children = children;
}
CFNode::CFNode() {
  this->curCF = std::make_shared<CF>();
  this->isLeaf = true;
}
CFNode::~CFNode() {}
bool CFNode::getIsLeaf() { return this->isLeaf; }
void CFNode::setIsLeaf(bool leaf) { this->isLeaf = leaf; }
NodePtr CFNode::copy() { return std::make_shared<CFNode>(*this); }
void CFNode::setCF(CFPtr &cf) {
  this->curCF->setN(cf->getN());
  std::vector<double> ls = cf->getLS();
  std::vector<double> ss = cf->getSS();
  this->curCF->setLS(ls);
  this->curCF->setSS(ss);
}
// when need to clear the parent node, directly set the index = -1
void CFNode::clearParents() { this->parent->setIndex(-1); }
void CFNode::removeChild(NodePtr &child) {
  for (int i = 0; i < this->children.size(); i++) {
    if (this->children[i]->getIndex() == child->getIndex()) {
      auto childCF = this->children[i]->getCF();
      auto removeCF = child->getCF();
      bool flag = true;
      for (int j = 0; j < childCF->getLS().size(); j++) {
        if (childCF->getLS()[j] != removeCF->getLS()[j] or
            childCF->getSS()[j] != removeCF->getSS()[j]) {
          flag = false;
          break;
        }
      }
      if (flag) {
        this->children.erase(this->children.begin() + i);
      }
    }
  }
}
bool SESAME::CFNode::getOutlier() { return this->outlier; }
void SESAME::CFNode::setOutlier(bool flag) { this->outlier = flag; }

ClusteringFeaturesTree::~ClusteringFeaturesTree() {}

ClusteringFeaturesTree::ClusteringFeaturesTree(const SesameParam &param)
    : dim(param.dim), max_in_nodes(param.max_in_nodes),
      max_leaf_nodes(param.max_leaf_nodes),
      distance_threshold(param.distance_threshold) {
  root_ = GenericFactory::New<Node>(nullptr, param.dim);
  root_->index = leafMask++;
}

ClusteringFeaturesTree::NodePtr ClusteringFeaturesTree::Insert(PointPtr point) {
  auto curNode = root_;
  if (curNode->cf.num == 0) {
    curNode->Update(point, true);
    clusters_.push_back(curNode);
  } else {
    while (1) {
      if (curNode->IsLeaf()) {
        auto centroid = curNode->Centroid();
        if (point->L2Dist(centroid) <=
            distance_threshold) { // concept drift detection
          // whether the new radius is lower than threshold T
          curNode->Update(point, true);
          // means this point could get included in this cluster
          // SESAME_DEBUG("No concept drift occurs(t <= T), insert tha point
          // into the leaf node...");
          break;
          // Normally insert the data point into the tree leafNode without
          // concept drift
        } else {
          // concept drift adaption
          // SESAME_DEBUG("Concept drift occurs(t > T), the current leaf node
          // capacity reaches the threshold T");
          return backwardEvolution(curNode, point);
        }
      } else {
        curNode = CalcClosestNode(curNode->children, point).first;
      }
    }
  }
  return curNode;
}

ClusteringFeaturesTree::NodePtr ClusteringFeaturesTree::Insert(NodePtr node) {
  auto curNode = root_;
  auto center = node->Centroid();
  while (1) {
    auto childrenNode = curNode->children;
    if (curNode->IsLeaf()) {
      // timerMeter.clusterUpdateAccMeasure();
      // timerMeter.dataInsertAccMeasure();
      auto centroid = curNode->Centroid();
      // timerMeter.dataInsertEndMeasure();
      if (center->L2Dist(centroid) <=
          distance_threshold) { // concept drift detection
        // whether the new radius is lower than threshold T
        // timerMeter.dataInsertAccMeasure();
        curNode->Update(node, true);

        // means this point could get included in this cluster
        // SESAME_DEBUG("No concept drift occurs(t <= T), insert tha point
        // into the leaf node...");
        break;
        // Normally insert the data point into the tree leafNode without
        // concept drift
      } else {
        // concept drift adaption
        // SESAME_DEBUG("Concept drift occurs(t > T), the current leaf node
        // capacity reaches the threshold T");
        // timerMeter.clusterUpdateAccMeasure();
        return backwardEvolution(curNode, node);
        // timerMeter.clusterUpdateEndMeasure();
        // break;
      }
    } else {
      // timerMeter.dataInsertAccMeasure();
      curNode = CalcClosestNode(childrenNode, center).first;
      // timerMeter.dataInsertEndMeasure();
    }
  }
  return curNode;
}

void ClusteringFeaturesTree::Init() { root_->tree = shared_from_this(); }

void ClusteringFeaturesTree::Remove(NodePtr node) {
  auto parent = node->parent;
  if (parent != nullptr)
    parent->RemoveChild(node);
  const auto [first, last] = std::ranges::remove_if(
      clusters_, [node](auto &cluster) { return cluster == node; });
  clusters_.erase(first, last);
}

void ClusteringFeaturesTree::ForEach(std::function<void(NodePtr)> func) {
  std::queue<NodePtr> queue;
  queue.push(root_);
  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop();
    func(node);
    for (auto &child : node->children) {
      queue.push(child);
    }
  }
}

template <typename T>
ClusteringFeaturesTree::NodePtr
ClusteringFeaturesTree::backwardEvolution(NodePtr node, T input) {
  if (node->parent == nullptr) { // means current node is root node
    // l <= L, create a new leaf node and insert the point into it(root
    // change)
    auto newRoot = GenericFactory::New<Node>(shared_from_this(), dim);
    newRoot->AddChild(node);

    auto newNode = GenericFactory::New<Node>(shared_from_this(), dim);
    newRoot->AddChild(newNode);
    newRoot->cf = node->cf;
    newRoot->index = leafMask++;
    // here we need to remove the old root and add the new one into the
    // leafnodes set update the parent node
    newNode->Update(input, true);
    clusters_.push_back(newRoot);
    root_ = newRoot;
    return newNode;
  } else {
    auto parent = node->parent;
    auto newNode = GenericFactory::New<Node>(shared_from_this(), dim);
    parent->AddChild(newNode);
    newNode->Update(input, false);
    if (parent->children.size() < max_leaf_nodes) {
      // whether the number of CFs(clusters_) in the current leaf node is
      // lower thant threshold L l <= L, create a new leaf node and insert
      // the point into it update the parent node and all nodes on the path
      // to root node
      parent->Update(input, true);
    } else {
      // l > L, parent node of the current leaf node capacity reaches the
      // threshold L, split a new parent node from the old one
      bool nodeIsClus = true;
      while (true) {
        NodePtr parParent;
        if (parent->parent == nullptr) {
          // if the parent node is the root, we need to create a new root as
          // a parParent
          parParent = GenericFactory::New<Node>(shared_from_this(), dim);
          // parParent->children = root_->children;
          root_ = parParent;
          // since the parent node's nls has not been updated by the point,
          // so we directly copy the nls in parent node to the parParent one
          parParent->cf = parent->cf;
          parParent->index = leafMask++;
          parParent->AddChild(parent);
        } else {
          // if the parent node is not the root, we can get the parParent
          // one directly
          parParent = parent->parent;
        }
        // we need to create a new parent node since the old one has to
        // split
        auto newParentA = GenericFactory::New<Node>(shared_from_this(), dim);
        // insert the new parent into the allNode list
        // we also need to insert the new parent node into the clusterNode
        // list if its children is a leaf node.
        if (parent->children[0]->IsLeaf()) {
          newParentA->index = leafMask++;
          clusters_.push_back(newParentA);
        }
        // we only create a new parent rather and keep the old parent node
        // as the split two sub-nodes so we need to refresh the old parent
        // node as a blank one and treat it as a new parent B

        // parent->parent = parParent; // link the parparent node and the new
        //                             // created new parent A
        parParent->AddChild(newParentA);
        // clean cf of the old parent node and initialize the cf of new
        // parent A (ls and ss all have d number of 0)
        // split the child nodes of the old parent nodes
        std::vector<NodePtr> broNodes;
        parent->children.swap(broNodes);
        auto adjMatrix =
            CalcAdjMatrix(broNodes); //  calculate the distance between
                                     //  each two brother nodes
        // choose two farthest CFs as seedA and seedB
        int seedA = 0, seedB = 0;
        double maxDis = 0;
        for (int i = 0; i < broNodes.size(); i++) {
          for (int j = i; j < broNodes.size(); j++) {
            if (maxDis < adjMatrix[i][j]) {
              seedA = i, seedB = j;
              maxDis = adjMatrix[i][j];
            }
          }
        }
        // insert the child node into the nearest seed(A / B)
        for (auto node : broNodes) {
          node->ClearParents();
        }
        // insert seedA node into new parent A and link them
        newParentA->AddChild(broNodes[seedA]);
        newParentA->Update(broNodes[seedA]);
        // insert seed B node into new parent B and link them
        parent->AddChild(broNodes[seedB]);
        parent->Update(broNodes[seedB]);
        // if other one brother node is near seed A then split it into new
        // parent A, otherwise new parent B.
        for (int i = 0; i < broNodes.size(); i++) {
          if (i != seedA && i != seedB) {
            if (adjMatrix[i][seedA] < adjMatrix[i][seedB]) {
              newParentA->AddChild(broNodes[i]);
              newParentA->Update(
                  broNodes[i]); // since the brother nodes list
                                // contains the one we insert our
                                // point, so after this function, the
                                // parent node's nls are also updated.
            } else {
              parent->AddChild(broNodes[i]);
              parent->Update(broNodes[i]);
            }
          }
        }
        // if the current node(parent) is a cluster nodes, then we need to
        // update the nls of its parent using new point. we only update the
        // parparent in the first loop.
        if (nodeIsClus) {
          parParent->Update(input, true);
        }

        if (parParent->children.size() <= max_in_nodes) {
          // b < B, remove the old node and insert the new nodeA and nodeB
          // into the parent node.
          break;
        } else {
          // b >= B, parent node of the current interior node capacity
          // reaches the threshold B.
          node = node->parent;
          parent = parParent;
          nodeIsClus = false;
        }
      }
    }
    return newNode;
  }
}

std::vector<ClusteringFeaturesTree::NodePtr> &
ClusteringFeaturesTree::clusters() {
  return clusters_;
}

std::string ClusteringFeaturesTree::Serialize() {
  std::deque<NodePtr> q;
  std::deque<int> dep;
  std::string s;
  q.push_back(root_);
  dep.push_back(0);
  while (!q.empty()) {
    auto x = q.front();
    auto d = dep.front();
    q.pop_front();
    dep.pop_front();
    s += x->Serialize(d);
    for (auto child : x->children) {
      q.push_front(child);
      dep.push_front(d + 1);
    }
  }
  return s;
}

ClusteringFeaturesList::ClusteringFeaturesList(const SesameParam &param)
    : dim(param.dim), distance_threshold(param.distance_threshold) {}

ClusteringFeaturesList::~ClusteringFeaturesList() {}

ClusteringFeaturesList::NodePtr ClusteringFeaturesList::Insert(PointPtr point) {
  if (clusters_.empty()) {
    auto node = GenericFactory::New<Node>(point);
    clusters_.push_back(node);
    return node;
  } else {
    auto [node, dist] = CalcClosestNode(clusters_, point);
    if (dist >= distance_threshold) {
      node = GenericFactory::New<Node>(dim);
      clusters_.push_back(node);
    }
    node->Update(point);
    return node;
  }
}

ClusteringFeaturesList::NodePtr ClusteringFeaturesList::Insert(NodePtr node) {
  clusters_.push_back(node);
  return node;
}

std::vector<ClusteringFeaturesList::NodePtr> &
ClusteringFeaturesList::clusters() {
  return clusters_;
}

void ClusteringFeaturesList::Remove(NodePtr node) {
  auto it = std::find(clusters_.begin(), clusters_.end(), node);
  if (it != clusters_.end()) {
    clusters_.erase(it);
  }
}

} // namespace SESAME
