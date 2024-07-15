// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 26/07/2021.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_TREENODE_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_TREENODE_HPP_

#include <Algorithm/DataStructure/Point.hpp>
#include <memory>
#include <vector>

namespace SESAME {

class TreeNode;
typedef std::shared_ptr<TreeNode> TreeNodePtr;

class TreeNode {
public:
  // number of points in this node
  int n;

  // array with pointers on points
  std::vector<PointPtr> points;

  // pointer on the centre of the treenode
  PointPtr centre;

  // pointer on the left childnode
  TreeNodePtr lc;

  // pointer on the right childnode
  TreeNodePtr rc;

  // pointer on the parent node
  TreeNodePtr parent;

  // cost of the treenode
  double cost;
};
} // namespace SESAME

#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_TREENODE_HPP_
